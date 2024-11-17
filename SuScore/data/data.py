import pandas as pd
import numpy as np
import torch as th
from rdkit import Chem
import os
import tempfile
import shutil
from joblib import Parallel, delayed
from torch_geometric.data import Batch, Data, Dataset  # , InMemoryDataset
from torch.utils.data import DataLoader
from ..feats.derivate.split_mol2radius import prot_to_graph, mol_to_graph, load_mol, mol_to_radiusg
# from ..feats.init_mol2graph import prot_to_graph, mol_to_graph, load_mol
#from ..feats.extract_pocket_prody import extract_pocket
import logging
import warnings
import pickle
logging.basicConfig(level=logging.INFO)


# from ..feats.extract_pocket_prody import extract_pocket

class PDBbindDataset(Dataset):
    def __init__(self,
                 ids=None,
                 ligs=None,
                 prots=None,
                 labels=None,
                 ):
        super(PDBbindDataset, self).__init__()
        self.labels = labels
        if isinstance(ids, np.ndarray) or isinstance(ids, list):
            self.pdbids = ids
        else:
            try:
                self.pdbids = np.load(ids)
                # print(ids)
                # print(self.pdbids)
            except:
                raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
            if self.pdbids.shape[0] == 1:
                pass
            elif self.pdbids.shape[0] == 2:
                self.labels = self.pdbids[-1].astype(float) * (-1.36)
                # self.labels = self.pdbids[-1].astype(float)
                self.pdbids = self.pdbids[0]
            else:
                raise ValueError('the file to store numpy.ndarray should have one/two dimensions')

        if isinstance(ligs, np.ndarray) or isinstance(ligs, tuple) or isinstance(ligs, list):
            if isinstance(ligs[0], Data):
                self.gls = ligs
            else:
                raise ValueError(
                    'the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')
        else:
            try:
                self.gls = th.load(ligs)
            except:
                raise ValueError(
                    'the variable "ligs" should be a set of (or a file to store) torch_geometric.data.Data objects.')

        if isinstance(prots, np.ndarray) or isinstance(prots, th.Tensor) or isinstance(prots, list):
            if isinstance(prots[0], Data):
                self.gps = prots
            else:
                raise ValueError(
                    'the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')
        else:
            try:
                self.gps = th.load(prots)
            except:
                raise ValueError(
                    'the variable "prots" should be a set of (or a file to store) torch_geometric.data.Data objects.')

        self.gls = Batch.from_data_list(self.gls)
        self.gps = Batch.from_data_list(self.gps)
        assert len(self.pdbids) == self.gls.num_graphs == self.gps.num_graphs
        if self.labels is None:
            self.labels = th.zeros(len(self.pdbids))
        else:
            self.labels = th.tensor(self.labels)

    def len(self):
        return len(self.pdbids)

    def get(self, idx):
        pdbid = self.pdbids[idx]
        gp = self.gps[idx]
        gl = self.gls[idx]
        label = self.labels[idx]
        return pdbid, gp, gl, label

    def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.pdbids))
        val_inds = np.random.choice(np.arange(len(self.pdbids)), valnum, replace=False)
        train_inds = np.setdiff1d(np.arange(len(self.pdbids)), val_inds)
        return train_inds, val_inds


class VSDataset(Dataset):
    def __init__(self,
                 ids=None,
                 ligs=None,
                 prot=None,
                 labels=None,
                 gen_pocket=False,
                 cutoff=None,
                 reflig=None,
                 explicit_H=False,
                 use_chirality=True,
                 parallel=True
                 ):
        super(VSDataset, self).__init__()
        self.labels = labels
        self.gp = None
        self.gls = None
        self.pocketdir = None
        self.prot = None
        self.ligs = None
        self.cutoff = cutoff
        self.explicit_H = explicit_H
        self.use_chirality = use_chirality
        self.parallel = parallel

        if isinstance(prot, Chem.rdchem.Mol):
            assert gen_pocket == False
            self.prot = prot
            self.gp = mol_to_radiusg(self.prot, cutoff)
        else:
            if gen_pocket:
                if cutoff is None or reflig is None:
                    raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
                try:
                    self.pocketdir = tempfile.mkdtemp()
                    extract_pocket(prot, reflig, cutoff,
                                   protname="temp",
                                   workdir=self.pocketdir)
                    pocket = load_mol("%s/temp_pocket_%s.pdb" % (self.pocketdir, cutoff),
                                      explicit_H=explicit_H, use_chirality=use_chirality)
                    self.prot = pocket
                    self.gp = mol_to_radiusg(self.prot, cutoff)
                except:
                    raise ValueError('The graph of pocket cannot be generated')
            else:
                try:
                    pocket = load_mol(prot, explicit_H=explicit_H, use_chirality=use_chirality)
                    # self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)
                    self.prot = pocket
                    self.gp = mol_to_radiusg(self.prot, explicit_H=explicit_H)
                except:
                    raise ValueError('The graph of pocket cannot be generated')

        if isinstance(ligs, np.ndarray) or isinstance(ligs, list):
            if isinstance(ligs[0], Chem.rdchem.Mol):
                self.ligs = ligs
                self.gls = self._mol_to_graph()
            elif isinstance(ligs[0], Data):
                self.gls = ligs
            else:
                raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
        else:
            if ligs.endswith(".mol2"):
                lig_blocks = self._mol2_split(ligs)
                self.ligs = [Chem.MolFromMol2Block(lig_block) for lig_block in lig_blocks]
                self.gls = self._mol_to_graph()
            elif ligs.endswith(".sdf"):
                lig_blocks = self._sdf_split(ligs)
                self.ligs = [Chem.MolFromMolBlock(lig_block) for lig_block in lig_blocks]
                self.gls = self._mol_to_graph()
            else:
                try:
                    self.gls, _ = load_graphs(ligs)
                except:
                    raise ValueError(
                        'Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')

        if ids is None:
            if self.ligs is not None:
                self.idsx = ["%s-%s" % (self.get_ligname(lig), i) for i, lig in enumerate(self.ligs)]
            else:
                self.idsx = ["lig%s" % i for i in range(len(self.gls))]
        else:
            self.idsx = ids

        self.ids, self.gls = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.gls)))
        self.ids = list(self.ids)
        self.gls = Batch.from_data_list(self.gls)
        assert len(self.ids) == self.gls.num_graphs
        if self.labels is None:
            self.labels = th.zeros(len(self.ids))
        else:
            self.labels = th.tensor(self.labels)

        if self.pocketdir is not None:
            shutil.rmtree(self.pocketdir)

    def len(self):
        return len(self.ids)

    def get(self, idx):
        id = self.ids[idx]
        gp = self.gp
        gl = self.gls[idx]
        label = self.labels[idx]
        return id, gp, gl, label

    def _mol2_split(self, infile):
        contents = open(infile, 'r').read()
        return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]

    def _sdf_split(self, infile):
        contents = open(infile, 'r').read()
        return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]

    def _mol_to_graph0(self, lig):
        try:
            gx = mol_to_graph(lig, explicit_H=self.explicit_H, use_chirality=self.use_chirality)
        except:
            print("failed to scoring for {} and {}".format(self.gp, lig))
            return None
        return gx

    def _mol_to_graph(self):
        if self.parallel:
            return Parallel(n_jobs=-1, backend="threading")(delayed(self._mol_to_graph0)(lig) for lig in self.ligs)
        else:
            graphs = []
            for lig in self.ligs:
                graphs.append(self._mol_to_graph0(lig))
            return graphs

    def get_ligname(self, m):
        if m is None:
            return None
        else:
            if m.HasProp("_Name"):
                return m.GetProp("_Name")
            else:
                return None


class HLDataset(Dataset):
    def __init__(self,
                 ids=None,
                 datadir=None,
                 labels=None,
                 ):
        super(HLDataset, self).__init__()
        self.labels = labels
        self.datadir = datadir
        if isinstance(ids, np.ndarray) or isinstance(ids, list):
            self.pdbids = ids
        else:
            try:
                self.pdbids = np.load(ids)
            except:
                raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
            if self.pdbids.shape[0] == 1:
                pass
            elif self.pdbids.shape[0] == 2:
                self.labels = self.pdbids[-1].astype(float)
                self.pdbids = self.pdbids[0]
            else:
                raise ValueError('the file to store numpy.ndarray should have one/two dimensions')

        # self.gls = Batch.from_data_list(self.gls)
        # self.gps = Batch.from_data_list(self.gps)
        # assert len(self.pdbids) == self.gls.num_graphs == self.gps.num_graphs

    def len(self):
        return len(self.pdbids)

    def get(self, idx):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                pdbid = self.pdbids[idx]
                # base_name = pdbid.split('/')[-1]
                prot_path = os.path.join(self.datadir, pdbid, pdbid + '_prot.pt')
                lig_path = os.path.join(self.datadir, pdbid, pdbid + '_lig.pt')
                gp = th.load(prot_path)
                gl = th.load(lig_path)
                gp.edge_index = th.tensor(gp.edge_index, dtype=th.long)
                gl.edge_index = th.tensor(gl.edge_index, dtype=th.long)
                label = self.labels[idx] * (-10)
                return pdbid, gp, gl, label
        except:
            logging.info(f"Variable value: {prot_path}")



    def collate_fn(self, batch):

        pdbids, gps, gls, labels = zip(*batch)
        gps_batch = Batch.from_data_list(gps)
        gls_batch = Batch.from_data_list(gls)
        return list(pdbids), gps_batch, gls_batch, th.tensor(np.array(list(labels)))

    def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.pdbids))
        val_inds = np.random.choice(np.arange(len(self.pdbids)), valnum, replace=False)
        remaining_inds = np.setdiff1d(np.arange(len(self.pdbids)), val_inds)
        test_inds = np.random.choice(remaining_inds, 200000, replace=False)
        train_inds = np.setdiff1d(remaining_inds, test_inds)
        return train_inds, val_inds


class PLIDataLoader(DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=dataset.collate_fn, **kwargs)


class LITDataset(Dataset):
    def __init__(self,
                 ids=None,
                 datadir=None,
                 labels=None,
                 ):
        super(LITDataset, self).__init__()
        self.labels = labels
        self.datadir = datadir
        if isinstance(ids, np.ndarray) or isinstance(ids, list):
            self.pdbids = ids
        else:
            try:
                self.pdbids = np.load(ids, allow_pickle=True)
            except:
                raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
            if self.pdbids.shape[0] == 1:
                pass
            elif self.pdbids.shape[0] == 2:
                self.labels = self.pdbids[-1].astype(float) * (-1.36)
                self.pdbids = self.pdbids[0]
            else:
                raise ValueError('the file to store numpy.ndarray should have one/two dimensions')

        # self.gls = Batch.from_data_list(self.gls)
        # self.gps = Batch.from_data_list(self.gps)
        # assert len(self.pdbids) == self.gls.num_graphs == self.gps.num_graphs

    def len(self):
        return len(self.pdbids)

    def get(self, idx):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                pdbid = self.pdbids[idx]
                # base_name = pdbid.split('/')[-1]
                complex_path = os.path.join(self.datadir, pdbid)
                label = self.labels[idx]
                with open(complex_path, "rb") as f:
                    lig, prot = pickle.load(f)

                gp = mol_to_radiusg(prot, explicit_H=False, use_chirality=True)
                gl = mol_to_graph(lig, explicit_H=False, use_chirality=True)
                gp.edge_index = th.tensor(gp.edge_index, dtype=th.long)
                gl.edge_index = th.tensor(gl.edge_index, dtype=th.long)
                return pdbid, gp, gl, label
        except:
            logging.info(f"Variable value: {complex_path}")



    def collate_fn(self, batch):

        pdbids, gps, gls, labels = zip(*batch)
        gps_batch = Batch.from_data_list(gps)
        gls_batch = Batch.from_data_list(gls)
        return list(pdbids), gps_batch, gls_batch, th.tensor(np.array(list(labels)))

    def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.pdbids))
        val_inds = np.random.choice(np.arange(len(self.pdbids)), valnum, replace=False)
        train_inds = np.setdiff1d(np.arange(len(self.pdbids)), val_inds)
        return train_inds, val_inds

class LINTDataset(Dataset):
    def __init__(self,
                 ids=None,
                 datadir=None,
                 prot=None,
                 labels=None,
                 ):
        super(LINTDataset, self).__init__()
        self.labels = labels
        self.datadir = datadir

        pocket = load_mol(prot, explicit_H=False, use_chirality=True)
        # self.graphp = mol_to_graph(pocket, explicit_H=explicit_H, use_chirality=use_chirality)
        self.prot = pocket
        self.gp = mol_to_radiusg(self.prot, explicit_H=False)
        # self.gp = prot_to_graph(self.prot, 10.0)

        if isinstance(ids, np.ndarray) or isinstance(ids, list):
            self.pdbids = ids
        else:
            try:
                self.pdbids = np.load(ids, allow_pickle=True)
            except:
                raise ValueError('the variable "ids" should be numpy.ndarray or list or a file to store numpy.ndarray')
            if self.pdbids.shape[0] == 1:
                pass
            elif self.pdbids.shape[0] == 2:
                self.labels = self.pdbids[-1].astype(float)
                self.pdbids = self.pdbids[0]
            else:
                raise ValueError('the file to store numpy.ndarray should have one/two dimensions')

        # self.gls = Batch.from_data_list(self.gls)
        # self.gps = Batch.from_data_list(self.gps)
        # assert len(self.pdbids) == self.gls.num_graphs == self.gps.num_graphs

    def len(self):
        return len(self.pdbids)

    def get(self, idx):
        # try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            pdbid = self.pdbids[idx]
            # base_name = pdbid.split('/')[-1]
            lig_path = os.path.join(self.datadir, pdbid + '.mol2')
            label = self.labels[idx]
            lig = load_mol(lig_path, explicit_H=False, use_chirality=True)

            gp = self.gp
            gl = mol_to_graph(lig, explicit_H=False, use_chirality=True)
            gp.edge_index = th.tensor(gp.edge_index, dtype=th.long)
            gl.edge_index = th.tensor(gl.edge_index, dtype=th.long)
            return pdbid, gp, gl, label
        # except:
        #     logging.info(f"Variable value: {lig_path}")

    def collate_fn(self, batch):

        pdbids, gps, gls, labels = zip(*batch)
        gps_batch = Batch.from_data_list(gps)
        gls_batch = Batch.from_data_list(gls)
        return list(pdbids), gps_batch, gls_batch, th.tensor(np.array(list(labels)))

    def train_and_test_split(self, valfrac=0.2, valnum=None, seed=0):
        # random.seed(seed)
        np.random.seed(seed)
        if valnum is None:
            valnum = int(valfrac * len(self.pdbids))
        val_inds = np.random.choice(np.arange(len(self.pdbids)), valnum, replace=False)
        train_inds = np.setdiff1d(np.arange(len(self.pdbids)), val_inds)
        return train_inds, val_inds
