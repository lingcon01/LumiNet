import pandas as pd
import numpy as np
from rdkit import Chem
import torch as th
import re, os
from itertools import permutations
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from joblib import Parallel, delayed
from rdkit.Chem import AllChem
# import MDAnalysis as mda
# from MDAnalysis.analysis import dihedrals
# from MDAnalysis.analysis import distances
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 24


def prot_to_graph(prot, cutoff):
    """obtain the residue graphs"""
    u = mda.Universe(prot)
    # Add nodes
    num_residues = len(u.residues)

    res_feats = np.array([calc_res_features(res) for res in u.residues])
    edgeids, distm = obatin_edge(u, cutoff)
    src_list, dst_list = zip(*edgeids)

    ca_pos = th.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    center_pos = th.tensor(u.atoms.center_of_mass(compound='residues'))
    dis_matx_ca = distance_matrix(ca_pos, ca_pos)
    cadist = th.tensor([dis_matx_ca[i, j] for i, j in edgeids]) * 0.1
    dis_matx_center = distance_matrix(center_pos, center_pos)
    cedist = th.tensor([dis_matx_center[i, j] for i, j in edgeids]) * 0.1
    edge_connect = th.tensor(np.array([check_connect(u, x, y) for x, y in edgeids]))
    edge_feats = th.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), th.tensor(distm)], dim=1)

    # res_max_natoms = max([len(res.atoms) for res in u.residues])
    res_coods = th.tensor(np.array(
        [np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS - len(res.atoms), 3), np.nan)], axis=0) for res in
         u.residues]))
    g = Data(x=th.tensor(res_feats, dtype=th.float),
             edge_index=th.tensor([src_list, dst_list]),
             pos=res_coods,
             edge_attr=th.tensor(np.array(edge_feats), dtype=th.float))

    # g.ndata.pop("ca_pos")
    # g.ndata.pop("center_pos")
    # g.ndata["pos"] = th.tensor(np.array([np.concatenate([res.atoms.positions, np.full((RES_MAX_NATOMS-len(res.atoms), 3), np.nan)],axis=0) for res in u.residues]))
    return g


def obtain_ca_pos(res):
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except:  ##some residues loss the CA atoms
            return res.atoms.positions.mean(axis=0)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
    try:
        # xx = res.atoms.select_atoms("not name H*")
        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except:
        return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except:
        return [0, 0, 0, 0]


def calc_res_features(res):
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +  # 32  residue type
                    obtain_self_dist(res) +  # 5
                    obtain_dihediral_angles(res)  # 4
                    )


def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname


##'FE', 'SR', 'GA', 'IN', 'ZN', 'CU', 'MN', 'SR', 'K' ,'NI', 'NA', 'CD' 'MG','CO','HG', 'CS', 'CA',

def obatin_edge(u, cutoff=10.0):
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T


def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0


def calc_dist(res1, res2):
    # xx1 = res1.atoms.select_atoms('not name H*')
    # xx2 = res2.atoms.select_atoms('not name H*')
    # dist_array = distances.distance_array(xx1.positions,xx2.positions)
    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


# return dist_array.max()*0.1, dist_array.min()*0.1


def calc_atom_features(atom, explicit_H=False):
    """
    atom: rdkit.Chem.rdchem.Atom
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'C', 'N', 'O', 'S', 'F', 'P', 'Cl',
            'Br', 'I', 'B', 'Si', 'Fe', 'Zn',
            'Cu', 'Mn', 'Mo', 'other'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    # [atom.GetIsAromatic()] # set all aromaticity feature blank.
    # In case of explicit hydrogen(QM8, QM9), avoid calling `GetTotalNumHs`
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    return np.array(results)


def calc_bond_features(bond, use_chirality=True):
    """
    bond: rdkit.Chem.rdchem.Bond
    use_chirality: whether to use chirality
    """
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def get_vdw_radius(a):
    metal_symbols = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
    atomic_number = a.GetAtomicNum()
    atomic_number_to_radius = {6: 1.90, 7: 1.8, 8: 1.7, 16: 2.0, 15: 2.1,
                               9: 1.5, 17: 1.8, 35: 2.0, 53: 2.2, 30: 1.2, 25: 1.2, 26: 1.2, 27: 1.2,
                               12: 1.2, 28: 1.2, 20: 1.2, 29: 1.2}
    if atomic_number in atomic_number_to_radius.keys():
        return atomic_number_to_radius[atomic_number]
    return Chem.GetPeriodicTable().GetRvdw(atomic_number)


def get_hydrophobic_atom(m):
    n = m.GetNumAtoms()
    retval = np.zeros((n,))
    for i in range(n):
        a = m.GetAtomWithIdx(i)
        s = a.GetSymbol()
        if s.upper() in ["F", "CL", "BR", "I"]:
            retval[i] = 1
        elif s.upper() in ["C"]:
            n_a = [x.GetSymbol() for x in a.GetNeighbors()]
            diff = list(set(n_a) - set(["C"]))
            if len(diff) == 0:
                retval[i] = 1
        else:
            continue
    return retval


def get_hbond_donor_indice(m):
    """
    indice = m.GetSubstructMatches(HDonorSmarts)
    if len(indice)==0: return np.array([])
    indice = np.array([i for i in indice])[:,0]
    return indice
    """
    # smarts = ["[!$([#6,H0,-,-2,-3])]", "[!H0;#7,#8,#9]"]
    n = m.GetNumAtoms()
    retval = np.full((n,), -1)
    smarts = ["[!#6;!H0]"]
    indice = []
    for s in smarts:
        s = Chem.MolFromSmarts(s)
        indice += [i[0] for i in m.GetSubstructMatches(s)]
    indice = np.array(indice)
    for num in range(len(indice)):
        retval[num] = indice[num]
    return retval


def get_hbond_acceptor_indice(m):
    # smarts = ["[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    #          "[#6,#7;R0]=[#8]"]
    n = m.GetNumAtoms()
    retval = np.full((n,), -1)
    smarts = [
        "[$([!#6;+0]);!$([F,Cl,Br,I]);!$([o,s,nX3]);!$([Nv5,Pv5,Sv4,Sv6])]"]
    indice = []
    for s in smarts:
        s = Chem.MolFromSmarts(s)
        indice += [i[0] for i in m.GetSubstructMatches(s)]
    indice = np.array(indice)
    for num in range(len(indice)):
        retval[num] = indice[num]
    return retval


def get_metal_indice(m):
    # smarts = ["[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    #          "[#6,#7;R0]=[#8]"]
    n = m.GetNumAtoms()
    retval = np.full((n,), -1)
    metal_symbols = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
    indices = np.array([i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).GetSymbol() in metal_symbols])
    for i in range(len(indices)):
        retval[i] = indices[i]
    return retval


def load_mol(molpath, explicit_H=False, use_chirality=True):
    # load mol
    if re.search(r'.pdb$', molpath):
        mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
    elif re.search(r'.mol2$', molpath):
        mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
    elif re.search(r'.sdf$', molpath):
        mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")

    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def cal_charge(m):
    try:
        charges = AllChem.CalcEEMcharges(m)
        AllChem.ComputeGasteigerCharges(m)
    except:
        charges = None
    if charges is None:
        charges = [float(m.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
                   for i in range(m.GetNumAtoms())]
    else:
        for i in range(m.GetNumAtoms()):
            if charges[i] > 3 or charges[i] < -3:
                charges[i] = float(m.GetAtomWithIdx(
                    i).GetProp("_GasteigerCharge"))

    return np.array(charges)


def mol_to_radiusg(ligand_mol, dis_threshold=6, use_chirality=True, explicit_H=False):
    '''
    半径图构建, efeats=7
    efeats: [单键, 双键，三键，芳香键，非共价键...], 5种共价键类型+1种非共价键类型
    '''

    # construct graph2
    m2_postions = ligand_mol.GetConformer().GetPositions()

    num_atoms = ligand_mol.GetNumAtoms()

    # add edges, ligand_mol
    dis_mat = distance_matrix(m2_postions, m2_postions)
    node_idx = np.where(dis_mat < dis_threshold)  # 此处已经包含了自循环
    # g2.add_edges(node_idx[0], node_idx[1])

    # 获取距离特征
    distances_feats = np.array([dis_mat[i, j] for (i, j) in zip(node_idx[0], node_idx[1])])

    # 获取共价键的id
    num_bonds = ligand_mol.GetNumBonds()
    src_ls = []
    dst_ls = []
    covbf = []

    vdw_radius = np.expand_dims(np.array([get_vdw_radius(a) for a in ligand_mol.GetAtoms()]), axis=1)
    metal_symbols = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
    no_metal = np.expand_dims(np.array([1 if a.GetSymbol() not in metal_symbols else 0 for a in ligand_mol.GetAtoms()]),
                              axis=1)
    hydrophobic = np.expand_dims(get_hydrophobic_atom(ligand_mol), axis=1)
    h_acc_indice = np.expand_dims(get_hbond_acceptor_indice(ligand_mol), axis=1)
    h_donor_indice = np.expand_dims(get_hbond_donor_indice(ligand_mol), axis=1)
    metal_indice = np.expand_dims(get_metal_indice(ligand_mol), axis=1)
    charge = np.expand_dims(cal_charge(ligand_mol), axis=1)
    physic_feats = np.concatenate(
        [vdw_radius, no_metal, hydrophobic, h_acc_indice, h_donor_indice, metal_indice, charge],
        axis=1)

    rotor = CalcNumRotatableBonds(ligand_mol)

    for i in range(num_bonds):
        bond = ligand_mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = calc_bond_features(bond, use_chirality=False)
        src_ls.extend([u, v])
        dst_ls.extend([v, u])
        covbf.append(bond_feats)
        covbf.append(bond_feats)

    bond_tps = np.zeros((len(distances_feats), 7))  # 键类型特征，长度为6, 5种共价键类型+1种非共价键类型
    bond_tps[:, -1] = 1.0  # 最后一列大多数值为1.0, 为非共价键

    res1 = list(zip(node_idx[0], node_idx[1]))
    res2 = list(zip(src_ls, dst_ls))
    for idx, r in enumerate(res2):
        bond_tps[res1.index(r)] = np.append(covbf[idx], 0.0)  # 最后一列为0.0, 此处的键为共价键
    distances_feats = distances_feats.reshape(-1, 1)
    edge_feats = np.hstack((distances_feats, bond_tps))
    edge_feats = th.tensor(edge_feats, dtype=th.float)

    # assign atom features
    # 'h', features of atoms
    atom_feats = np.array([calc_atom_features(a, explicit_H=explicit_H) for a in ligand_mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(ligand_mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    g = Data(x=th.tensor(atom_feats, dtype=th.float), physic_feats=th.tensor(physic_feats, dtype=th.float),
             rotor=th.tensor(rotor, dtype=th.float),
             edge_index=th.tensor([node_idx[0], node_idx[1]]),
             pos=th.tensor(ligand_mol.GetConformers()[0].GetPositions(), dtype=th.float),
             edge_attr=th.tensor(np.array(edge_feats), dtype=th.float))

    return g


def mol_to_graph(mol, explicit_H=False, use_chirality=True):
    """
    mol: rdkit.Chem.rdchem.Mol
    explicit_H: whether to use explicit H
    use_chirality: whether to use chirality
    """
    # Add nodes
    num_atoms = mol.GetNumAtoms()

    atom_feats = np.array([calc_atom_features(a, explicit_H=explicit_H) for a in mol.GetAtoms()])
    if use_chirality:
        chiralcenters = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                  useLegacyImplementation=False)
        chiral_arr = np.zeros([num_atoms, 3])
        for (i, rs) in chiralcenters:
            if rs == 'R':
                chiral_arr[i, 0] = 1
            elif rs == 'S':
                chiral_arr[i, 1] = 1
            else:
                chiral_arr[i, 2] = 1
        atom_feats = np.concatenate([atom_feats, chiral_arr], axis=1)

    vdw_radius = np.expand_dims(np.array([get_vdw_radius(a) for a in mol.GetAtoms()]), axis=1)
    metal_symbols = ["Zn", "Mn", "Co", "Mg", "Ni", "Fe", "Ca", "Cu"]
    no_metal = np.expand_dims(np.array([1 if a.GetSymbol() not in metal_symbols else 0 for a in mol.GetAtoms()]),
                              axis=1)
    hydrophobic = np.expand_dims(get_hydrophobic_atom(mol), axis=1)
    h_acc_indice = np.expand_dims(get_hbond_acceptor_indice(mol), axis=1)
    h_donor_indice = np.expand_dims(get_hbond_donor_indice(mol), axis=1)
    metal_indice = np.expand_dims(get_metal_indice(mol), axis=1)
    charge = np.expand_dims(cal_charge(mol), axis=1)
    physic_feats = np.concatenate(
        [vdw_radius, no_metal, hydrophobic, h_acc_indice, h_donor_indice, metal_indice, charge], axis=1)

    rotor = CalcNumRotatableBonds(mol)

    # obtain the positions of the atoms
    atomCoords = mol.GetConformer().GetPositions()

    # Add edges
    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        bond_feats = calc_bond_features(bond, use_chirality=use_chirality)
        coord_diff = atomCoords[u] - atomCoords[v]
        radial = np.array([np.sum((coord_diff) ** 2, 0)])
        bond_feats = np.concatenate((bond_feats, radial), 0)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g = Data(x=th.tensor(atom_feats, dtype=th.float), physic_feats=th.tensor(physic_feats, dtype=th.float),
             rotor=th.tensor(rotor, dtype=th.float),
             edge_index=th.tensor([src_list, dst_list]),
             pos=th.tensor(atomCoords, dtype=th.float),
             edge_attr=th.tensor(np.array(bond_feats_all), dtype=th.float))

    return g


def mol_to_graph2(prot_path, lig_path, cutoff=10.0, explicit_H=False, use_chirality=True):
    prot = load_mol(prot_path, explicit_H=explicit_H, use_chirality=use_chirality)
    lig = load_mol(lig_path, explicit_H=explicit_H, use_chirality=use_chirality)
    # gp = mol_to_graph(prot, explicit_H=explicit_H, use_chirality=use_chirality)
    gp = mol_to_radiusg(prot, explicit_H=explicit_H, use_chirality=use_chirality)
    gl = mol_to_graph(lig, explicit_H=explicit_H, use_chirality=use_chirality)
    return gp, gl


def label_query(pdbid, df):
    return df.loc[pdbid, "labels"]


def pdbbind_handle(pdbid, args, i):
    prot_path = "%s/%s/%s_pocket_%s.pdb" % (args.dir, pdbid, pdbid, args.cutoff)
    lig_path = "%s/%s/ligand.mol2" % (args.dir, pdbid)
    # try:
    gp, gl = mol_to_graph2(prot_path,
                           lig_path,
                           cutoff=args.cutoff,
                           explicit_H=args.useH,
                           use_chirality=args.use_chirality)
    # except:
    #     print("%s failed to generare the graph" % pdbid)
    #     gp, gl = None, None
    # gm = None
    return pdbid, gp, gl, label_query(pdbid.split('-')[-1], args.ref)


def UserInput():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-d', '--dir', default="/home/suqun/tmp/GMP/data/SARS/data",
                   help='The directory to store the protein-ligand complexes.')
    p.add_argument('-c', '--cutoff', default=5.0, type=float,
                   help='the cutoff to determine the pocket')
    p.add_argument('-o', '--outprefix', default="SARS_H",
                   help='The output bin file.')
    p.add_argument('-r', '--ref', default="/home/suqun/tmp/GMP/data/SARS/id2labels.csv",
                   help='The reference file to query the label of the complex.')
    p.add_argument('-usH', '--useH', default=False, action="store_true",
                   help='whether to use the explicit H atoms.')
    p.add_argument('-uschi', '--use_chirality', default=True, action="store_true",
                   help='whether to use chirality.')
    p.add_argument('-p', '--parallel', default=False, action="store_true",
                   help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')

    args = p.parse_args()
    return args


def main():
    args = UserInput()
    i = 0
    pdbids = [x for x in os.listdir(args.dir) if os.path.isdir("%s/%s" % (args.dir, x))]
    args.ref = pd.read_csv(args.ref, index_col=0, header=0)
    if args.parallel:
        results = Parallel(n_jobs=-1)(delayed(pdbbind_handle)(pdbid, args) for pdbid in pdbids)
    else:
        results = []
        for pdbid in pdbids:
            results.append(pdbbind_handle(pdbid, args, i))
            i = i + 1
            print(i)
    results = list(filter(lambda x: x[1] != None, results))
    ids, graphs_p, graphs_l, labels = list(zip(*results))
    np.save("./SARS/%s_ids" % args.outprefix, (ids, labels))
    th.save(graphs_p, "./SARS/%s_prot.pt" % args.outprefix)
    th.save(graphs_l, "./SARS/%s_lig.pt" % args.outprefix)


if __name__ == '__main__':
    main()
