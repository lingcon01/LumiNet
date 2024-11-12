import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import os, sys
import pickle
from scipy.stats import pearsonr
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

sys.path.append("/home/suqun/tmp/GMP/pretrain")
from torch_geometric.loader import DataLoader
from GenScore.data.data import PDBbindDataset
from GenScore.model.ET_finetune import GenScore, GraphTransformer, GatedGCN, SubGT
from GenScore.model.mdn_utils import GIP_eval_epoch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
args = {}
args["batch_size"] = 8
args["dist_threhold"] = 5.
args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
args['seeds'] = 126
args["num_workers"] = 2
args["model_path1"] = "/home/suqun/tmp/GMP/pretrain/EGMDN/ET_finetune/cleaned_result/version4_inter2_each_fep.pth"
args["model_path2"] = "/home/suqun/tmp/GMP/pretrain/EGMDN/ET_finetune/cleaned_result/version4_inter2_each_derivate.pth"
args["data_dir1"] = "/home/suqun/tmp/GMP/pretrain/GenScore/feats/fep/split_data"
args["data_dir2"] = "/home/suqun/tmp/GMP/pretrain/GenScore/feats/derivate_new/split_data"
args["test_prefix1"] = ["cdk8", "cmet", "eg5", "hif2a", "pfkfb3", "shp2", "syk", "tnks2"]
args["test_prefix2"] = ["Bace", "CDK2", "Jnk1", "MCL1", "p38", "PTP1B", "Thrombin", "Tyk2"]
args["test_prefix3"] = ["all"]
args["test_prefix4"] = ["derivate"]
args["cutoff"] = 10.0
args["encoder"] = "gt"
args["num_node_featsp"] = 41
args["num_node_featsl"] = 41
args["num_edge_featsp"] = 7
args["num_edge_featsl"] = 10
args["hidden_dim0"] = 128
args["hidden_dim"] = 128
args["n_gaussians"] = 10
args["dropout_rate"] = 0.15


# args["outprefix"] = "cmet"


def scoring(ids, prots, ligs, model, seed=None, **kwargs):
    """
	prot: The input protein file ('.pdb')
	lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
	modpath: The path to store the pre-trained model
	gen_pocket: whether to generate the pocket from the protein file.
	reflig: The reference ligand to determine the pocket.
	cutoff: The distance within the reference ligand to determine the pocket.
	explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
	use_chirality: whether to adopt the information of chirality to represent the molecules.	
	parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
	kwargs: other arguments related with model
	"""
    # try:
    dataset = PDBbindDataset(ids=ids, prots=prots, ligs=ligs)

    _, val_inds = dataset.train_and_test_split(valnum=6, seed=seed)
    print(f"fep_score: {val_inds}")

    data = PDBbindDataset(ids=dataset.pdbids[val_inds],
                          ligs=dataset.gls[val_inds],
                          prots=dataset.gps[val_inds],
                          labels=dataset.labels[val_inds]
                          )

    test_loader = DataLoader(dataset=dataset,
                             batch_size=kwargs["batch_size"],
                             shuffle=False,
                             num_workers=kwargs["num_workers"])

    spearman, pr, preds, rmse, _ = GIP_eval_epoch(model=model, data_loader=test_loader,
                                               dist_threhold=kwargs['dist_threhold'], device=kwargs['device'])

    # print(preds)
    return spearman, pr, rmse


def fep_score(model, seed):
    fep_score_spearman = []
    fep_score_pr = []
    derivate_score_spearman = []
    derivate_score_pr = []
    fep_score_rmse = []
    derivate_score_rmse = []

    for prefix in args["test_prefix1"]:
        print(f"================================={prefix}================================")

        prots = '%s/%s_prot.pt' % (args["data_dir1"], prefix)
        ligs = '%s/%s_lig.pt' % (args["data_dir1"], prefix)
        ids = '%s/%s_ids.npy' % (args["data_dir1"], prefix)

        spearman, pr, rmse = scoring(ids, prots, ligs, model, parallel=True, **args)

        fep_score_spearman.append(spearman)
        fep_score_pr.append(pr)
        fep_score_rmse.append(rmse)

    for prefix in args["test_prefix2"]:
        print(f"================================={prefix}================================")

        prots = '%s/%s_prot.pt' % (args["data_dir2"], prefix)
        ligs = '%s/%s_lig.pt' % (args["data_dir2"], prefix)
        ids = '%s/%s_ids.npy' % (args["data_dir2"], prefix)

        spearman, pr, rmse = scoring(ids, prots, ligs, model, parallel=True, seed=seed, **args)

        derivate_score_spearman.append(spearman)
        derivate_score_pr.append(pr)
        derivate_score_rmse.append(rmse)

    fep_spearman = sum(fep_score_spearman) / len(fep_score_spearman)
    fep_pr = sum(fep_score_pr) / len(fep_score_pr)
    fep_rmse = sum(fep_score_rmse) / len(fep_score_rmse)
    derivate_spearman = sum(derivate_score_spearman) / len(derivate_score_spearman)
    derivate_pr = sum(derivate_score_pr) / len(derivate_score_pr)
    derivate_rmse = sum(derivate_score_rmse) / len(derivate_score_rmse)

    return fep_spearman, fep_pr, derivate_spearman, derivate_pr, fep_rmse, derivate_rmse
    # return derivate_score_spearman, derivate_score_pr, derivate_score_rmse, derivate_pr
    # return fep_score_spearman, fep_score_pr, fep_score_rmse, fep_pr
