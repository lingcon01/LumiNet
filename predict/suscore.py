import numpy as np
import torch as th
from joblib import Parallel, delayed
import pandas as pd
import argparse
import os, sys
import MDAnalysis as mda

sys.path.append(os.path.abspath(__file__).replace("suscore.py", ".."))
from torch_geometric.loader import DataLoader
from GenScore.data.data import VSDataset
from GenScore.model.ET_MDN import GenScore, GraphTransformer, SubGT
from GenScore.model.mdn_utils import GIP_eval_epoch
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


# you need to set the babel libdir first if you need to generate the pocket
# os.environ["BABEL_LIBDIR"] = "/home/shenchao/.conda/envs/my3/lib/openbabel/3.1.0"

def Input():
    p = argparse.ArgumentParser()
    p.add_argument('-p', '--prot', required=True,
                   help='Input protein file (.pdb)')
    p.add_argument('-l', '--lig', required=True,
                   help='Input ligand file (.sdf/.mol2)')
    p.add_argument('-m', '--model',
                   default="./ckpt/base_model.pt",
                   help='trained model path (default: ""./ckpt/base_model.pt"')
    p.add_argument('-e', '--encoder', default="gt", choices=["gt", "gatedgcn"],
                   help='the feature encoders for the representation of proteins and ligands (default: "gt")')
    p.add_argument('-o', '--outprefix', default="out",
                   help='the prefix of output file (default: "out")')
    p.add_argument('-gen_pocket', '--gen_pocket', action="store_true", default=False,
                   help='whether to generate the pocket')
    p.add_argument('-c', '--cutoff', default=5.0, type=float,
                   help='the cutoff the define the pocket and interactions within the pocket (default: 10.0)')
    p.add_argument('-rl', '--reflig', default=None,
                   help='the reference ligand to determine the pocket(.sdf/.mol2)')
    p.add_argument('-pl', '--parallel', default=False, action="store_true",
                   help='whether to obtain the graphs in parallel (When the dataset is too large,\
						 it may be out of memory when conducting the parallel mode).')
    args = p.parse_args()
    if args.gen_pocket:
        if args.reflig is None:
            raise ValueError("if pocket is generated, the reference ligand should be provided.")
    return args


def scoring(prot, lig, modpath,
            cut=5.0,
            gen_pocket=False,
            reflig=None,
            encoder="gt",
            explicit_H=False,
            use_chirality=True,
            parallel=False,
            **kwargs
            ):
    """
	prot: The input protein file ('.pdb')
	lig: The input ligand file ('.sdf|.mol2', multiple ligands are supported)
	modpath: The path to store the pre-trained model
	gen_pocket: whether to generate the pocket from the protein file.
	encoder: The feature encoders for the representation of proteins and ligands.
	reflig: The reference ligand to determine the pocket.
	cut: The distance within the reference ligand to determine the pocket.
	atom_contribution: whether the decompose the score at atom level.
	res_contribution: whether the decompose the score at residue level.
	explicit_H: whether to use explicit hydrogen atoms to represent the molecules.
	use_chirality: whether to adopt the information of chirality to represent the molecules.
	parallel: whether to generate the graphs in parallel. (This argument is suitable for the situations when there are lots of ligands/poses)
	kwargs: other arguments related with model
	"""
    # try:
    data = VSDataset(ligs=lig,
                     prot=prot,
                     cutoff=cut,
                     gen_pocket=gen_pocket,
                     reflig=reflig,
                     explicit_H=explicit_H,
                     use_chirality=use_chirality,
                     parallel=parallel)

    test_loader = DataLoader(dataset=data,
                             batch_size=kwargs["batch_size"],
                             shuffle=False,
                             num_workers=kwargs["num_workers"]
                             )

    if encoder == "gt":
        ligmodel = SubGT(in_channels=kwargs["num_node_featsl"],
                         edge_features=kwargs["num_edge_featsl"],
                         num_hidden_channels=kwargs["hidden_dim0"],
                         activ_fn=th.nn.SiLU(),
                         transformer_residual=True,
                         num_attention_heads=4,
                         norm_to_apply='batch',
                         dropout_rate=0.15,
                         num_layers=6
                         )

        protmodel = GraphTransformer(in_channels=kwargs["num_node_featsp"],
                                     edge_features=kwargs["num_edge_featsp"],
                                     num_hidden_channels=kwargs["hidden_dim0"],
                                     activ_fn=th.nn.SiLU(),
                                     transformer_residual=True,
                                     num_attention_heads=4,
                                     norm_to_apply='batch',
                                     dropout_rate=0.15,
                                     num_layers=6
                                     )

    else:
        raise ValueError("Please choose gt model.")

    model = GenScore(ligmodel, protmodel,
                     in_channels=kwargs["hidden_dim0"],
                     hidden_dim=kwargs["hidden_dim"],
                     n_gaussians=kwargs["n_gaussians"],
                     dropout_rate=kwargs["dropout_rate"],
                     dist_threhold=kwargs["dist_threhold"]).to(kwargs['device'])

    checkpoint = th.load(modpath, map_location=th.device(kwargs['device']))
    # model.load_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(checkpoint)

    loss, pr, preds, rmse, atom_energy = GIP_eval_epoch(model=model, data_loader=test_loader,
                                                         dist_threhold=kwargs['dist_threhold'],
                                                         device=kwargs['device'])

    return data.ids, preds, atom_energy.cpu().numpy()


def main():
    inargs = Input()
    args = {}
    args["batch_size"] = 128
    args["dist_threhold"] = 5.
    args['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
    args["num_workers"] = 10
    args["num_node_featsp"] = 41
    args["num_node_featsl"] = 41
    args["num_edge_featsp"] = 7
    args["num_edge_featsl"] = 10
    args["hidden_dim0"] = 128
    args["hidden_dim"] = 128
    args["n_gaussians"] = 10
    args["dropout_rate"] = 0.15

    ids, preds, atom_energy = scoring(prot=inargs.prot, lig=inargs.lig, modpath=inargs.model, cut=inargs.cutoff,
                                      gen_pocket=inargs.gen_pocket,
                                      reflig=inargs.reflig, encoder=inargs.encoder,
                                      explicit_H=False,
                                      use_chirality=True,
                                      parallel=inargs.parallel,
                                      **args)

    # for each_energy in atom_energy:
    #
    #     node_idx = np.where(each_energy < 4)
    #     energy = np.array([each_energy[i, j] for (i, j) in zip(node_idx[0], node_idx[1])])
    #     print(energy)

    # predict_info = {"pdbid": ids, "total_energy": preds, "atom_energy": (node_idx, energy)}
    df = pd.DataFrame({"pdbid":ids, "score": preds})
    df.to_csv("./SuScore_fragment_2200.csv")
    predict_info = {"pdbid": ids, "total_energy": preds, "atom_energy": atom_energy}

    np.save("/home/suqun/tmp/GMP/pretrain/predict/1", predict_info)


if __name__ == '__main__':
    main()

