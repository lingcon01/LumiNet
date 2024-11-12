import torch as th
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn.functional as F
import sys

sys.path.append("/home/suqun/tmp/GMP/pretrain")
from SuScore.data.data import PDBbindDataset
from SuScore.model.ET_MDN import GenScore, GraphTransformer, SubGT
from SuScore.model.mdn_utils import EarlyStopping, set_random_seed, run_a_train_epoch, run_an_eval_epoch, mdn_loss_fn, \
    GIP_train_epoch, GIP_eval_epoch, GIP_semi_label
from fep_score import fep_score
import torch.multiprocessing
import logging

logging.basicConfig(filename='train_EMGP.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    import argparse

    seed = 'seed0'

    p = argparse.ArgumentParser()
    p.add_argument('--num_epochs', type=int, default=5000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--aux_weight', type=float, default=0.001)
    p.add_argument('--affi_weight', type=float, default=0)
    p.add_argument('--patience', type=int, default=150)
    p.add_argument('--num_workers', type=int, default=2)
    p.add_argument('--model_path', type=str,
                   default=f"/home/suqun/tmp/GMP/pretrain/train_and_test/semi_train_policy/ET_MDN/derivate/{seed}/retrain2/retrain_2_{seed}.pth")
    p.add_argument('--encoder', type=str, choices=['gt', 'gatedgcn'], default="gt")
    p.add_argument('--mode', type=str, choices=['lower', 'higher'], default="lower")
    p.add_argument('--finetune', action="store_true", default=True)
    p.add_argument('--original_model_path', type=str,
                   default='/home/suqun/tmp/GMP/pretrain/train_and_test/semi_train_policy/ET_MDN/derivate/{seed}/retrain1/retrain_1_{seed}.pth')
    p.add_argument('--lr', type=int, default=3)
    p.add_argument('--weight_decay', type=int, default=5)
    p.add_argument('--data_dir', type=str, default="/home/suqun/tmp/GMP/pretrain/GenScore/feats")
    p.add_argument('--val_dir', type=str, default="/home/suqun/tmp/GMP/pretrain/GenScore/feats/fep")
    p.add_argument('--val_dir3', type=str, default="/home/suqun/tmp/GMP/pretrain/GenScore/feats/derivate/split_data")
    p.add_argument('--data_prefix', type=str, default="pignet")
    p.add_argument('--valnum', type=int, default=0)
    p.add_argument('--seeds', type=int, default=126)
    p.add_argument('--hidden_dim0', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--n_gaussians', type=int, default=10)
    p.add_argument('--dropout_rate', type=float, default=0.15)
    p.add_argument('--dist_threhold', type=float, default=7., help="the distance threhold for training")
    p.add_argument('--dist_threhold2', type=float, default=5., help="the distance threhold for testing")
    # p.add_argument('--device', type=str, default="cpu")
    args = p.parse_args()
    args.device = 'cuda' if th.cuda.is_available() else 'cpu'
    args.test_prefix2 = ["BACE", "CDK2", "Jnk1", "MCL1", "p38", "PTP1B", "Thrombin", "Tyk2"]

    datadir = '/home/suqun/tmp/GMP/pretrain/GenScore/feats/derivate/virtual_data'

    print(f'{args.device}')
    train_dataset1 = PDBbindDataset(ids="%s/%s_ids.npy" % (args.data_dir, args.data_prefix),
                                    ligs="%s/%s_lig.pt" % (args.data_dir, args.data_prefix),
                                    prots="%s/%s_prot.pt" % (args.data_dir, args.data_prefix)
                                    )

    train_dataset2 = PDBbindDataset(ids="%s/%s_ids.npy" % (datadir, f'ET_MDN/init'),
                                    ligs="%s/%s_lig.pt" % (datadir, 'derivate'),
                                    prots="%s/%s_prot.pt" % (datadir, 'derivate')
                                    )

    val_dataset1 = PDBbindDataset(ids="%s/%s_ids.npy" % (args.data_dir, 'casf2016'),
                                  ligs="%s/%s_lig.pt" % (args.data_dir, 'casf2016'),
                                  prots="%s/%s_prot.pt" % (args.data_dir, 'casf2016')
                                  )

    for prefix in args.test_prefix2:
        prots = '%s/%s_prot.pt' % (args.val_dir3, prefix)
        ligs = '%s/%s_lig.pt' % (args.val_dir3, prefix)
        ids = '%s/%s_ids.npy' % (args.val_dir3, prefix)

        dataset = PDBbindDataset(ids=ids, prots=prots, ligs=ligs)

        _, val_inds = dataset.train_and_test_split(valnum=2, seed=0)

        print(val_inds)

        split_data = PDBbindDataset(ids=dataset.pdbids[val_inds],
                                    ligs=dataset.gls[val_inds],
                                    prots=dataset.gps[val_inds],
                                    labels=dataset.labels[val_inds]
                                    )
        train_dataset2 = ConcatDataset([train_dataset2, split_data])

    set_random_seed(args.seeds)

    if args.encoder == "gt":
        ligmodel = SubGT(in_channels=41,
                         edge_features=10,
                         num_hidden_channels=args.hidden_dim0,
                         activ_fn=th.nn.SiLU(),
                         transformer_residual=True,
                         num_attention_heads=4,
                         norm_to_apply='batch',
                         dropout_rate=0.15,
                         num_layers=6
                         )

        protmodel = GraphTransformer(in_channels=41,
                                     edge_features=7,
                                     num_hidden_channels=args.hidden_dim0,
                                     activ_fn=th.nn.SiLU(),
                                     transformer_residual=True,
                                     num_attention_heads=4,
                                     norm_to_apply='batch',
                                     dropout_rate=0.15,
                                     num_layers=6
                                     )

    model = GenScore(ligmodel, protmodel,
                     in_channels=args.hidden_dim0,
                     hidden_dim=args.hidden_dim,
                     n_gaussians=args.n_gaussians,
                     dropout_rate=args.dropout_rate,
                     dist_threhold=args.dist_threhold).to(args.device)

    if args.finetune:
        if args.original_model_path is None:
            raise ValueError('the argument "original_model_path" should be given')
        model_dict = model.state_dict()
        checkpoint = th.load(args.original_model_path, map_location=th.device(args.device))

        # 遍历checkpoint的键值对，将对应的参数加载到模型参数字典中
        for k, v in checkpoint['model_state_dict'].items():
        # for k, v in checkpoint.items():
            if k in model_dict:
                model_dict[k] = v

        for name, param in model.named_parameters():
            print(name, param.shape)

        # 将加载的模型参数加载到模型中
        model.load_state_dict(model_dict)

    optimizer = th.optim.Adam(model.parameters(), lr=10 ** -args.lr, weight_decay=10 ** -args.weight_decay)

    train_loader1 = DataLoader(dataset=train_dataset1,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers)

    train_loader2 = DataLoader(dataset=train_dataset2,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers)

    val_loader1 = DataLoader(dataset=val_dataset1,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers)

    stopper = EarlyStopping(patience=args.patience, mode=args.mode, filename=args.model_path)

    for epoch in range(args.num_epochs):

        total_loss_train, train_pr = GIP_semi_label(epoch,
                                                    model,
                                                    train_loader1,
                                                    train_loader2,
                                                    optimizer,
                                                    affi_weight=args.affi_weight,
                                                    aux_weight=args.aux_weight,
                                                    dist_threhold=args.dist_threhold2,
                                                    device=args.device)

        total_loss_test1, test_pr1, _, _ = GIP_eval_epoch(model,
                                                          val_loader1,
                                                          dist_threhold=args.dist_threhold2,
                                                          affi_weight=args.affi_weight,
                                                          aux_weight=args.aux_weight,
                                                          device=args.device)

        spearman, pr, rmse, aver_pr = fep_score(model)

        new_ben = rmse

        early_stop = stopper.step(
            {"BACE": new_ben[0], "CDK2": new_ben[1], "JNK1": new_ben[2], "MCL1": new_ben[3], "P38": new_ben[4],
             "PTP1B": new_ben[5], "Thrombin": new_ben[6], "TYK2": new_ben[7]}, model)

        try:
            logging.info('epoch {:d}/{:d}, train_loss {:.4f}, total_loss_val1 {:.4f}, affi_loss_val1 {:.4f}, '
                         'BACE_pr {:.4f}, BACE_spearman {:.4f}, BACE_rmse {:.4f}, CDK2_pr {:.4f}, CDK2_spearman {:.4f},'
                         'CDK2_rmse {:.4f}, JNK1_pr {:.4f}, JNK1_spearman {:.4f}, JNK1_rmse {:.4f}, MCL1_pr {:.4f}, '
                         'MCL1_spearman {:.4f}, MCL1_rmse {:.4f}, P38_pr {:.4f}, P38_spearman {:.4f}'
                         'P38_rmse {:.4f}, PTP1B_pr {:.4f}, PTP1B_spearman {:.4f}, PTP1B_rmse {:.4f}, Thrombin_pr {:.4f}, '
                         'Thrombin_spearman {:4f}, Thrombin_rmse {:.4f}, TYK2_pr {:.4f}, TYK2_spearman {:.4f}, TYK2_rmse {:.4f}'
                         'best BACE validation {:.4f}, best CDK2 validation {:.4f}, best JNK1 validation {:.4f},'
                         'best MCL1 validation {:.4f}, best P38 validation {:.4f}, best PTP1B validation {:.4f}, '
                         'best Thrombin validation {:.4f}, best TYK2 validation {:.4f}, aver_pr {:.4f}'.format(
                epoch + 1, args.num_epochs, total_pr, total_loss_test1, test_pr1, pr[0], spearman[0], rmse[0],
                pr[1], spearman[1], rmse[1], pr[2], spearman[2], rmse[2], pr[3], spearman[3], rmse[3], pr[4],
                spearman[4], rmse[4], pr[5], spearman[5], rmse[5], pr[6], spearman[6], rmse[6], pr[7], spearman[7],
                rmse[7], stopper.best_score["BACE"], stopper.best_score["CDK2"], stopper.best_score["JNK1"],
                stopper.best_score["MCL1"], stopper.best_score["P38"], stopper.best_score["PTP1B"],
                stopper.best_score["Thrombin"], stopper.best_score["TYK2"], aver_pr))
        except:
            pass

        print(
            'epoch {:d}/{:d}, train_loss {:.4f}, total_loss_val1 {:.4f}, affi_loss_val1 {:.4f}, '
            'BACE_pr {:.4f}, BACE_spearman {:.4f}, BACE_rmse {:.4f}, CDK2_pr {:.4f}, CDK2_spearman {:.4f},'
            'CDK2_rmse {:.4f}, JNK1_pr {:.4f}, JNK1_spearman {:.4f}, JNK1_rmse {:.4f}, MCL1_pr {:.4f}, '
            'MCL1_spearman {:.4f}, MCL1_rmse {:.4f}, P38_pr {:.4f}, P38_spearman {:.4f}'
            'P38_rmse {:.4f}, PTP1B_pr {:.4f}, PTP1B_spearman {:.4f}, PTP1B_rmse {:.4f}, Thrombin_pr {:.4f}, '
            'Thrombin_spearman {:4f}, Thrombin_rmse {:.4f}, TYK2_pr {:.4f}, TYK2_spearman {:.4f}, TYK2_rmse {:.4f}'
            'best BACE validation {:.4f}, best CDK2 validation {:.4f}, best JNK1 validation {:.4f},'
            'best MCL1 validation {:.4f}, best P38 validation {:.4f}, best PTP1B validation {:.4f}, '
            'best Thrombin validation {:.4f}, best TYK2 validation {:.4f}, aver_pr {:.4f}'.format(
                epoch + 1, args.num_epochs, train_pr, total_loss_test1, test_pr1, pr[0], spearman[0], rmse[0],
                pr[1], spearman[1], rmse[1], pr[2], spearman[2], rmse[2], pr[3], spearman[3], rmse[3], pr[4],
                spearman[4], rmse[4], pr[5], spearman[5], rmse[5], pr[6], spearman[6], rmse[6], pr[7], spearman[7],
                rmse[7], stopper.best_score["BACE"], stopper.best_score["CDK2"], stopper.best_score["JNK1"],
                stopper.best_score["MCL1"], stopper.best_score["P38"], stopper.best_score["PTP1B"],
                stopper.best_score["Thrombin"], stopper.best_score["TYK2"], aver_pr)
        )  # +' validation result:', validation_result)
        if early_stop:
            break
