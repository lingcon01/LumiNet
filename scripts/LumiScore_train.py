import torch as th
import numpy as np
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import sys

sys.path.append("/home/suqun/tmp/GMP/pretrain")
from SuScore.data.data import PDBbindDataset
from SuScore.model.ET_MDN import GenScore, GraphTransformer, SubGT
from SuScore.model.mdn_utils import EarlyStopping, set_random_seed, run_a_train_epoch, run_an_eval_epoch, mdn_loss_fn, \
    GIP_train_epoch, GIP_eval_epoch, GIP_semi_label
import torch.multiprocessing
import logging

logging.basicConfig(filename='train_EMGP.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--num_epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--aux_weight', type=float, default=0.001)
    p.add_argument('--affi_weight', type=float, default=0)
    p.add_argument('--patience', type=int, default=150)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--model_path', type=str,
                   default="/home/suqun/tmp/GMP/pretrain/train_and_test/my_test/SuScore_local3_88.pth")
    p.add_argument('--encoder', type=str, choices=['gt', 'gatedgcn'], default="gt")
    p.add_argument('--mode', type=str, choices=['lower', 'higher'], default="higher")
    p.add_argument('--finetune', action="store_true", default=True)
    p.add_argument('--original_model_path', type=str,
                   default='/home/suqun/tmp/GMP/pretrain/EGMDN/ET_pretrain.pth')
    p.add_argument('--lr', type=int, default=3)
    p.add_argument('--weight_decay', type=int, default=5)
    p.add_argument('--data_dir', type=str, default="/home/suqun/tmp/GMP/pretrain/GenScore/feats")
    p.add_argument('--data_prefix', type=str, default="pignet")
    p.add_argument('--valnum', type=int, default=500)
    p.add_argument('--seeds', type=int, default=88)
    p.add_argument('--hidden_dim0', type=int, default=128)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--n_gaussians', type=int, default=10)
    p.add_argument('--dropout_rate', type=float, default=0.15)
    p.add_argument('--dist_threhold', type=float, default=7., help="the distance threhold for training")
    p.add_argument('--dist_threhold2', type=float, default=5., help="the distance threhold for testing")
    # p.add_argument('--device', type=str, default="cpu")
    args = p.parse_args()
    args.device = 'cuda' if th.cuda.is_available() else 'cpu'

    datadir = '/home/suqun/tmp/GMP/pretrain/GenScore/feats/fep/virtual_data'

    print(f'{args.device}')
    dataset = PDBbindDataset(ids="%s/%s_ids.npy" % (args.data_dir, args.data_prefix),
                             ligs="%s/%s_lig.pt" % (args.data_dir, args.data_prefix),
                             prots="%s/%s_prot.pt" % (args.data_dir, args.data_prefix)
                             )

    train_inds, val_inds = dataset.train_and_test_split(valnum=args.valnum, seed=args.seeds)

    train_data = PDBbindDataset(ids=dataset.pdbids[train_inds],
                                ligs=dataset.gls[train_inds],
                                prots=dataset.gps[train_inds],
                                labels=dataset.labels[train_inds]
                                )

    val_data = PDBbindDataset(ids=dataset.pdbids[val_inds],
                              ligs=dataset.gls[val_inds],
                              prots=dataset.gps[val_inds],
                              labels=dataset.labels[val_inds]
                              )

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

        for k, v in checkpoint['model_state_dict'].items():
            if k in model_dict:
                model_dict[k] = v

        for name, param in model.named_parameters():
            print(name, param.shape)

        # load model_state_dict
        model.load_state_dict(model_dict)

    optimizer = th.optim.Adam(model.parameters(), lr=10 ** -args.lr, weight_decay=10 ** -args.weight_decay)

    train_loader = DataLoader(dataset=train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers)

    val_loader = DataLoader(dataset=val_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.num_workers)

    stopper = EarlyStopping(patience=args.patience, mode=args.mode, filename=args.model_path)

    ckpt_list = []
    best_loss = 0.0
    max_ckpt_maintain = 10

    for epoch in range(args.num_epochs):
        total_loss_train, train_pr = GIP_train_epoch(epoch,
                                                     model,
                                                     train_loader,
                                                     optimizer,
                                                     affi_weight=args.affi_weight,
                                                     aux_weight=args.aux_weight,
                                                     dist_threhold=args.dist_threhold2,
                                                     device=args.device)

        total_loss_val, val_pr, _, _, _ = GIP_eval_epoch(model,
                                                         val_loader,
                                                         dist_threhold=args.dist_threhold2,
                                                         affi_weight=args.affi_weight,
                                                         aux_weight=args.aux_weight,
                                                         device=args.device)

        if val_pr > best_loss:
                best_loss = val_losses[-1]
                
                state = {
                    "model": model.state_dict(),
                    'cur_epoch': epoch,
                    'best_loss': best_loss,
                }
                epoch = str(epoch) if epoch is not None else ''
                checkpoint = os.path.join(config.train.save_path, 'checkpoint%s' % epoch)

                if len(ckpt_list) >= max_ckpt_maintain:
                    try:
                        os.remove(ckpt_list[0])
                    except:
                        print('Remove checkpoint failed for', ckpt_list[0])
                    ckpt_list = ckpt_list[1:]
                    ckpt_list.append(checkpoint)
                else:
                    ckpt_list.append(checkpoint)

                th.save(state, checkpoint)

        print(
            'epoch {:d}/{:d}, train_loss {:.4f}, total_loss_val {:.4f}, affi_loss_val {:.4f}'.format(
                epoch + 1, args.num_epochs, total_loss_train, total_loss_val, val_pr)
        )  # +' validation result:', validation_result)
