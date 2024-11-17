import torch as th
import torch.nn as nn
import numpy as np
import random
import os
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_add
from sklearn import metrics
from sklearn.metrics import roc_auc_score, mean_squared_error, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr
import tqdm
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from math import sqrt

class EarlyStopping(object):
    """Early stop tracker
	
    Save model checkpoint when observing a performance improvement on
    the validation set and early stop if improvement has not been
    observed for a particular number of epochs.
	
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
	
    Examples
    --------
    Below gives a demo for a fake training process.
	
    >>> import torch
    >>> import torch.nn as nn
    >>> from torch.nn import MSELoss
    >>> from torch.optim import Adam
    >>> from dgllife.utils import EarlyStopping
	
    >>> model = nn.Linear(1, 1)
    >>> criterion = MSELoss()
    >>> # For MSE, the lower, the better
    >>> stopper = EarlyStopping(mode='lower', filename='test.pth')
    >>> optimizer = Adam(params=model.parameters(), lr=1e-3)

    >>> for epoch in range(1000):
    >>>     x = torch.randn(1, 1) # Fake input
    >>>     y = torch.randn(1, 1) # Fake label
    >>>     pred = model(x)
    >>>     loss = criterion(y, pred)
    >>>     optimizer.zero_grad()
    >>>     loss.backward()
    >>>     optimizer.step()
    >>>     early_stop = stopper.step(loss.detach().data, model)
    >>>     if early_stop:
    >>>         break

    >>> # Load the final parameters saved by the model
    >>> stopper.load_checkpoint(model)
    """

    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:
            # dt = datetime.datetime.now()
            filename = 'early_stop.pth'

        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'rp' or 'rs' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['rp', 'rs', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = {"CDK8": 0, "CMET": 0, "EG5": 0, "HIF-2α": 0, "PFKFB3": 0, "SHP2": 0, "SYK": 0, "TNKS2": 0}
        self.counter = {"BACE": 0, "CDK2": 0, "JNK1": 0, "MCL1": 0, "P38": 0, "PTP1B": 0, "Thrombin": 0, "TYK2": 0}
        self.timestep = 0
        self.filename = filename
        self.best_score = {}
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.

        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.

        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.

        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score_dict, model):
        """Update based on a new score.

        The new score is typically model performance on the validation set
        for a new epoch.

        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.

        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        self.timestep += 1

        for target, score in score_dict.items():
            filename = self.filename.split('.')[0] + '_' + target + '.pth'
            if target not in self.best_score:
                self.best_score[target] = score
                self.save_checkpoint(model, filename)

            elif self._check(score, self.best_score[target]):
                self.best_score[target] = score
                self.save_checkpoint(model, filename)
                self.counter[target] = 0
            else:
                self.counter[target] += 1
                print(
                    f'EarlyStopping counter: {self.counter[target]} out of {self.patience}')
                if self.counter[target] >= self.patience:
                    self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model, ckpt_path):
        '''Saves model when the metric on the validation set gets improved.

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        th.save({'model_state_dict': model.state_dict(),
                 'timestep': self.timestep}, ckpt_path)

    def load_checkpoint(self, model):
        '''Load the latest checkpoint

        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        model.load_state_dict(th.load(self.filename)['model_state_dict'])


def screen_train_epoch(epoch, model, data_loader, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                       dist_threhold=None, device='cpu'):
    # loss_fn = nn.MSELoss()
    loss_fn = th.nn.CrossEntropyLoss(label_smoothing=0.0).to(device)
    model.train()
    total_loss = []
    probs = []
    true = []
    for batch_id, batch_data in tqdm(enumerate(data_loader)):
        pdbids, bgp, bgl, labels = batch_data
        # print(f"pdbids is {pdbids}")
        # print(f"bgp is {bgp}")
        # print(f"bgl is {bgl}")
        # print(f"labels is {labels}")
        bgl, bgp = bgl.to(device), bgp.to(device)

        _, _, pred = model(bgl, bgp, train=True)

        # print(f"pred size is {pred}\tlabels size is {labels}")
        loss = loss_fn(pred, labels.long().to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs.append(pred.detach())
        true.append(labels.detach())
        # total_loss += loss.item() * batch.unique().size(0)
        total_loss.append(loss.detach())

        del loss, labels, pred, bgl, bgp, batch_data, pdbids
        th.cuda.empty_cache()

        # for name, param in model.named_parameters():
        #     print(name, param)

    return total_loss, probs, true


def screen_eval_epoch(model, data_loader, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                      dist_threhold=None, device='cpu'):
    # loss_fn = nn.MSELoss()
    loss_fn = th.nn.CrossEntropyLoss(label_smoothing=0.0).to(device)
    model.eval()
    total_loss = []
    probs = []
    true = []

    with th.no_grad():
        for batch_id, batch_data in tqdm(enumerate(data_loader)):
            pdbids, bgp, bgl, labels = batch_data
            bgl, bgp = bgl.to(device), bgp.to(device)

            _, _, pred = model(bgl, bgp, train=False)

            loss = loss_fn(pred, labels.long().to(device))

            pred = th.softmax(pred, dim=-1)[:, 1]

            probs.append(pred.detach().cpu().numpy())
            true.append(labels.detach().cpu().numpy())
            # total_loss += loss.item() * batch.unique().size(0)
            total_loss.append(loss.detach())

            del loss, labels, pred, bgl, bgp, batch_data, pdbids
            th.cuda.empty_cache()

    return total_loss, probs, true


def train_compre(epoch, model, data_loader1, data_loader2, optimizer, device='cpu'):
    pdb_iter, screen_iter = iter(data_loader1), iter(data_loader2)
    loss_fn = nn.MSELoss()
    model.train()
    probs = []
    true = []
    total_loss = 0.0

    while True:
        sample1 = next(pdb_iter, None)
        if sample1 is None:
            break

        pdbids_pdb, bgp_pdb, bgl_pdb, labels_pdb = sample1

        bgl_pdb, bgp_pdb = bgl_pdb.to(device), bgp_pdb.to(device)

        vdw_pred_pdb, vdw_radiu_pdb, pred_pdb, mdn_pdb, mdn_pred_pdb = model(bgl_pdb, bgp_pdb, train=True)
        pred_pdb = pred_pdb.squeeze(-1)
        mdn_pred_pdb = mdn_pred_pdb.squeeze(-1)

        labels_pdb = labels_pdb.float().to(device)
        vdw_pdb = labels_pdb * 0.8

        loss1_pdb = abs(loss_fn(vdw_pred_pdb, vdw_pdb) - 0.5)
        loss2_pdb = loss_fn(pred_pdb, labels_pdb)
        loss_mdn_pdb = th.corrcoef(th.stack([mdn_pred_pdb, labels_pdb]))[1, 0].item()

        loss_pdb = loss2_pdb + (mdn_pdb - loss_mdn_pdb) * 0.5 + 0.05 * loss1_pdb

        sample2 = next(screen_iter, None)

        pdbids_screen, bgp_screen, bgl_screen, labels_screen = sample2

        bgl_screen, bgp_screen = bgl_screen.to(device), bgp_screen.to(device)

        vdw_pred_screen, vdw_radius_screen, pred_screen, mdn_screen, mdn_pred_screen = model(bgl_screen, bgp_screen, train=True)
        pred_screen = pred_screen.squeeze(-1)
        mdn_pred_screen = mdn_pred_screen.squeeze(-1)

        labels_screen = labels_screen.float().to(device)
        vdw_screen = labels_screen * 0.8

        loss1_screen = abs(loss_fn(vdw_pred_screen, vdw_screen) + 1)
        loss2_screen = abs(loss_fn(pred_screen, labels_screen) + 1)
        loss_mdn_screen = th.corrcoef(th.stack([mdn_pred_screen, labels_screen]))[1, 0].item() - 0.2

        loss_screen = loss2_screen + (mdn_screen - loss_mdn_screen) * 0.5 + 0.05 * loss1_screen

        loss_all = loss_pdb + 0.01 * loss_screen

        optimizer.zero_grad()
        loss_all.backward()
        optimizer.step()

        probs.append(pred_pdb.detach())
        true.append(labels_pdb)
        # total_loss += loss.item() * batch.unique().size(0)
        total_loss += loss_all .detach()

        del loss_all, labels_pdb, labels_screen, pred_pdb, pred_screen, bgl_pdb, bgl_screen, bgp_screen, bgp_pdb, \
            sample1, sample2, pdbids_pdb, pdbids_screen, vdw_pred_pdb, vdw_pred_screen
        th.cuda.empty_cache()

    ys = th.cat(probs)
    ps = th.cat(true)
    affi_pr = th.corrcoef(th.stack([ys, ps]))[1, 0].item()

    return total_loss, affi_pr

def GIP_semi_label(epoch, model, data_loader1, data_loader2, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                    dist_threhold=None, device='cpu'):
    iter1, iter2 = iter(data_loader1), iter(data_loader2)
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = 0
    mdn_loss = 0
    affi_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    true = []

    while True:
        # sample fep or derivate data for virtual labels
        sample_vir = next(iter2, None)

        if sample_vir is None:
            break

        pdbids_vir, bgp_vir, bgl_vir, labels_vir = sample_vir

        bgl_vir, bgp_vir = bgl_vir.to(device), bgp_vir.to(device)

        vdw_pred_vir, vdw_radius_vir, pred_vir, mdn_vir, mdn_pred_vir = model(bgl_vir, bgp_vir, train=True)
        pred_vir = pred_vir.squeeze(-1)
        mdn_pred_vir = mdn_pred_vir.squeeze(-1)

        labels_vir = labels_vir.float().to(device)
        vdw_vir = labels_vir * 0.8

        loss1_vir = abs(loss_fn(vdw_pred_vir, vdw_vir) - 0.5)
        loss2_vir = abs(loss_fn(pred_vir, labels_vir) + 0.15)
        loss_mdn_vir = th.corrcoef(th.stack([mdn_pred_vir, labels_vir]))[1, 0].item() - 0.05

        loss_vir = loss2_vir + (mdn_vir - loss_mdn_vir) * 0.5 + 0.05 * loss1_vir

        # sample pdbbind2020 with real label

        sample_real = next(iter1, None)

        pdbids_real, bgp_real, bgl_real, labels_real = sample_real

        bgl_real, bgp_real = bgl_real.to(device), bgp_real.to(device)

        vdw_pred_real, vdw_radius_real, pred_real, mdn_real, mdn_pred_real = model(bgl_real, bgp_real, train=True)
        pred_real = pred_real.squeeze(-1)
        mdn_pred_real = mdn_pred_real.squeeze(-1)

        labels_real = labels_real.float().to(device)
        vdw_real = labels_real * 0.8

        loss1_real = abs(loss_fn(vdw_pred_real, vdw_real) - 0.5)
        loss2_real = loss_fn(pred_real, labels_real)
        loss_mdn_real = th.corrcoef(th.stack([mdn_pred_real, labels_real]))[1, 0].item()

        loss_real = loss2_real + (mdn_real - loss_mdn_real) * 0.5 + 0.05 * loss1_real

        loss = loss_real + 0.5 * loss_vir

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs.append(pred_vir.detach())
        true.append(labels_vir)
        # total_loss += loss.item() * batch.unique().size(0)
        total_loss += loss.detach()


        del loss, labels_real, labels_vir, pred_vir, pred_real, bgl_real, bgl_vir, bgp_real, bgp_vir, \
            sample_real, sample_vir, pdbids_real, pdbids_vir, vdw_pred_real, vdw_pred_vir
        th.cuda.empty_cache()

    ys = th.cat(probs)
    ps = th.cat(true)
    affi_pr = th.corrcoef(th.stack([ys, ps]))[1, 0].item()

    return total_loss, affi_pr


def GIP_train_epoch(epoch, model, data_loader, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                    dist_threhold=None, device='cpu'):
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = 0
    mdn_loss = 0
    affi_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    true = []
    for batch_id, batch_data in enumerate(data_loader):
        pdbids, bgp, bgl, labels = batch_data

        bgl, bgp = bgl.to(device), bgp.to(device)

        vdw_pred, vdw_radius, pred, mdn, mdn_pred, _ = model(bgl, bgp, train=True)
        pred = pred.squeeze(-1)
        mdn_pred = mdn_pred.squeeze(-1)

        labels = labels.float().to(device)
        vdw = labels * 0.8

        loss1 = abs(loss_fn(vdw_pred, vdw) - 0.5)
        loss2 = loss_fn(pred, labels)
        loss_radius = vdw_radius.pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probs.append(pred.detach())
        true.append(labels)
        # total_loss += loss.item() * batch.unique().size(0)
        total_loss += loss.detach()

        del loss, labels, pred, bgl, bgp, batch_data, pdbids, vdw_pred
        th.cuda.empty_cache()

        # for name, param in model.named_parameters():
        #     print(name, param)

    ys = th.cat(probs)
    ps = th.cat(true)
    affi_pr = th.corrcoef(th.stack([ys, ps]))[1, 0].item()

    return total_loss / len(data_loader.dataset), affi_pr


def GIP_eval_epoch(model, data_loader, pred=False, atom_contribution=False, res_contribution=False,
                   dist_threhold=None, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001, device='cpu'):
    loss_fn = nn.MSELoss()
    model.eval()
    total_loss = 0
    mdn_loss = 0
    affi_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    true = []
    atom_probs = []
    with th.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            pdbids, bgp, bgl, labels = batch_data
            bgl, bgp = bgl.to(device), bgp.to(device)
            # print(pdbids)
            _, _, pred, _, mdn_pred, atom_energy = model(bgl, bgp, train=False)
            pred = pred.squeeze(-1)

            labels = labels.float().to(device)

            loss = loss_fn(pred, labels)

            probs.append(pred.detach())
            true.append(labels)
            atom_probs.append(atom_energy.detach())
            # total_loss += loss.item() * batch.unique().size(0)
            total_loss += loss.detach()

            del loss, labels, pred, bgl, bgp, batch_data, pdbids
            th.cuda.empty_cache()

    ys = th.cat(probs)
    ps = th.cat(true)
    # atom_pair_energy = th.cat(atom_probs)
    # print(f"true is {ps}")
    # print(f"pred is {ys}")
    affi_pr = th.corrcoef(th.stack([ys, ps]))[1, 0].item()
    corr, p_value = spearmanr(ys.cpu().detach().numpy(), ps.cpu().detach().numpy())
    rmse = sqrt(mean_squared_error(ys.cpu().detach().numpy(), ps.cpu().detach().numpy()))
    print(f"corr is {corr} \t p is {p_value} \t pr is {affi_pr}")
    return corr, affi_pr, ys.cpu().numpy(), rmse, atom_energy


def run_a_train_epoch(epoch, model, data_loader, optimizer, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001,
                      dist_threhold=None, device='cpu'):
    loss_fn = nn.MSELoss()
    model.train()
    total_loss = 0
    mdn_loss = 0
    affi_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    for batch_id, batch_data in tqdm(enumerate(data_loader)):
        pdbids, bgp, bgl, labels = batch_data
        bgl, bgp = bgl.to(device), bgp.to(device)

        atom_labels = th.argmax(bgl.x[:, :17], dim=1, keepdim=False)
        bond_labels = th.argmax(bgl.edge_attr[:, :4], dim=1, keepdim=False)

        pi, sigma, mu, dist, atom_types, bond_types, batch = model(bgl, bgp, train=True)

        mdn, prob = mdn_loss_fn(pi, sigma, mu, dist)
        mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
        mdn = mdn.mean()

        batch = batch.to(device)
        if dist_threhold is not None:
            prob = prob[th.where(dist <= dist_threhold)[0]]
            y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]], dim=0, dim_size=batch.unique().size(0)).to(
                th.float32)
        else:
            y = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0)).to(th.float32)

        # labels = labels.float().to(device)
        # affi = loss_fn(y, labels)
        labels = labels.float().type_as(y).to(device)
        affi = th.corrcoef(th.stack([y, labels]))[1, 0]
        atom = F.cross_entropy(atom_types, atom_labels)
        bond = F.cross_entropy(bond_types, bond_labels)
        loss = (mdn * mdn_weight) + (affi * affi_weight) + (atom * aux_weight) + (bond * aux_weight)

        probs.append(y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # total_loss += loss.item() * batch.unique().size(0)
        mdn_loss += mdn.item() * batch.unique().size(0)
        # affi_loss += affi.item() * batch.unique().size(0)
        atom_loss += atom.item() * batch.unique().size(0)
        bond_loss += bond.item() * batch.unique().size(0)
        total_loss += mdn_loss + atom_loss * aux_weight + bond_loss * aux_weight + affi * affi_weight

        # print('Step, Total Loss: {:.3f}, MDN: {:.3f}'.format(total_loss, mdn_loss))
        if np.isinf(mdn_loss) or np.isnan(mdn_loss): break

        del loss, labels, y, bgl, bgp, batch_data, pdbids
        th.cuda.empty_cache()

    ys = th.cat(probs)
    affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1, 0].item()

    return total_loss / len(data_loader.dataset), mdn_loss / len(
        data_loader.dataset), affi_loss, atom_loss / len(data_loader.dataset), bond_loss / len(data_loader.dataset)


def run_an_eval_epoch(model, data_loader, pred=False, atom_contribution=False, res_contribution=False,
                      dist_threhold=None, mdn_weight=1.0, affi_weight=1.0, aux_weight=0.001, device='cpu'):
    model.eval()
    total_loss = 0
    affi_loss = 0
    mdn_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    at_contrs = []
    res_contrs = []
    with th.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            pdbids, bgp, bgl, labels = batch_data
            bgl, bgp = bgl.to(device), bgp.to(device)
            atom_labels = th.argmax(bgl.x[:, :17], dim=1, keepdim=False)
            bond_labels = th.argmax(bgl.edge_attr[:, :4], dim=1, keepdim=False)

            pi, sigma, mu, dist, atom_types, bond_types, batch = model(bgl, bgp, train=False)

            if pred or atom_contribution or res_contribution:
                prob = calculate_probablity(pi, sigma, mu, dist)
                if dist_threhold is not None:
                    prob[th.where(dist > dist_threhold)[0]] = 0.

                batch = batch.to(device)
                if pred:
                    probx = scatter_add(prob, batch, dim=0, dim_size=batch.unique().size(0))
                    probs.append(probx)
                if atom_contribution or res_contribution:
                    contribs = [prob[batch == i].reshape(len(bgl.x[bgl.batch == i]), len(bgp.x[bgp.batch == i])) for i
                                in th.arange(0, len(batch.unique()))]
                    if atom_contribution:
                        at_contrs.extend(
                            [contribs[i].sum(1).cpu().detach().numpy() for i in th.arange(0, len(batch.unique()))])
                    if res_contribution:
                        res_contrs.extend(
                            [contribs[i].sum(0).cpu().detach().numpy() for i in th.arange(0, len(batch.unique()))])

            else:
                mdn, prob = mdn_loss_fn(pi, sigma, mu, dist)
                mdn = mdn[th.where(dist <= model.dist_threhold)[0]]
                mdn = mdn.mean()

                batch = batch.to(device)
                if dist_threhold is not None:
                    prob = prob[th.where(dist <= dist_threhold)[0]]
                    y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]], dim=0,
                                    dim_size=batch.unique().size(0))
                else:
                    y = scatter_add(prob, batch[th.where(dist <= dist_threhold)[0]], dim=0,
                                    dim_size=batch.unique().size(0))
                labels = labels.float().type_as(y).to(device)

                affi = th.corrcoef(th.stack([y, labels]))[1, 0]

                atom = F.cross_entropy(atom_types, atom_labels)
                bond = F.cross_entropy(bond_types, bond_labels)
                # loss = (mdn * mdn_weight) + (atom * aux_weight) + (bond * aux_weight)
                loss = mdn + affi * affi_weight + (atom * aux_weight) + (bond * aux_weight)

                probs.append(y)

                total_loss += loss.item() * batch.unique().size(0)
                mdn_loss += mdn.item() * batch.unique().size(0)
                atom_loss += atom.item() * batch.unique().size(0)
                bond_loss += bond.item() * batch.unique().size(0)

    if atom_contribution or res_contribution:
        if pred:
            preds = th.cat(probs)
            affi_loss = th.corrcoef(th.stack([ys, data_loader.dataset.labels.to(device)]))[1, 0].item()
            print(f"pr is {affi_loss}")
            return [preds.cpu().detach().numpy(), at_contrs, res_contrs]
        else:
            return [None, at_contrs, res_contrs]
    else:
        if pred:
            preds = th.cat(probs)
            corr, p_value = spearmanr(preds.cpu().detach().numpy(), data_loader.dataset.labels.cpu().detach().numpy())
            affi_loss = th.corrcoef(th.stack([preds, data_loader.dataset.labels.to(device)]))[1, 0].item()
            print(f"pr is {affi_loss}")
            print(f"spearman is {corr}")
            return preds.cpu().detach().numpy()
        else:
            preds = th.cat(probs)
            corr, p_value = spearmanr(preds.cpu().detach().numpy(), data_loader.dataset.labels.cpu().detach().numpy())
            affi_loss = th.corrcoef(th.stack([preds, data_loader.dataset.labels.to(device)]))[1, 0].item()
            # print(f"ys is {preds}")
            print(f"pr is {affi_loss}")
            print(f"spearman is {corr}")
            # total_loss += - affi_loss
            # del preds
            # return total_loss / len(data_loader.dataset) + affi_loss * affi_weight, mdn_loss / len(
            #     data_loader.dataset), affi_loss, atom_loss / len(data_loader.dataset), bond_loss / len(
            #     data_loader.dataset)
            return corr, affi_loss, preds.cpu().numpy()


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.deterministic=True
    th.backends.cudnn.enabled = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    # if th.cuda.is_available():
    #     th.cuda.manual_seed(seed)
    #     th.cuda.manual_seed_all(seed)


def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += th.log(pi)
    prob = logprob.exp().sum(1)

    return prob


def mdn_loss_fn(pi, sigma, mu, y, eps1=1e-10, eps2=1e-10):
    normal = Normal(mu, sigma)
    # loss = th.exp(normal.log_prob(y.expand_as(normal.loc)))
    # loss = th.sum(loss * pi, dim=1)
    # loss = -th.log(loss)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    # loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)
    prob = (th.log(pi + eps1) + loglik).exp().sum(1)
    loss = -th.log(prob + eps2)
    return loss, prob
