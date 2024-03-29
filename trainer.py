import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import skimage.io
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm

import torchfcn

import sys

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, average_precision_score 
import matplotlib.pyplot as plt

fp = open('temp.txt', 'w')

def manual_roc_curve(y_true, y_prob, thresholds):

    fpr = []
    tpr = []

    for threshold in thresholds:

        y_pred = np.where(y_prob >= threshold, 1, 0)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        tp = np.sum((y_pred == 1) & (y_true == 1))

        fn = np.sum((y_pred == 0) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    return [fpr, tpr]


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, 2, h, w), target: (n, 1, h, w)
    print(f'cross input shape: {input.shape}')
    print(f'cross target shape: {target.shape}')
    n, c, h, w = input.size()
    _, _, th, tw = target.size()
    # log_p: (n, 2, h, w)
    if LooseVersion(torch.__version__) < LooseVersion('0.3'):
        # ==0.2.X
        log_p = F.log_softmax(input)
    else:
        # >=0.3
        log_p = F.log_softmax(input, dim=1)
    # log_p: (n*h*w, c)
    if h != th and w != tw:
        import pdb; pdb.set_trace()
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, infer_loader, out, max_iter,
                 size_average=False, interval_validate=None, arch_name='', resume=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.resume = resume
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.infer_loader = infer_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.arch_name = arch_name

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target, filename) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                score = self.model(data)

            score = score['out']
            target = target.long()
            # import pdb; pdb.set_trace()

            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu().numpy()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                # if self.val_loader.dataset.untransform:
                #     img, lt = self.val_loader.dataset.untransform(img, lt)
                lt = lt[0]
                img = np.moveaxis(img, 0, -1)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        skimage.io.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()

    def evaluate(self):
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds, label_preds_adx = [], [], []
        for batch_idx, (data, target, filename) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                score = self.model(data)

            score = score['out']
            target = target.long()

            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu().numpy()
            el = np.where(score.cpu().max(1) == 0, score.cpu()[0][0], score.cpu()[0][1])
            lbl_pred_adx = np.expand_dims(np.where(el < 0, 0, el), 0)
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for img, lt, lp, lpadx in zip(imgs, lbl_true, lbl_pred, lbl_pred_adx):
                # if self.val_loader.dataset.untransform:
                #     img, lt = self.val_loader.dataset.untransform(img, lt)
                lt = lt[0]
                img = np.moveaxis(img, 0, -1)
                label_trues.append(lt)
                label_preds.append(lp)
                label_preds_adx.append(lpadx)
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, n_class)


        _x = np.array(label_trues).ravel()
        _y = np.array(label_preds_adx).ravel()

        roc_fpr, roc_tpr, roc_thres = roc_curve(_x, _y, pos_label=1)
        roc_auc = auc(roc_fpr, roc_tpr)
        manroc_fpr, manroc_tpr = manual_roc_curve(_x, _y, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.])
        manroc_auc = auc(manroc_fpr, manroc_tpr)
        pr_precision, pr_recall, pr_thres = precision_recall_curve(_x, _y, pos_label=1)
        pr_auc = auc(pr_recall, pr_precision)

        ap = average_precision_score(np.array(label_trues).ravel(), np.array(label_preds_adx).ravel(), pos_label=1)
        print(f'ap : {ap}')


        out = osp.join('.', 'evaluation', self.resume)
        if not osp.exists(out):
            os.makedirs(out)

        fig, ax = plt.subplots(1,1)
        # fig.set_size_inches(12, 27)
        print(f'roc_auc : {roc_auc}')
        print(f'pr_auc : {pr_auc}')
        ax.plot(roc_fpr, roc_tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('FCN Receiver Operating Characteristic curve')
        ax.legend(loc="lower right")

        fig.savefig(osp.join(out, 'roc-curve'))

        fig, ax = plt.subplots(1,1)
        # fig.set_size_inches(12, 27)
        ax.plot(manroc_fpr, manroc_tpr, label='ROC curve (area = %0.2f)' % manroc_auc)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('FCN Receiver Operating Characteristic curve')
        ax.legend(loc="lower right")

        fig.savefig(osp.join(out, 'roc-curve-only-ten'))



        fig, ax = plt.subplots(1,1)
        # fig.set_size_inches(12, 27)
        ax.plot(pr_precision, pr_recall, label='PR curve (area = %0.2f)' % pr_auc)
        ax.plot([1, 0], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('FCN Precision Recall curve')
        ax.legend(loc="lower right")

        fig.savefig(osp.join(out, 'pr-curve'))

        val_loss /= len(self.val_loader)

        with open(osp.join(out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [ap] + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
            
    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target, filename) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            score = self.model(data)

            score = score['out']
            
            target = target.long()
            if not data.shape[-2:] == target.shape[-2:]:
                print(batch_idx, filename, file=fp)
                print(data.shape, score.shape, target.shape, file=fp)
                continue

            # import pdb; pdb.set_trace()
            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc = \
                torchfcn.utils.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Seoul')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break

    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            try:
                self.train_epoch()
                if self.iteration >= self.max_iter:
                    break
            except Exception as err:
                with open('error.log', 'w') as fp:
                    fp.write(err)

    def inference(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.infer_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target, filename) in tqdm.tqdm(
                enumerate(self.infer_loader), total=len(self.infer_loader),
                desc='Infer iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            with torch.no_grad():
                score = self.model(data)

            score = score['out']
            target = target.long()
            # import pdb; pdb.set_trace()

            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu().numpy()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                # if self.val_loader.dataset.untransform:
                #     img, lt = self.val_loader.dataset.untransform(img, lt)
                lt = lt[0]
                img = np.moveaxis(img, 0, -1)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, n_class)

        out = osp.join(self.out, self.arch_name)
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, f'infer_{self.arch_name}.jpg')
        skimage.io.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.infer_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Tokyo')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')


