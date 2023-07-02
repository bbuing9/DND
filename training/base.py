import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training.common import AverageMeter, one_hot, cut_input, get_embed, data_aug, sym_kld
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_base(args, loader, model, optimizer, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for i, (tokens, labels, _) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)
        tokens, _ = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)

        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)

        out_cls = model(tokens)
        loss = criterion(out_cls, labels).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # cls_acc
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)


    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_aug(args, loader, model, optimizer, bts_src, ctx_src, eda_src, epoch=0, logger=None, policy=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()
    losses['aug'] = AverageMeter()
    losses['aug_acc'] = AverageMeter()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    for i, (tokens, labels, indices) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)

        tokens = tokens.to(device)
        labels = labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)
        embed = get_embed(args, model, tokens)

        # Augmentation
        backtrans_aug = bts_src[indices]
        contextual_aug = ctx_src[indices]
        eda_aug = eda_src[indices]

        tokens, embed, aug_tokens, aug_embed = data_aug(args, model, tokens, embed, labels, backtrans_aug, contextual_aug, eda_aug, policy)

        out_cls = model(tokens, inputs_embed=None)
        out_aug = model(aug_tokens, inputs_embed=aug_embed)

        # Total loss
        loss_cls = criterion(out_cls, labels).mean()
        loss_aug = criterion(out_aug, labels).mean()
        loss_symkld = sym_kld(out_aug, out_cls).mean()

        loss = loss_cls + args.lambda_aug * loss_aug + args.lambda_kl * loss_symkld

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        _, pred_aug = out_aug.max(dim=1)
        corrects_aug = (pred_aug == labels).float()
        acc_aug = corrects_aug.sum() / batch_size

        losses['cls'].update(loss_cls.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)
        losses['aug'].update(loss_aug.item(), batch_size)
        losses['aug_acc'].update(acc_aug.item(), batch_size)

    msg = '[Epoch %2d] [Accuracy Orig %.3f] [Accuracy Aug %.3f] [Loss Orig %.3f] [Loss Aug %.3f]' \
          % (epoch, losses['cls_acc'].average, losses['aug_acc'].average,  losses['cls'].average, losses['aug'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_mixup(args, loader, model, optimizer, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    for i, (tokens, labels, indices) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)
        tokens, attention_mask = cut_input(args, tokens)

        tokens = tokens.to(device)
        labels = labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)
        embed = get_embed(args, model, tokens)

        # Mixup
        if args.mixup_alpha == 0:
            l = 1
        else:
            l = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            l = max(l, 1 - l)
        idx = torch.randperm(embed.size(0))
        labels_a, labels_b = labels, labels[idx]

        embed_a, embed_b = embed, embed[idx]
        mixed_embed = l * embed_a + (1 - l) * embed_b

        out_cls = model(tokens, inputs_embed=mixed_embed)  # (B, C)

        # mixed loss
        if args.dataset != 'stsb':
            loss = l * F.cross_entropy(out_cls, labels_a) + (1 - l) * F.cross_entropy(out_cls, labels_b)
        else:
            labels = l * labels_a.float() + (1 - l) * labels_b.float()
            loss = F.mse_loss(out_cls, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # cls_acc
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels_a).float()
        corrects2 = (pred_cls == labels_b).float()
        acc_cls = (corrects.sum() + corrects2.sum()) / (2 * batch_size)

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)