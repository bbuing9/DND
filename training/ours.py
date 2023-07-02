import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from training.common import AverageMeter, one_hot, cut_input, cut_aug_input, get_embed, sym_kld
from training.sent_sim import lm_recon, sent_similarity
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_dnd(args, loader, model, optimizer, bts_src, ctx_src, eda_src, policy, policy_optimizer, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'], losses['aug'], losses['sim'], losses['recon'] = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    losses['cls_acc'], losses['aug_acc'], losses['sim_acc'], losses['recon_acc'] = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')

    select_idx = np.random.randint(0, 10, (1,))[0]

    n_iter = (epoch - 1) * len(loader)
    for i, (tokens, labels, indices) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)

        tokens = tokens.to(device)
        tokens2 = tokens.clone()
        labels = labels.to(device)
        if args.dataset != 'stsb':
            labels = labels.squeeze(1)  # (B)
        indices = indices.long().to(device)
        embed = get_embed(args, model, tokens)

        # Augmentation
        backtrans_aug = bts_src[indices]
        contextual_aug = ctx_src[indices]
        eda_aug = eda_src[indices]

        aug_tokens, aug_embeds = policy(args, tokens, embed, labels, backtrans_aug, contextual_aug, eda_aug, model)  # (B, L, V)

        # Cutting for efficiency
        tokens, embed, aug_tokens, aug_embeds = cut_aug_input(args, tokens, embed, aug_tokens, aug_embeds)

        # Forward pass
        out_cls, sent_orig = model(tokens, inputs_embed=embed, get_embeds=True)
        out_aug, sent_aug = model(aug_tokens, inputs_embed=aug_embeds, get_embeds=True)

        #################### Classification loss ####################
        loss_cls = args.lambda_cls * criterion(out_cls, labels)
        loss_aug = args.lambda_aug * criterion(out_aug, labels)

        if args.reweight:
            p_orig = out_cls.softmax(dim=1)[torch.arange(batch_size), labels].detach()
            p_aug = out_aug.softmax(dim=1)[torch.arange(batch_size), labels].clone().detach()
            w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0))

            if w_aug.sum() > 0:
                w_aug /= (w_aug.mean().detach() + 1e-6)
            else:
                w_aug = 1

            loss = args.lambda_cls * loss_cls + w_aug * loss_aug
        else:
            loss = args.lambda_cls * loss_cls + loss_aug

        loss = loss.mean()

        #################### Similarity loss ####################
        loss_sim, acc_sim = sent_similarity(args, model, tokens, aug_tokens, sent_orig, sent_aug)
        loss += loss_sim

        #################### Reconstruction loss ####################
        loss_recon, acc_recon, num_masks = lm_recon(args, model, tokens, sent_orig)
        loss += args.lambda_recon * loss_recon

        # Update classifier
        policy_optimizer.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #################### Policy ####################
        if (n_iter % args.policy_update) == 0:
            embed2 = get_embed(args, model, tokens2)
            aug_tokens2, aug_embeds2 = policy(args, tokens2, embed2, labels, backtrans_aug, contextual_aug, eda_aug, model)  # (B, L, V)

            # Cutting for efficiency
            tokens2, embed2, aug_tokens2, aug_embeds2 = cut_aug_input(args, tokens2, embed2, aug_tokens2, aug_embeds2)

            out_aug2, sent_aug2 = model(aug_tokens2, inputs_embed=aug_embeds2, get_embeds=True)
            loss_sim2, _ = sent_similarity(args, model, tokens, aug_tokens2, sent_orig.detach(), sent_aug2)

            if args.reweight:
                p_orig = out_cls.softmax(dim=1)[torch.arange(batch_size), labels].detach()
                p_aug = out_aug2.softmax(dim=1)[torch.arange(batch_size), labels].clone().detach()
                w_aug = torch.sqrt(p_orig * torch.clamp(p_orig - p_aug, min=0))

                if w_aug.sum() > 0:
                    w_aug /= (w_aug.mean().detach() + 1e-6)
                else:
                    w_aug = 1

                loss_policy = -1 * (w_aug * args.lambda_aug * criterion(out_aug2, labels)).mean()
            else:
                loss_policy = -1 * (args.lambda_aug * criterion(out_aug2, labels)).mean()

            w_rel = (-1 * loss_policy / (1e-6 + loss_sim2)).clone().data

            loss_policy += w_rel * args.lambda_sim * loss_sim2
            
            # Update policy
            policy_optimizer.zero_grad()
            optimizer.zero_grad()
            loss_policy.backward()
            policy_optimizer.step()

        # Logging
        n_iter += 1
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        _, pred_aug = out_aug.max(dim=1)
        corrects_aug = (pred_aug == labels).float()
        acc_aug = corrects_aug.sum() / batch_size

        losses['cls'].update(loss_cls.mean().item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)
        losses['aug'].update(loss_aug.mean().item(), batch_size)
        losses['aug_acc'].update(acc_aug.item(), batch_size)

        losses['sim'].update(loss_sim.item(), 2 * batch_size)
        losses['sim_acc'].update(acc_sim.item(), 2 * batch_size)
        losses['recon'].update(loss_recon.item(), batch_size)
        losses['recon_acc'].update(acc_recon.item(), num_masks)

    msg = '[Epoch %2d] [Acc Orig %.3f] [Acc Aug %.3f] [Acc Sim %.3f] [Acc Recon %.3f] [L_Orig %.3f] [L_Aug %.3f] [L_Sim %.3f] [L_Recon %.3f]'\
          % (epoch, losses['cls_acc'].average, losses['aug_acc'].average,  losses['sim_acc'].average, losses['recon_acc'].average,
             losses['cls'].average, losses['aug'].average, losses['sim'].average, losses['recon'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)