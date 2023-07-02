import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad as torch_grad
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mean_pooling(tokens, aug_tokens, sent_orig, sent_aug):
    '''
    Pre-processing (mean pooling)
    '''
    batch_size = tokens.size(0)
    num_mask = (tokens != 1).float().sum(dim=1).long()
    num_mask_aug = (aug_tokens != 1).float().sum(dim=1).long()
    mean_sent_orig, mean_sent_aug = [], []

    for b in range(batch_size):
        mean_sent_orig_b = sent_orig[b, :num_mask[b], :].mean(dim=0).unsqueeze(0)  # (1, dim)
        mean_sent_aug_b = sent_aug[b, :num_mask_aug[b], :].mean(dim=0).unsqueeze(0)  # (1, dim)

        mean_sent_orig.append(mean_sent_orig_b)
        mean_sent_aug.append(mean_sent_aug_b)

    mean_sent_orig = torch.cat(mean_sent_orig, dim=0)  # (batch_size, dim)
    mean_sent_aug = torch.cat(mean_sent_aug, dim=0)  # (batch_size, dim)

    return mean_sent_orig, mean_sent_aug

def sent_similarity(args, model, tokens, aug_tokens, sent_orig, sent_aug):
    '''
    Goal: modeling a sentence similarity
    '''
    mean_sent_orig, mean_sent_aug = mean_pooling(tokens, aug_tokens, sent_orig, sent_aug)
    loss_sent, acc_sent = siamese(model, mean_sent_orig, mean_sent_aug)

    return loss_sent, acc_sent

def siamese(model, orig_sent, aug_sent, temperature=1.0):
    '''
    Learning sentence similarity with siamese network: https://arxiv.org/abs/1908.10084
    '''
    batch_size = orig_sent.size(0)
    # Pairing for positive and negative pairs
    u = orig_sent
    v_p = aug_sent

    shuffle_idx = (batch_size - 1) - torch.arange(batch_size)
    v_n = aug_sent[shuffle_idx, :]

    pos_sent_output = torch.cat([u, v_p, torch.abs(u - v_p)], dim=1)  # (batch_size, 3*dim)
    pos_labels = torch.zeros(batch_size)
    neg_sent_output = torch.cat([u, v_n, torch.abs(u - v_n)], dim=1)  # (batch_size, 3*dim)
    neg_labels = torch.ones(batch_size)

    # Concat & calculating loss
    sent_output = torch.cat([pos_sent_output, neg_sent_output], dim=0)  # (2*batch_size, 3*dim)
    sent_labels = torch.cat([pos_labels, neg_labels], dim=0).cuda().long()

    if torch.cuda.device_count() > 1:
        out_sent = model.module.net_sent(sent_output)
    else:
        out_sent = model.net_sent(sent_output)

    loss_sent = F.cross_entropy(out_sent / temperature, sent_labels)

    # Logging
    _, pred_sent = out_sent.max(dim=1)
    corrects_sent = (pred_sent == sent_labels).float()
    acc_sent = corrects_sent.sum() / (2 * batch_size)

    return loss_sent, acc_sent

def lm_recon(args, model, tokens, outputs_sent):
    '''
    Language modeling reconstruction
    '''
    batch_size = tokens.size(0)
    if torch.cuda.device_count() > 1:
        if 'roberta' in args.backbone:
            LM_head = model.module.backbone.lm_head
        else:
            LM_head = model.module.backbone.cls
    else:
        if 'roberta' in args.backbone:
            LM_head = model.backbone.lm_head
        else:
            LM_head = model.backbone.cls

    if 'roberta' in args.backbone:
        attention_mask = (tokens != 1).float().cpu()
    else:
        attention_mask = (tokens > 0).float().cpu()

    num_tokens = attention_mask.sum(dim=1).long()
    attention_mask[:, 0] = 0
    attention_mask[torch.arange(batch_size), num_tokens - 1] = 0

    mask = torch.ones(tokens.size()) * attention_mask.cpu()  # B x L
    num_mask = (num_tokens - 2).cuda() # except start, end tokens
    mask_idx = mask.nonzero()

    labels_ssl = -1 * torch.ones(tokens.size()).to(device).long()  # Sampled : 1, not sampled : -1
    labels_ssl[mask_idx[:, 0], mask_idx[:, 1]] = tokens[mask_idx[:, 0], mask_idx[:, 1]]

    out_ssl = LM_head(outputs_sent)
    out_ssl = out_ssl.permute(0, 2, 1)

    # lm_acc
    _, pred_ssl = out_ssl.max(dim=1)
    mask_ssl = (labels_ssl != -1).float()
    corrects = (pred_ssl == labels_ssl).float() * mask_ssl
    acc_ssl = corrects.sum() / num_mask.sum()

    # self-supervision loss
    loss_ssl = F.cross_entropy(out_ssl, labels_ssl, ignore_index=-1, reduction='none')  # ignore non-masks (-1)
    loss_ssl = loss_ssl.sum(1) / num_mask

    return loss_ssl.mean(), acc_ssl, num_mask.sum()