import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
import math

import operator
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count

def one_hot(labels, n_classes):
    one_hot = torch.zeros(labels.size(0), n_classes).to(device)
    one_hot[torch.arange(labels.size(0)), labels] = 1
    return one_hot

def uniform_labels(labels, n_classes):
    unif = torch.ones(labels.size(0), n_classes).to(device)
    return unif / n_classes

def cut_input(args, tokens):
    if 'roberta' in args.backbone:
        attention_mask = (tokens != 1).float()
    else:
        attention_mask = (tokens > 0).float()
    max_len = int(torch.max(attention_mask.sum(dim=1)))
    return tokens[:, :max_len], attention_mask[:, :max_len]

def cut_aug_input(args, tokens, embeds, aug_tokens, aug_embeds):
    if 'roberta' in args.backbone:
        attention_mask = (tokens != 1).float()
        attention_mask2 = (aug_tokens != 1).float()
    else:
        attention_mask = (tokens > 0).float()
        attention_mask2 = (aug_tokens > 0).float()

    max_len_orig = int(torch.max(attention_mask.sum(dim=1)))
    max_len_aug = int(torch.max(attention_mask2.sum(dim=1)))

    max_len = max(max_len_orig, max_len_aug)

    return tokens[:, :max_len], embeds[:, :max_len, :], aug_tokens[:, :max_len], aug_embeds[:, :max_len, :]

def get_embed(args, model, tokens):
    if torch.cuda.device_count() > 1:
        if 'roberta' in args.backbone:
            if 'vanilla' in args.backbone:
                embed = model.module.backbone.embeddings.word_embeddings(tokens)
            else:
                embed = model.module.backbone.roberta.embeddings.word_embeddings(tokens)
        elif 'bert' in args.backbone:
            embed = model.module.backbone.bert.embeddings.word_embeddings(tokens)
        else:
            embed = model.module.backbone.embeddings.word_embeddings(tokens)
    else:
        if 'roberta' in args.backbone:
            if 'vanilla' in args.backbone:
                embed = model.backbone.embeddings.word_embeddings(tokens)
            else:
                embed = model.backbone.roberta.embeddings.word_embeddings(tokens)
        elif 'bert' in args.backbone:
            embed = model.backbone.bert.embeddings.word_embeddings(tokens)
        else:
            embed = model.backbone.embeddings.word_embeddings(tokens)

    return embed

def sym_kld(aug_logits, orig_logits):
    return (F.kl_div(F.log_softmax(aug_logits, dim=-1, dtype=torch.float32),
                       F.softmax(orig_logits, dim=-1, dtype=torch.float32),
                       None, None,"none",)\
           + F.kl_div(F.log_softmax(orig_logits, dim=-1, dtype=torch.float32),
               F.softmax(aug_logits, dim=-1, dtype=torch.float32),
               None, None, "none",)
           ).sum(dim=1)

##### Data augmentations #####

def generate_noise(embed, epsilon=1e-5):
    noise = embed.data.new(embed.size()).normal_(0, 1) * epsilon
    noise.detach()
    noise.requires_grad_()
    return noise

def norm_grad(grad, epsilon=1e-6, norm_p='l2'):
    if norm_p == 'l2':
        direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + epsilon)
    elif norm_p == 'l1':
        direction = grad.sign()
    else:
        direction = grad / (grad.abs().max(-1, keepdim=True)[0] + epsilon)
    return direction

def data_aug(args, model, tokens, embed, labels, bts_tokens, ctx_tokens, eda_tokens, policy):
    batch_size = tokens.size(0)

    if policy is None:
        tokens, attention_mask = cut_input(args, tokens)
        embed = get_embed(args, model, tokens)
        num_tokens = attention_mask.sum(dim=1).long()

    if args.dataset == 'stsb':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if policy is not None:
        aug_tokens, aug_embed = policy(args, tokens, embed, labels, bts_tokens, ctx_tokens, model)  # (B, L, V)
        tokens, embed, aug_tokens, aug_embed = cut_aug_input(args, tokens, embed, aug_tokens, aug_embed)
    elif 'cutoff' in args.train_type:
        '''
        From the official; https://github.com/dinghanshen/cutoff
        '''
        len_cutoff = args.cutoff
        embed_cutoff = []
        for i in range(batch_size):
            cutoff_length = int(num_tokens[i] * len_cutoff)
            if args.s_type == 'max' or args.s_type == 'min':
                start_idx = int(start_indices[i])
            else:
                start_idx = int(torch.rand(1) * (int(num_tokens[i]) - cutoff_length))
            cutoff_embed = torch.cat((embed[i][:start_idx],
                                      torch.zeros([cutoff_length, embed.shape[-1]], dtype=torch.float).to(embed),
                                      embed[i][start_idx + cutoff_length:]), dim=0)
            embed_cutoff.append(cutoff_embed)
        embed_cutoff = torch.stack(embed_cutoff, dim=0)

        aug_tokens, aug_embed = tokens, embed_cutoff.detach()
    elif 'r3f' in args.train_type:
        '''
        From the official; https://github.com/pytorch/fairseq/blob/master/examples/rxf/rxf_src/sentence_prediction_r3f.py
        '''
        if 'normal' in args.train_type:
            noise_sampler = torch.distributions.normal.Normal(loc=0.0, scale=args.eps)
        else:
            noise_sampler = torch.distributions.uniform.Uniform(low=-args.eps, high=args.eps)
        noise = noise_sampler.sample(sample_shape=embed.shape).to(embed)
        noise = noise * attention_mask.unsqueeze(2)  # to remove the noise on [PAD] ma
       
        aug_tokens, aug_embed = tokens, embed.detach().clone() + noise
    elif 'adv' in args.train_type:
        noise = generate_noise(embed)

        adv_logits = model(tokens, cls_token=None, inputs_embed=embed + noise, cls=True)
        adv_loss = criterion(adv_logits, labels)
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()
        if torch.isnan(norm) or torch.isinf(norm):
            return 0
        delta_grad = norm_grad(delta_grad)
        adv_embed = embed + delta_grad * args.step_size
        adv_embed = adv_embed.detach()

        aug_tokens, aug_embed = tokens, adv_embed
    elif 'backtrans' in args.train_type:
        aug_tokens = bts_tokens.clone().to(device)
        aug_tokens, _ = cut_input(args, aug_tokens)
        aug_embed = get_embed(args, model, aug_tokens)
    elif 'contextual' in args.train_type:
        aug_tokens = ctx_tokens.clone().to(device)
        aug_tokens, _ = cut_input(args, aug_tokens)
        aug_embed = get_embed(args, model, aug_tokens)
    elif 'coda' in args.train_type:
        bts_tokens = bts_tokens.clone().to(device)
        bts_tokens, _ = cut_input(args, bts_tokens)
        bts_embed = get_embed(args, model, bts_tokens)

        noise = generate_noise(bts_embed)

        adv_logits = model(bts_tokens, cls_token=None, inputs_embed=bts_embed + noise, cls=True)
        adv_loss = criterion(adv_logits, labels)
        delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
        norm = delta_grad.norm()
        if torch.isnan(norm) or torch.isinf(norm):
            return 0
        delta_grad = norm_grad(delta_grad)
        bts_embed = bts_embed + delta_grad * args.step_size
        adv_embed = bts_embed.detach()

        aug_tokens, aug_embed = bts_tokens, adv_embed
    elif 'eda' in args.train_type:
        aug_tokens = eda_tokens.clone().to(device)
        aug_tokens, _ = cut_input(args, aug_tokens)
        aug_embed = get_embed(args, model, aug_tokens)
    else:
        aug_tokens, aug_embed = tokens, embed
    return tokens, embed, aug_tokens, aug_embed