# Code is mainly adopted from https://github.com/moskomule/dda/tree/fasteraa/faster_autoaugment
""" `functional` contains deterministic functions
img image tensor `img` is expected to be CxHxW or BxCxHxW and its range should be [0, 1]
`mag=0` expects no transformation
"""

import functools
from typing import Optional

import numpy as np
from torch import nn
from torch.nn import functional as F
import torch.distributions.categorical as categorical

# Note: if magnitude should be updated => ste(output, mag). If not, ste(output, input)

__all__ = ['cutoff', 'adversarial', 'cbert', 'backtrans', 'r3f', 'eda']

# helper functions

from typing import Tuple

import torch
from torch.autograd import Function

class _STE(Function):
    """ StraightThrough Estimator """
    @staticmethod
    def forward(ctx,
                input_forward: torch.Tensor,
                input_backward: torch.Tensor) -> torch.Tensor:
        ctx.shape = input_backward.shape
        return input_forward

    @staticmethod
    def backward(ctx,
                 grad_in: torch.Tensor) -> Tuple[None, torch.Tensor]:
        return None, grad_in.sum_to_size(ctx.shape)


def ste(input_forward: torch.Tensor,
        input_backward: torch.Tensor) -> torch.Tensor:
    """
    Straight-through estimator
    :param input_forward:
    :param input_backward:
    :return:
    """

    return _STE.apply(input_forward, input_backward).clone()

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

def tensor_function(func):
    # check if the input is correctly given
    @functools.wraps(func)
    def inner(*args):
        if len(args) == 1:
            img = args[0]
            mag = None
        elif len(args) == 2:
            img, mag = args
        else:
            arg, img, embed, labels, bts, ctx, eda, mag, model = args

        if not torch.is_tensor(img):
            raise RuntimeError(f'img is expected to be torch.Tensor, but got {type(img)} instead')

        if torch.is_tensor(mag) and mag.nelement() != 1 and mag.size(0) != img.size(0):
            raise RuntimeError('Shape of `mag` is expected to be `1` or `B`')

        out1, out2 = func(arg, img, embed, labels, bts, ctx, eda, mag, model) if len(args) == 9 else func(img, mag)
        return out1, out2

    return inner


# NLP transformation functions
@tensor_function
def r3f(args,
        tokens: torch.Tensor,
        embed: torch.Tensor,
        labels: torch.Tensor,
        bts: torch.Tensor,
        ctx: torch.Tensor,
        eda: torch.Tensor,
        mag: torch.Tensor,
        model: nn.Module) -> torch.Tensor:
    eps = mag.view(-1, 1, 1).cuda()
    attention_mask = (tokens != 1).float().unsqueeze(2).cuda()

    noise_sampler = torch.distributions.uniform.Uniform(low=-1, high=1)
    noise = noise_sampler.sample(sample_shape=embed.shape).to(embed)
    noise = eps * noise * attention_mask  # to remove the noise on [PAD] mask

    #return tokens, ste(embed + noise, mag.cuda())
    return tokens, embed + noise

@tensor_function
def backtrans(args,
              tokens: torch.Tensor,
              embed: torch.Tensor,
              labels: torch.Tensor,
              bts: torch.Tensor,
              ctx: torch.Tensor,
              eda: torch.Tensor,
              mag: torch.Tensor,
              model: nn.Module) -> torch.Tensor:

    if torch.cuda.device_count() > 1:
        if args.backbone == 'bert':
            Embedding = model.module.backbone.bert.embeddings.word_embeddings
        else:
            Embedding = model.module.backbone.roberta.embeddings.word_embeddings
    else:
        if args.backbone == 'bert':
            Embedding = model.backbone.bert.embeddings.word_embeddings
        else:
            Embedding = model.backbone.roberta.embeddings.word_embeddings
    en2de_batch = bts.cuda()

    return en2de_batch, ste(Embedding(en2de_batch), embed)

@tensor_function
def cutoff(args,
           tokens: torch.Tensor,
           embed: torch.Tensor,
           labels: torch.Tensor,
           bts: torch.Tensor,
           ctx: torch.Tensor,
           eda: torch.Tensor,
           mag: torch.Tensor,
           model: nn.Module) -> torch.Tensor:

    batch_size = tokens.size(0)
    mag = mag.view(-1, 1, 1)
    attention_mask = (tokens != 1).float()
    num_tokens = attention_mask.sum(dim=1, keepdim=True).cpu()

    # change of tensor is verified
    embed_cutoff = []
    for i in range(batch_size):
        cutoff_size = mag[0].data
        cutoff_length = int(num_tokens[i] * float(cutoff_size))
        start_idx = int(torch.rand(1) * (int(num_tokens[i]) - cutoff_length))
        cutoff_embed = torch.cat((embed[i][:start_idx],
                                  torch.zeros([cutoff_length, embed.shape[-1]], dtype=torch.float).to(embed),
                                  embed[i][start_idx + cutoff_length:]), dim=0)
        embed_cutoff.append(cutoff_embed)
    embed_cutoff = torch.stack(embed_cutoff, dim=0)

    return tokens, ste(embed_cutoff, mag.cuda())

@tensor_function
def adversarial(args,
                tokens: torch.Tensor,
                embed: torch.Tensor,
                labels: torch.Tensor,
                bts: torch.Tensor,
                ctx: torch.Tensor,
                eda: torch.Tensor,
                mag: torch.Tensor,
                model: nn.Module) -> torch.Tensor:

    if args.dataset == 'stsb':
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    mag = mag.view(-1, 1, 1).cuda()
    step_size = mag
    noise = generate_noise(embed)

    adv_logits = model(tokens, inputs_embed=embed + noise)
    adv_loss = criterion(adv_logits, labels)
    delta_grad, = torch.autograd.grad(adv_loss, noise, only_inputs=True)
    norm = delta_grad.norm()
    if torch.isnan(norm) or torch.isinf(norm):
        return 0
    delta_grad = norm_grad(delta_grad)
    aug_embed = embed + delta_grad * step_size

    return tokens, aug_embed

@tensor_function
def cbert(args,
          tokens: torch.Tensor,
          embed: torch.Tensor,
          labels: torch.Tensor,
          bts: torch.Tensor,
          ctx: torch.Tensor,
          eda: torch.Tensor,
          mag: torch.Tensor,
          model: nn.Module) -> torch.Tensor:
    if torch.cuda.device_count() > 1:
        if args.backbone == 'bert':
            Embedding = model.module.backbone.bert.embeddings.word_embeddings
        else:
            Embedding = model.module.backbone.roberta.embeddings.word_embeddings
    else:
        if args.backbone == 'bert':
            Embedding = model.backbone.bert.embeddings.word_embeddings
        else:
            Embedding = model.backbone.roberta.embeddings.word_embeddings
    en2de_batch = ctx.cuda()

    return en2de_batch, ste(Embedding(en2de_batch), embed)

@tensor_function
def eda(args,
        tokens: torch.Tensor,
        embed: torch.Tensor,
        labels: torch.Tensor,
        bts: torch.Tensor,
        ctx: torch.Tensor,
        eda: torch.Tensor,
        mag: torch.Tensor,
        model: nn.Module) -> torch.Tensor:
    if torch.cuda.device_count() > 1:
        if args.backbone == 'bert':
            Embedding = model.module.backbone.bert.embeddings.word_embeddings
        else:
            Embedding = model.module.backbone.roberta.embeddings.word_embeddings
    else:
        if args.backbone == 'bert':
            Embedding = model.backbone.bert.embeddings.word_embeddings
        else:
            Embedding = model.backbone.roberta.embeddings.word_embeddings
    en2de_batch = eda.cuda()

    return en2de_batch, ste(Embedding(en2de_batch), embed)