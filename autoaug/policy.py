# Code is mainly adopted from https://github.com/moskomule/dda/tree/fasteraa/faster_autoaugment
from __future__ import annotations

import random
from copy import deepcopy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Categorical

from autoaug.operations import *

class SubPolicyStage(nn.Module):
    def __init__(self,
                 operations: nn.ModuleList,
                 temperature: float,
                 ):
        super(SubPolicyStage, self).__init__()
        self.operations = operations
        self._weights = nn.Parameter(torch.ones(len(self.operations)))
        self.temperature = temperature

    def forward(self,
                args,
                input: Tensor,
                embed: Tensor,
                labels: Tensor,
                bts: Tensor,
                ctx: Tensor,
                eda: Tensor,
                model: nn.Module
                ) -> Tensor:
        if self.training:
            sampled_op = F.gumbel_softmax(self._weights, tau=self.temperature, hard=True).cuda()
            if self._weights.data.mean() == 1:
                sampled_idx = int(torch.randint(0, len(self.operations), (1,)))
            else:
                sampled_idx = torch.max(sampled_op, dim=0)[1]

            inputs, embeds = self.operations[sampled_idx](args, input, embed, labels, bts, ctx, eda, model)
            return inputs, (embeds.unsqueeze(0) * sampled_op.view(-1, 1, 1, 1)).sum(0)
        else:
            return self.operations[Categorical(self.weights).sample()](args, input, embed, labels, bts, ctx, eda, model)

    @property
    def weights(self
                ):
        return self._weights.div(self.temperature).softmax(0).cuda()


class SubPolicy(nn.Module):
    def __init__(self,
                 sub_policy_stage: SubPolicyStage,
                 sub_policy_stage2: SubPolicyStage,
                 operation_count: int,
                 ):
        super(SubPolicy, self).__init__()
        self.stages = nn.ModuleList([deepcopy(sub_policy_stage) if o == 0 else deepcopy(sub_policy_stage2)
                                     for o in range(operation_count)])

    def forward(self,
                args,
                input: Tensor,
                embed: Tensor,
                labels: Tensor,
                bts: Tensor,
                ctx: Tensor,
                eda: Tensor,
                model: nn.Module
                ) -> Tensor:
        for stage in self.stages:
            input, embed = stage(args, input, embed, labels, bts, ctx, eda, model)
        return input, embed


class Policy_gumbel(nn.Module):
    def __init__(self,
                 operations: nn.ModuleList,
                 operations2: nn.ModuleList,
                 num_sub_policies: int,
                 temperature: float = 0.05,
                 operation_count: int = 2,
                 num_chunks: int = 4,
                 ):
        super(Policy_gumbel, self).__init__()
        self.sub_policies = nn.ModuleList([SubPolicy(SubPolicyStage(operations, temperature),
                                                     SubPolicyStage(operations2, temperature), operation_count)
                                           for _ in range(num_sub_policies)])
        self.num_sub_policies = num_sub_policies
        self.temperature = temperature
        self.operation_count = operation_count
        self.num_chunks = num_chunks

    def forward(self,
                args,
                input: Tensor,
                embed: Tensor,
                labels: Tensor,
                bts: Tensor,
                ctx: Tensor,
                eda: Tensor,
                model: nn.Module
                ) -> Tensor:
        if self.num_chunks > 1:
            input_chunk = input.chunk(self.num_chunks)
            embed_chunk = embed.chunk(self.num_chunks)
            labels_chunk = labels.chunk(self.num_chunks)
            indices_chunk = indices.chunk(self.num_chunks)

            x_outs = []
            y_outs = []
            for n in range(self.num_chunks):
                x_out, y_out = self._forward(args, input_chunk[n], embed_chunk[n], labels_chunk[n], indices_chunk[n], model)
                x_outs.append(x_out)
                y_outs.append(y_out)
            x = torch.cat(x_outs, dim=0)
            y = torch.cat(y_outs, dim=0)
        else:
            x, y = self._forward(args, input, embed, labels, bts, ctx, eda, model)
        return x, y

    def _forward(self,
                 args,
                 input: Tensor,
                 embed: Tensor,
                 labels: Tensor,
                 bts: Tensor,
                 ctx: Tensor,
                 eda: Tensor,
                 model: nn.Module
                 ) -> Tensor:
        index = random.randrange(self.num_sub_policies)
        return self.sub_policies[index](args, input, embed, labels, bts, ctx, eda, model)

    @staticmethod
    def nlp_operations():
        return [
            Cutoff(),
            Adversarial(),
            Cbert(),
            BackTrans(),
            R3F(),
            EDA(),
        ]

    @staticmethod
    def nlp_operations2():
        return [
            Cutoff(),
            Adversarial(),
            R3F(),
        ]

    @staticmethod
    def faster_auto_augment_policy(num_sub_policies: int,
                                   temperature: float,
                                   operation_count: int,
                                   num_chunks: int,
                                   ) -> Policy:
        return Policy_gumbel(nn.ModuleList(Policy_gumbel.nlp_operations()), nn.ModuleList(Policy_gumbel.nlp_operations2()),
                      num_sub_policies, temperature, operation_count, num_chunks)
