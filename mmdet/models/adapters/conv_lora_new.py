#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class Expert(nn.Module):
    def __init__(self,
                 dim, # Lora 中间维度
                 interpolate_rate, # 插值的比例
                 patch_resolution  # (H, W)
                 ):
        super(Expert, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # 卷积初始化
        nn.init.zeros_(self.conv.weight)
        self.conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)

        self.interpolate_rate = interpolate_rate
        self.patch_resolution = patch_resolution

    def forward(self, x):

        B, C, H, W = x.shape
        x = F.interpolate(
            x,
            size=(H * self.interpolate_rate, W * self.interpolate_rate),
            mode='bicubic',
            align_corners=False)
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=(H, W),
            mode='bicubic',
            align_corners=False)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

        return x


# Copy from mmpretain/models/peft
class LoRALinear(nn.Module):
    r"""Implements LoRA in a linear layer.

    Args:
        original_layer (nn.Linear): The linear layer to be finetuned.
        alpha (int): The scale factor of LoRA. Defaults to 1.
        rank (int): The rank of LoRA. Defaults to 0.
        drop_rate (float): The drop out rate for LoRA. Defaults to 0.

    Note:
        The forward process of LoRA linear layer is:

        .. math::
            `y = W_0 x + BAx * (\alpha / r)`

        Where :math:`x` is the input, :math:`y` is the output,
        :math:`W_0` is the parameter of the original layer,
        :math:`A` and :math:`B` are the low-rank decomposition matrixs,
        :math: `\alpha` is the scale factor and :math: `r` is the rank.
    """

    def __init__(self,
                 original_layer: nn.Linear,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.,
                 # patch_resolution这里有bug
                 patch_resolution: tuple = (84, 84)):
        super(LoRALinear, self).__init__()
        self.patch_resolution = patch_resolution
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)

        # 在这个位置加三个 expert 和一个 MoE
        expert_num = 3
        # 论文中的全局平均池化，作为MoE gate的输入
        self.AvgPool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Linear(rank, expert_num)
        nn.init.uniform_(self.gate.weight)
        self.expert_0 = Expert(dim=rank, interpolate_rate=2, patch_resolution=patch_resolution)
        self.expert_1 = Expert(dim=rank, interpolate_rate=1, patch_resolution=patch_resolution)
        self.expert_2 = Expert(dim=rank, interpolate_rate=4, patch_resolution=patch_resolution)
        self.expert = [
            self.expert_0,
            self.expert_1,
            self.expert_2
        ]

        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer

    def forward(self, x: torch.Tensor):
        # x shape为(B, N, C)
        out = self.original_layer(x)

        x = self.lora_dropout(x)

        x = self.lora_down(x)

        B, N, C = x.shape
        patch_w, patch_h = self.patch_resolution[0], self.patch_resolution[1]
        cls_embed = x[:, :1]
        x = x[:, 1:].reshape(B, patch_w, patch_h, C).permute(0, 3, 1, 2)
        avgpool = self.AvgPool(x)
        avgpool = avgpool.squeeze(-1).squeeze(-1)

        gate_score = self.gate(avgpool)
        out_list = []
        for i in range(B):
            max_index = torch.argmax(gate_score[i]).tolist()

            out_list.append(self.expert[max_index](x[i].unsqueeze(0)) * self.scaling)
        
        lora_out = torch.cat(out_list, dim=0)
        lora_out = torch.cat([cls_embed, lora_out], dim=1)

        lora_out = self.lora_up(lora_out)

        return out + lora_out