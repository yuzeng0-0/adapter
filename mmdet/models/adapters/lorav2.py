#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition

def forward_dinov2_block_type1(self, x):
    x = x + self.attn(self.ln1(x))
    x = self.ffn(self.ln2(x), identity=x) + self.loralinear(x) 
    return x

def forward_eva02_block_type1(self, x,patch_resolution):

    x = x + self.drop_path(self.attn(self.norm1(x),patch_resolution))
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.loralinear(self.norm2(x))) 
    return x

def forward_sam_block_type1(self, x):
    
    shortcut = x 
    x = self.norm1(x)
    
    
    # Window partition
    if self.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)

    x = self.attn(x)
    # Reverse window partition
    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))

    x = shortcut + x
    x = x + self.mlp(self.norm2(x)) +  self.loralinear(x) 
    
    return x
        
        
# Copy from mmpretain/models/peft
class LoRALinearv2(nn.Module):
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
                 embed_dims,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.):
        super(LoRALinearv2, self).__init__()
        # in_features = original_layer.in_features
        # out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(embed_dims, rank, bias=False)
        self.lora_up = nn.Linear(rank, embed_dims, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # self.original_layer = original_layer

    def forward(self, x: torch.Tensor):

        lora_x = self.lora_dropout(x)
        lora_out = self.lora_up(self.lora_down(lora_x)) * self.scaling

        return lora_out