#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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
      
class LoraHypernet(nn.Module):
    
    def __init__(self,in_size=12,backbone_embed=768,rank=8) -> None:
        super(LoraHypernet, self).__init__()
        self.in_size = in_size
        self.backbone_embed = backbone_embed
        self.rank = rank
        
        self.linear = nn.Linear(in_size, self.backbone_embed * self.rank)
        
    def forward(self,z):
        
        weight = self.linear(z)
        
        weight = weight.view(self.backbone_embed,self.rank)
        
        return weight
        
    
        
        
# Copy from mmpretain/models/peft
class HyperLoRALinear(nn.Module):
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
                 hypernet,
                 lora_mask_token,
                 original_layer: nn.Linear,
                 alpha: int = 1,
                 rank: int = 0,
                 drop_rate: float = 0.):
        super(HyperLoRALinear, self).__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer
        self.hypernet = hypernet
        self.lora_mask_token = lora_mask_token
        
        self.lora_meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(in_features, 64)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(64,  self.hypernet.in_size))
        ]))

    def forward(self, x: torch.Tensor):
        
        out = self.original_layer(x)
        
        
        B, H,W, C = x.shape
        prompt_feat = self.lora_meta_net(x)
        mask =  torch.matmul(prompt_feat,self.lora_mask_token.weight.t())
        mask = mask.permute(3,0,1,2)
        num_mask = mask.shape[0]
        
        mask = mask.reshape(num_mask,B,-1)
        mask = mask.softmax(-1)
        mask = mask.reshape(num_mask,B,H,W)
        
        mask_wise_feat = torch.zeros_like(prompt_feat)
        for mask_id in range(num_mask):
            mask_wise_feat = mask_wise_feat + mask[mask_id].unsqueeze(-1) * prompt_feat
            
        mask_wise_feat = mask_wise_feat.sum(1).sum(1)
        
        lora_out_list = []
        for idx in range(B):
            
            down_weight = self.hypernet(mask_wise_feat[idx])

            lora_x = self.lora_dropout(x[idx])
            lora_down = F.linear(lora_x,down_weight.t())
            lora_out = self.lora_up(lora_down) * self.scaling
            
            lora_out_list.append(lora_out.unsqueeze(0))
        lora_out = torch.cat(lora_out_list,dim=0)

        return out + lora_out