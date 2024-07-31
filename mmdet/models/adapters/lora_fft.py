#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List
from mmcv.ops import DeformConv2dPack
from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition
        
        
# Copy from mmpretain/models/peft
class LoRAFFT(nn.Module):
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
                 dcn_stage=[False,False,False],
                 ):
        super(LoRAFFT, self).__init__()
        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer
        
        conv_types = []
        for dcn in dcn_stage:
            conv_type = DeformConv2dPack if dcn else nn.Conv2d
            conv_types.append(conv_type)
            
        fft_lora = False
        self.main = conv_types[0](in_features, out_features, kernel_size=3, padding=1) if not fft_lora else None
        self.mag = conv_types[1](rank, rank, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            conv_types[2](rank, rank, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            conv_types[2](rank, rank, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor):

        
        out = self.original_layer(x)

        lora_x = self.lora_dropout(x)
        lora_down = self.lora_down(lora_x)
        
        # if x.dim()==3:
        #     B,_,C = lora_down.shape
        #     cls_token = lora_down[:,0,:]
        #     lora_down = lora_down[:,1:,:]
        #     lora_down = lora_down.view(B,self.patch_resolution[0],self.patch_resolution[1],C)
        
        
        lora_down = lora_down.permute(0,3,1,2)
        
        fft_feat = self.fft_forward(lora_down)
        fft_feat = fft_feat.permute(0,2,3,1)
        
        # if x.dim()==3:
        #     fft_feat = fft_feat.view(B,-1,C)
        #     fft_feat = torch.cat([cls_token,fft_feat],dim=1)
        
        
        
        lora_out = self.lora_up(fft_feat) * self.scaling
        

        return out + lora_out
    
    
    def fft_forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.mag(mag)
        mag_res = mag_out - mag
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')
        return y
    
    
# 用于替换整个ffn  or attn 
class LoRAFFTv2(nn.Module):
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
                 embed,
                 rank: int = 0,
                 drop_rate: float = 0.,
                 dcn_stage=[False,False,False],
                 ):
        super(LoRAFFTv2, self).__init__()
        # in_features = original_layer.in_features
        # out_features = original_layer.out_features

        self.lora_dropout = nn.Dropout(drop_rate)
        self.lora_down = nn.Linear(embed, rank, bias=False)
        self.lora_up = nn.Linear(rank, embed, bias=False)
        # self.scaling = alpha / rank

        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

        # self.original_layer = original_layer
        
        
        conv_types = []
        for dcn in dcn_stage:
            conv_type = DeformConv2dPack if dcn else nn.Conv2d
            conv_types.append(conv_type)
            
        fft_lora = False
        self.main = conv_types[0](embed, embed, kernel_size=3, padding=1) if not fft_lora else None
        self.mag = conv_types[1](rank, rank, kernel_size=1, padding=0)
        self.pha = nn.Sequential(
            conv_types[2](rank, rank, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            conv_types[2](rank, rank, kernel_size=1, padding=0),
        )

    def forward(self, x: torch.Tensor):
        # out = self.original_layer(x)

        lora_x = self.lora_dropout(x)
        lora_down = self.lora_down(lora_x)
        lora_down = lora_down.permute(0,3,1,2)
        
        fft_feat = self.fft_forward(lora_down)
        fft_feat = fft_feat.permute(0,2,3,1)
        
        lora_out = self.lora_up(fft_feat) 
        

        return lora_out
    
    
    def fft_forward(self, x):
        _, _, H, W = x.shape
        fre = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(fre)
        pha = torch.angle(fre)
        mag_out = self.mag(mag)
        mag_res = mag_out - mag
        pooling = torch.nn.functional.adaptive_avg_pool2d(mag_res, (1, 1))
        pooling = torch.nn.functional.softmax(pooling, dim=1)
        pha1 = pha * pooling
        pha1 = self.pha(pha1)
        pha_out = pha1 + pha
        real = mag_out * torch.cos(pha_out)
        imag = mag_out * torch.sin(pha_out)
        fre_out = torch.complex(real, imag)
        y = torch.fft.irfft2(fre_out, s=(H, W), norm='backward')
        return y
    
    
    
def forward_sam_block_lorafft_mlp(self, x):
    
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(self.norm2(x)) * self.s
    
    return x