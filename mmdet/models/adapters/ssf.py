
import torch
import torch.nn as nn
import torch.nn.functional as F


import math
from typing import Optional, List

class SSF_adapter(nn.Module):
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
                 original_module: nn.Linear):
        
        super(SSF_adapter, self).__init__()
        
        if original_module.__class__.__name__ == 'LayerNorm':
        
            dim = original_module.normalized_shape[0]
        
        if original_module.__class__.__name__ == 'Linear':
            
            dim = original_module.out_features
            
        if original_module.__class__.__name__ == 'Conv2d':
            
            dim = original_module.out_channels
            
        
        
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

        nn.init.normal_(self.scale, mean=1, std=.02)
        nn.init.normal_(self.shift, std=.02)
        
        self.original_module = original_module

    def forward(self, x: torch.Tensor):
        
        x_out = self.original_module(x)
        
        assert self.scale.shape == self.shift.shape
        if x_out.shape[-1] == self.scale.shape[0]:
            return x_out * self.scale + self.shift
        elif x_out.shape[1] == self.scale.shape[0]:
            return x_out * self.scale.view(1, -1, 1, 1) + self.shift.view(1, -1, 1, 1)
        else:
            raise ValueError('the input tensor shape does not match the shape of the scale factor.')