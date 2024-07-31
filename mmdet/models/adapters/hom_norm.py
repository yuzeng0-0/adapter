
import torch
from torch import nn
import timm
import math

from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition
from mmengine.model.base_module import ModuleList

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x



class Hom_norm(nn.Module):
    def __init__(self, input_dim,dim=8, xavier_init=True,linear_init='zero'):
        super().__init__()
        # self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        # if xavier_init:
        #     nn.init.xavier_uniform_(self.adapter_conv.weight)
        # else:
        #     nn.init.zeros_(self.adapter_conv.weight)
        #     self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
        # nn.init.zeros_(self.adapter_conv.bias)
            
        assert linear_init=='zero' or linear_init=='xavier'
        
        self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
        
        self.layer_norm = LayerNorm2d(dim)

        if linear_init=='zero':
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.zeros_(self.adapter_up.weight)
        else:
            nn.init.xavier_uniform_(self.adapter_down.weight)
            nn.init.xavier_uniform_(self.adapter_up.weight)
            
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
            
        # x_patch = self.adapter_conv(x_patch)
        x_patch = self.layer_norm(x_patch)

        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up