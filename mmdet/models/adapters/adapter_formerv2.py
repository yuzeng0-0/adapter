import math
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition


def forward_dinov2_block(self, x,patch_resolution):
    # for dino v2
    x = x + self.attn(self.ln1(x))
    x = self.ffn(self.ln2(x), identity=x) + self.adapter_mlp(x)

    return x

def forward_eva02_block(self, x,patch_resolution):
    # for dino v2
    x = x + self.drop_path(self.attn(self.norm1(x),patch_resolution))
    
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x))) 
    return x

def forward_sam_block(self, x):
    
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(x) 
    
    return x


@MODELS.register_module()
class AdapterFormerv2(nn.Module):
    def __init__(self,
                 embed_dims,
                 down_size = 8,
                 dropout=0.0,
                 adapter_scale="1.0",
                 adapter_layernorm_option="in",
                 add_residual = True
                 ):
        super().__init__()
        # self.n_embd = config.d_model if d_model is None else d_model
        # self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.down_size = down_size
        self.embed_dims = embed_dims
        # self.original_layer = original_layer
        self.add_residual = add_residual
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.embed_dims)

        if adapter_scale == "learnable_scalar":
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = float(adapter_scale)

        self.adapter_down_proj = nn.Linear(self.embed_dims, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.adapter_up_proj = nn.Linear(self.down_size, self.embed_dims)

        self.dropout = nn.Dropout(dropout)
        # if init_option == "bert":
        #     raise NotImplementedError
        # elif init_option == "lora":
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.adapter_down_proj.weight, a=math.sqrt(5))
            nn.init.zeros_(self.adapter_up_proj.weight)
            nn.init.zeros_(self.adapter_down_proj.bias)
            nn.init.zeros_(self.adapter_up_proj.bias)



    def forward(self, x):
        

        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.adapter_down_proj(x)
        down = self.non_linear_func(down)
        down =  self.dropout(down)
        up = self.adapter_up_proj(down)
        # TODO 在外面也乘了scale,需要去掉一个 scale是1的时候无所谓，其他值时候就存在问题了,将外面的去掉
        up = up * self.adapter_scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)
        
        return up