import math
import torch
import torch.nn as nn
from mmdet.registry import MODELS

@MODELS.register_module()
class AdapterFormer(nn.Module):
    def __init__(self,
                 original_layer,
                 embed_dims,
                 down_size = 8,
                 dropout=0.0,
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in",
                 add_residual = True
                 ):
        super().__init__()
        # self.n_embd = config.d_model if d_model is None else d_model
        # self.down_size = config.attn_bn if bottleneck is None else bottleneck
        self.down_size = down_size
        self.embed_dims = embed_dims
        self.original_layer = original_layer
        self.add_residual = add_residual
        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.embed_dims)

        if adapter_scalar == "learnable_scalar":
            self.adapter_scale = nn.Parameter(torch.ones(1))
        else:
            self.adapter_scale = float(adapter_scalar)

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



    def forward(self, x,identity):
        
        residual = self.original_layer(x)
        
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.adapter_down_proj(x)
        down = self.non_linear_func(down)
        down =  self.dropout(down)
        up = self.adapter_up_proj(down)

        up = up * self.adapter_scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if self.add_residual:
            output = up + residual
        else:
            output = up
        if identity is not None:
            # 这里的identity是只经过了attention模块之后的,正常来说是在self.original_layer的ffn中加的一个残差，但这里我们在最后加是一样的
            '''
                在dinov2和clip中identity不为none
                在sam中identity为none,因为sam中在每个block的forward中加了过了
            '''
            output = output + identity
        
        return output