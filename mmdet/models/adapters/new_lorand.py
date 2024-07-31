
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from torch import Tensor
import torch.nn.functional as F
from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition
import torch.utils.checkpoint as cp

class LoRandv2(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, mid_features: int, bias: bool = True,
                 num_branch=2,
                 kernel_dim=2) -> None:
        super(LoRandv2, self).__init__()
        self.in_features = in_features
        self.mid_features = mid_features

        # self.A = Parameter(torch.Tensor(in_features, 2))
        # self.g = Parameter(torch.Tensor(2, 2))
        # self.B = Parameter(torch.Tensor(2, out_features))
        self.num_branch = num_branch
        self.param_dict={}
        count = 0
        while count < num_branch:
            # Down Projection
            DP_layer_name = 'DP'+str(count)
            DQ_layer_name = 'DQ'+str(count)
            self.param_dict[DP_layer_name] = Parameter(torch.Tensor(in_features, kernel_dim))
            self.param_dict[DQ_layer_name] = Parameter(torch.Tensor(kernel_dim, mid_features))
            nn.init.kaiming_uniform_(self.param_dict[DP_layer_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.param_dict[DQ_layer_name], a=math.sqrt(5))
            # Kernel
            K_layer_name = 'K'+str(count)
            self.param_dict[K_layer_name] = Parameter(torch.Tensor(kernel_dim, kernel_dim))
            nn.init.kaiming_uniform_(self.param_dict[K_layer_name], a=math.sqrt(5))
            # Up Projection
            UP_layer_name = 'UP' + str(count)
            UQ_layer_name = 'UQ' + str(count)
            self.param_dict[UP_layer_name] = Parameter(torch.Tensor(mid_features, kernel_dim))
            self.param_dict[UQ_layer_name] = Parameter(torch.Tensor(kernel_dim, in_features))
            nn.init.kaiming_uniform_(self.param_dict[UP_layer_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.param_dict[UQ_layer_name], a=math.sqrt(5))
            count += 1

        self.param_dict = nn.ParameterDict(self.param_dict)


        # self.weight = Parameter(torch.mm(torch.mm(self.A, self.g), self.B).t())


        if bias:
            self.bias_D = Parameter(torch.Tensor(mid_features))
            self.bias_U = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.bias_U is not None and self.bias_D is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            count = 0
            D_weight = 0
            U_weight = 0
            while count < self.num_branch:
                D_weight = D_weight + torch.mm(torch.mm(self.param_dict['DP'+str(count)], self.param_dict['K'+str(count)]),
                                           self.param_dict['DQ'+str(count)]).t()

                U_weight = U_weight + torch.mm(
                    torch.mm(self.param_dict['UP' + str(count)], self.param_dict['K' + str(count)]),
                    self.param_dict['UQ' + str(count)]).t()
                count += 1
            # weight = weight/self.num_branch        # avg_weight

            fan_in_D, _ = nn.init._calculate_fan_in_and_fan_out(D_weight)
            bound_D = 1 / math.sqrt(fan_in_D)
            nn.init.uniform_(self.bias_D, -bound_D, bound_D)

            fan_in_U, _ = nn.init._calculate_fan_in_and_fan_out(U_weight)
            bound_U = 1 / math.sqrt(fan_in_U)
            nn.init.uniform_(self.bias_U, -bound_U, bound_U)


    def forward(self, input: Tensor) -> Tensor:

        count = 0
        weight_D = 0
        weight_U = 0
        while count < self.num_branch:
            weight_D = weight_D + torch.mm(torch.mm(self.param_dict['DP' + str(count)], self.param_dict['K' + str(count)]),
                                       self.param_dict['DQ' + str(count)]).t()
            weight_U = weight_U + torch.mm(torch.mm(self.param_dict['UP' + str(count)], self.param_dict['K' + str(count)]),
                                       self.param_dict['UQ' + str(count)])
            count += 1
        # weight = weight/self.num_branch        # avg_weight
        return input + F.linear(F.gelu(F.linear(input, weight_D, self.bias_D)), weight_U.t(), self.bias_U)
 

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )



def forward_sam_block_lorand(self, x):
    
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
        
        
    # lorand 1 
    x = self.adapter_lorand1(x)

    x = shortcut + x
    
    identity = x
    
    x = self.norm2(x)
    
    # lorand2
    x = self.adapter_lorand2(x)
    
    x = self.mlp(x)
    
    x = identity + x
    
    return x


def forward_dinov2_layer_lorand(self, x,patch_resolution):
    
    shortcut = x 
    
    x= self.attn(self.ln1(x))
    
    x = self.adapter_lorand1(x)
    
    x = shortcut + x
    
    identity = x
    
    x = self.ln2(x)
    
    x = self.adapter_lorand2(x)
    
    x = self.ffn(x,identity=identity)
    
    return x
    
    
def forward_eva02_layer_lorand(self, x,patch_resolution):
    
    inputs = x
    x = self.norm1(x)
    x = self.attn(x, patch_resolution)
    
    x = self.adapter_lorand1(x)
    
    x = self.drop_path(x)
    x = inputs + x



    inputs = x
    x = self.norm2(x)
    
    x = self.adapter_lorand2(x)
    
    x = self.mlp(x)
    x = self.drop_path(x)
    x = inputs + x
    
    return x
    
    
# 用于替换SwinBlock的forward
def forward_swin_block_lorand(self, x, hw_shape):

    def _inner_forward(x):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = self.adapter_lorand1(x)

        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.adapter_lorand2(x)
        x = self.ffn(x, identity=identity)
        

        return x

    if self.with_cp and x.requires_grad:
        x = cp.checkpoint(_inner_forward, x)
    else:
        x = _inner_forward(x)

    return x


def forward_swin_block_lorandv2(self, x, hw_shape):

    def _inner_forward(x):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw_shape)
        x = self.adapter_lorand1(x)

        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        x = self.adapter_lorand2(x)
        

        return x

    if self.with_cp and x.requires_grad:
        x = cp.checkpoint(_inner_forward, x)
    else:
        x = _inner_forward(x)

    return x





