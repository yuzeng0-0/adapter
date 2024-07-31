
from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition
from mmengine.model.base_module import ModuleList
from torch import nn
import torch

class DWConv(nn.Module):
    def __init__(self,dim):
 
        #这一行千万不要忘记
        super(DWConv, self).__init__()
 
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=dim,
                                    out_channels=dim,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=dim)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
 
        #逐点卷积
        self.point_conv = nn.Conv2d(in_channels=dim,
                                    out_channels=dim,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)
    
    def forward(self,input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
    
def forward_sam_block_t1(self, x):
    '''
        conv(ffn+attn) + lora(ffn)
    '''
    
    shortcut = x + self.adapter_attn(self.norm1(x)) * self.s
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(self.norm2(x)) * self.s + self.loralinear(x) 
    
    return x

def forward_sam_block_t2(self, x):
    '''
        conv(attn) + lora(ffn)
    '''
    
    shortcut = x + self.adapter_attn(self.norm1(x)) * self.s
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
    x = x + self.mlp(self.norm2(x)) + self.loralinear(x) 
    
    return x

def forward_sam_block_t3(self, x):
    '''
        conv(ffn) + lora(ffn)
    '''
    
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(self.norm2(x)) * self.s + self.loralinear(x) 
    
    return x


def forward_sam_block_t4(self, x):
    
    '''
        conv(all) + lora(ffn)
    '''
    
    conv_shortcut = self.adapter_all(self.norm1(x)) * self.s
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
    x = x + self.mlp(self.norm2(x))  + conv_shortcut + self.loralinear(x) 
    
    return x

def forward_sam_block_t5(self, x):
    
    '''
        conv(attn+mlp) + lora(all)
    '''
    lora_shortcut = self.loralinear(x) 
    shortcut = x + self.adapter_attn(self.norm1(x)) * self.s
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
    x = x + self.mlp(self.norm2(x))  +  self.adapter_mlp(self.norm2(x)) * self.s + lora_shortcut
    
    return x


def forward_sam_block_t6(self, x):
    
    '''
        conv(attn+mlp) + lora(attn)
    '''
    shortcut = x + self.adapter_attn(self.norm1(x)) * self.s + self.loralinear(x) 
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
    x = x + self.mlp(self.norm2(x))  +  self.adapter_mlp(self.norm2(x)) * self.s
    
    return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class Convpass_swin(nn.Module):
    def __init__(self, input_dim,dw_conv=False,dim=8, xavier_init=True,linear_init='zero'):
        super().__init__()
        self.dw_conv = dw_conv
        if dw_conv:
            self.adapter_conv = DWConv(dim)
        else:
            self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
            if xavier_init:
                nn.init.xavier_uniform_(self.adapter_conv.weight)
            else:
                nn.init.zeros_(self.adapter_conv.weight)
                self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
            nn.init.zeros_(self.adapter_conv.bias)
            
        
        # self.adapter_conv = ModuleList()  
        # for i in range(num_conv):
        #     this_adapter = nn.Conv2d(dim, dim, 3, 1, 1)            
        #     if xavier_init:
        #         nn.init.xavier_uniform_(this_adapter.weight)
        #     else:
        #         nn.init.zeros_(this_adapter.weight)
        #         this_adapter.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
        #     nn.init.zeros_(this_adapter.bias)
        #     self.adapter_conv.append(this_adapter)
            
            
        assert linear_init=='zero' or linear_init=='xavier'
        
        self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
        
        
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
        # if self.dw_conv:
        #     x_patch = x_patch.reshape(B,self.dim,-1)
        #     x_patch = x_patch.permute(0,2,1)
        #     x_patch = self.adapter_conv(x_patch,H,W)
        # else:
        x_patch = self.adapter_conv(x_patch)
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    