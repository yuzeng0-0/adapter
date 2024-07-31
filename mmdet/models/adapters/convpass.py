
import torch
from torch import nn
import timm
import math

from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition
from mmengine.model.base_module import ModuleList



def forward_dinov2_block_vpt(self, x,patch_resolution):
    # for dino v2

    x = x + self.attn(self.ln1(x)) + self.adapter_attn(self.ln1(x),patch_resolution) * self.s
    x = self.ffn(self.ln2(x), identity=x) + self.adapter_mlp(self.ln2(x),patch_resolution) * self.s
    
    return x
        
def forward_eva02_block_vpt(self, x,patch_resolution):
    # for dino v2
    x = x + self.drop_path(self.attn(self.norm1(x),patch_resolution)) + self.drop_path(self.adapter_attn(self.norm1(x),patch_resolution)) * self.s
    
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x),patch_resolution)) * self.s
    return x


def forward_sam_block_vpt(self, x,patch_resolution):
    
        shortcut = x + self.adapter_attn(self.norm1(x),patch_resolution) * self.s
        x = self.norm1(x)
        
        
        # befor window partition,change the input size
        H, W = patch_resolution
        B, L, C = x.shape
        L = L - self.num_vpt_token
        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)
        
        # x = x.view(B, H, W, C)
        
        # Window partition
        if self.window_size > 0:
            prompt_emb = x[:, :self.num_vpt_token, :]
            x = x[:, self.num_vpt_token:, :]
            x = x.view(B, H, W, C)
            
            
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            # num_window*B,window_size*window_size,c
            x = x.view(-1, self.window_size * self.window_size, C)
            
            # expand prompt_emb
            num_windows = int(x.shape[0] / B)
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_vpt_token, C))
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.attn(x,patch_resolution,self.window_size)
        
        # Reverse window partition
        if self.window_size > 0:
            
            
            prompt_emb = x[:, :self.num_vpt_token, :]
            x = x[:, self.num_vpt_token:, :]
            prompt_emb = prompt_emb.view(-1, B, self.num_vpt_token, C)
            prompt_emb = prompt_emb.mean(0)
            
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            x = x.view(B,-1,C)
            x = torch.cat((prompt_emb, x), dim=1)
            
            

        x = shortcut + x
 
        x = x + self.mlp(self.norm2(x)) + self.adapter_mlp(self.norm2(x),patch_resolution) * self.s

        return x


def forward_dinov2_block_t1(self, x,patch_resolution):
    '''
        t1 means attn + mlp
    '''
    # for dino v2
    x = x + self.attn(self.ln1(x)) + self.adapter_attn(self.ln1(x),patch_resolution) * self.s
    x = self.ffn(self.ln2(x), identity=x) + self.adapter_mlp(self.ln2(x),patch_resolution) * self.s
    
    return x

def forward_eva02_block_t1(self, x,patch_resolution):
    # for dino v2
    x = x + self.drop_path(self.attn(self.norm1(x),patch_resolution)) + self.drop_path(self.adapter_attn(self.norm1(x),patch_resolution)) * self.s
    
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x),patch_resolution)) * self.s
    return x

def forward_sam_block_t1(self, x):
    
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(self.norm2(x)) * self.s
    
    return x

def forward_dinov2_block_t2(self, x,patch_resolution):
    '''
        t2 means only adapter mlp
    '''
    x = x + self.attn(self.ln1(x))
    # print( "mlp adapter:",self.adapter_mlp(self.ln2(x),patch_resolution) * self.s)
    x = self.ffn(self.ln2(x), identity=x) + self.adapter_mlp(self.ln2(x),patch_resolution) * self.s
    
    return x

def forward_eva02_block_t2(self, x,patch_resolution):
    '''
        t2 means only adapter mlp
    '''
    x = x + self.drop_path(self.attn(self.norm1(x),patch_resolution))
    
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x),patch_resolution)) * self.s
    return x

def forward_sam_block_t2(self, x):
    
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

def forward_dinov2_block_t3(self, x,patch_resolution):
    '''
        t3 means only adapter attn
    '''
    x = x + self.attn(self.ln1(x)) + self.adapter_attn(self.ln1(x),patch_resolution) * self.s
    x = self.ffn(self.ln2(x), identity=x)
    
    return x

def forward_eva02_block_t3(self, x,patch_resolution):
    '''
        t3 means only adapter attn
    '''
    x = x + self.drop_path(self.attn(self.norm1(x),patch_resolution)) + self.drop_path(self.adapter_attn(self.norm1(x),patch_resolution)) * self.s
    
    x = x + self.drop_path(self.mlp(self.norm2(x)))
    return x

def forward_sam_block_t3(self, x):
    
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
    x = x + self.mlp(self.norm2(x))
    
    return x


def forward_dinov2_block_t4(self, x,patch_resolution):
    '''
        t4 means attn + mlp + all
    '''
    # for dino v2
    conv_shortcut = self.adapter_all(self.ln1(x),patch_resolution) * self.s
    x = x + self.attn(self.ln1(x)) + self.adapter_attn(self.ln1(x),patch_resolution) * self.s
    x = self.ffn(self.ln2(x), identity=x) + self.adapter_mlp(self.ln2(x),patch_resolution) * self.s + conv_shortcut
    
    return x

def forward_eva02_block_t4(self, x,patch_resolution):
    # for dino v2
    conv_shortcut = self.adapter_all(self.norm1(x),patch_resolution) * self.s
    
    x = x + self.drop_path(self.attn(self.norm1(x),patch_resolution)) + self.drop_path(self.adapter_attn(self.norm1(x),patch_resolution)) * self.s
    
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x),patch_resolution)) * self.s + conv_shortcut
    return x

def forward_sam_block_t4(self, x):
    
    conv_shortcut = self.adapter_all(self.norm1(x)) * self.s
    
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(self.norm2(x)) * self.s  + conv_shortcut
    
    return x



def forward_sam_block_t5(self, x):
    
    # attn + all
    
    conv_shortcut = self.adapter_all(self.norm1(x)) * self.s
    
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
    x = x + self.mlp(self.norm2(x)) + conv_shortcut
    
    return x


def forward_sam_block_t6(self, x):
    
    # mlp + all
    
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(self.norm2(x)) * self.s  + conv_shortcut
    
    return x



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Convpass(nn.Module):
    def __init__(self, input_dim,patch_resolution,dim=8,xavier_init=False,linear_init='zero'):
        super().__init__()
        self.adapter_conv = ModuleList()
        
        
      
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)
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
        self.patch_resolution = patch_resolution

    def forward(self, x,patch_resolution):
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        patch_w,patch_h = patch_resolution[0],patch_resolution[1]   
            
        x_patch = x_down[:, 1:].reshape(B, patch_w, patch_h, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, patch_w * patch_h, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        return x_up


class Convpass_swin(nn.Module):
    def __init__(self, input_dim,dim=8, xavier_init=True,linear_init='zero'):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)
            
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
            
        x_patch = self.adapter_conv(x_patch)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    

class ConvpassVPT(nn.Module):
    def __init__(self, input_dim,patch_resolution,dim=8,xavier_init=False,linear_init='zero'):
        super().__init__()

        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)
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
        self.patch_resolution = patch_resolution

    def forward(self, x,patch_resolution):
        B, N, C = x.shape
        
        num_patch = patch_resolution[0]*patch_resolution[1]
        num_vpt_token = N - num_patch -1
        
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        patch_w,patch_h = patch_resolution[0],patch_resolution[1]   
            
        x_patch = x_down[:, 1+num_vpt_token:].reshape(B, patch_w, patch_h, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, patch_w * patch_h, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        x_cls = self.adapter_conv(x_cls)
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)
        
        vpt_w, vpt_h = self.find_closest_factors(num_vpt_token)
        x_vpt = x_down[:,1:1+num_vpt_token].reshape(B, vpt_w, vpt_h, self.dim).permute(0, 3, 1, 2)
        x_vpt = self.adapter_conv(x_vpt)
        x_vpt = x_vpt.permute(0, 2, 3, 1).reshape(B, num_vpt_token, self.dim)

        x_down = torch.cat([x_cls,x_vpt,x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        return x_up
        
        
    def find_closest_factors(self,n):
        sqrt_n = int(math.sqrt(n))
        factor1 = sqrt_n
        factor2 = sqrt_n
        
        while factor1 * factor2 != n:
            if factor1 * factor2 < n:
                factor2 += 1
            else:
                factor1 -= 1
        
        return factor1, factor2
    
    
    
    
class ConvpassVPT_swin(nn.Module):
    def __init__(self, input_dim,dim=8, xavier_init=True,linear_init='zero'):
        super().__init__()
        self.adapter_conv = nn.Conv2d(dim, dim, 3, 1, 1)
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(8, dtype=torch.float)
        nn.init.zeros_(self.adapter_conv.bias)

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

    def forward(self, x,patch_resolution):
        
        '''
        swin 没有 cls
        '''
        B, N, C = x.shape
        
        num_patch = patch_resolution[0]*patch_resolution[1]
        num_vpt_token = N - num_patch
        patch_w,patch_h = patch_resolution[0],patch_resolution[1]   
        x_down = self.adapter_down(x)
         
         
        x_patch = x_down[:, num_vpt_token:].reshape(B, patch_w, patch_h, self.dim).permute(0, 3, 1, 2)
        x_patch = self.adapter_conv(x_patch)
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, patch_w * patch_h, self.dim)
        
        
        vpt_w, vpt_h = self.find_closest_factors(num_vpt_token)
        x_vpt = x_down[:,:num_vpt_token].reshape(B, vpt_w, vpt_h, self.dim).permute(0, 3, 1, 2)
        x_vpt = self.adapter_conv(x_vpt)
        x_vpt = x_vpt.permute(0, 2, 3, 1).reshape(B, num_vpt_token, self.dim)
        
        
        x_down = torch.cat([x_vpt,x_patch], dim=1)
        
        # # H = int(math.sqrt(N))
        # x_down = self.adapter_down(x)
        # # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        # x_patch = x_down.permute(0, 3, 1, 2)
        # x_patch = self.act(x_patch)
        # x_patch = self.adapter_conv(x_patch)
        # x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    
    def find_closest_factors(self,n):
        sqrt_n = int(math.sqrt(n))
        factor1 = sqrt_n
        factor2 = sqrt_n
        
        while factor1 * factor2 != n:
            if factor1 * factor2 < n:
                factor2 += 1
            else:
                factor1 -= 1
        
        return factor1, factor2
    
    
    
    
    
class Convpass_swin_share(nn.Module):
    def __init__(self, shared_conv,input_dim,dim=8, xavier_init=True,linear_init='zero'):
        super().__init__()

        self.adapter_conv = shared_conv
        if xavier_init:
            nn.init.xavier_uniform_(self.adapter_conv.weight)
        else:
            nn.init.zeros_(self.adapter_conv.weight)
            self.adapter_conv.weight.data[:, :, 1, 1] += torch.eye(dim, dtype=torch.float)
            nn.init.zeros_(self.adapter_conv.bias)
            
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
        x_patch = self.adapter_conv(x_patch)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    
