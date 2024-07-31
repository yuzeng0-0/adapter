import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict

from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition

class HyperNetwork_conv(nn.Module):
    # TODO add bias hypernetwork
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        '''
            f_size:kernel size
            out_size:out_channel
            in_size:in_channel
            z_dim:layer embedding dim
            
            This module generates the weights for the adapter conv layers
        
        '''
        super(HyperNetwork_conv, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        
        self.w1 = nn.Linear(self.z_dim, self.out_size*self.f_size*self.f_size)
        self.w2 = nn.Linear(self.z_dim, self.in_size*self.z_dim)
        
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w1.bias)
        nn.init.zeros_(self.w2.bias)


    def forward(self, z):

        h_in = self.w2(z)
        h_in = h_in.view(self.in_size, self.z_dim)
        
        h_final = self.w1(h_in)
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel
    

    
class SingleHyperNetwork_conv(nn.Module):
    # TODO add bias hypernetwork
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16,init_type='default'):
        '''
            f_size:kernel size
            out_size:out_channel
            in_size:in_channel
            z_dim:layer embedding dim
            
            This module generates the weights for the adapter conv layers
        
        '''
        super(SingleHyperNetwork_conv, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.init_type = init_type
        
      
            
        self.w = nn.Linear(self.z_dim, self.out_size*self.f_size*self.f_size*self.in_size)
            
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.zeros_(self.w.bias)
            
    

    def forward(self, z):
    
        h_final = self.w(z)
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel    
    

    
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)




class Convpass_swin_hypernet_mask_serial(nn.Module):
    def __init__(self, conv_hypernet,mask_token,input_dim,dim):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.mask_token = mask_token
            
        self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, 64)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(64, 64))
        ]))


        
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''
        input = x
        B, H,W, C = x.shape
        prompt_feat = self.meta_net(x)
        mask =  torch.matmul(prompt_feat,self.mask_token.weight.t())
        mask = mask.permute(3,0,1,2)
        num_mask = mask.shape[0]
        
        mask = mask.reshape(num_mask,B,-1)
        mask = mask.softmax(-1)
        mask = mask.reshape(num_mask,B,H,W)
        
        mask_wise_feat = torch.zeros_like(prompt_feat)
        for mask_id in range(num_mask):
            mask_wise_feat = mask_wise_feat + mask[mask_id].unsqueeze(-1) * prompt_feat
            
        mask_wise_feat = mask_wise_feat.sum(1).sum(1)
        
        
        x_up_all = []
        
        for idx in range(B):
        
      
            adapter_conv_weight = self.adapter_conv_hypernet(mask_wise_feat[idx])
            
            # name = 'adapter_conv_param/adapter_conv_{}.pt'.format(int(torch.rand(1).item()*10000))
            # torch.save(adapter_conv_weight,name)
            
            # H = int(math.sqrt(N))
            x_down = self.adapter_down(x[idx]).unsqueeze(0)
            # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
            x_patch = x_down.permute(0, 3, 1, 2)
            x_patch = self.act(x_patch)
    
            x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
                
            x_down = x_patch.permute(0, 2, 3, 1)
            x_down = self.act(x_down)
            x_down = self.dropout(x_down)
            x_up = self.adapter_up(x_down)
            
            x_up_all.append(x_up)
            
        x_up = torch.cat(x_up_all,dim=0)

        return x_up + input
    
  
        
class Convpass_hypernet_mask_serial(nn.Module):
    def __init__(self, conv_hypernet,mask_token,input_dim,dim):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.mask_token = mask_token
            
        self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, 64)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(64, 64))
        ]))


        
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x,patch_resolution):
        
        input = x
        
        B, L, C = x.shape
        patch_w,patch_h = patch_resolution[0],patch_resolution[1]   
        
        # cls_token = x[:,:1,:]
        
        # 计算mask wise feat
        prompt_feat = self.meta_net(x)
        mask =  torch.matmul(prompt_feat[:,1:,:],self.mask_token.weight.t())
        mask = mask.permute(2,0,1)
        num_mask = mask.shape[0]
        mask = mask.softmax(-1)
        
        mask_wise_feat = torch.zeros_like(prompt_feat[:,1:,:])
        for mask_id in range(num_mask):
            mask_wise_feat = mask_wise_feat + mask[mask_id].unsqueeze(-1) * prompt_feat[:,1:,:]
            
        mask_wise_feat = mask_wise_feat.sum(1)
        
        
        x_up_all = []
        
        for idx in range(B):
        
      
            adapter_conv_weight = self.adapter_conv_hypernet(mask_wise_feat[idx])
            name = 'adapter_conv_param/adapter_conv_{}.pt'.format(int(torch.rand(1).item()*10000))
            
            torch.save(adapter_conv_weight,name)
            
            # H = int(math.sqrt(N))
            x_down = self.adapter_down(x[idx]).unsqueeze(0)
            x_down = self.act(x_down)
            
            x_patch = x_down[:, 1:].reshape(1, patch_w, patch_h, self.dim).permute(0, 3, 1, 2)
            
            x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            x_patch = x_patch.permute(0, 2, 3, 1).reshape(1, patch_w * patch_h, self.dim)
            
            
            x_cls = x_down[:, :1].reshape(1, 1, 1, self.dim).permute(0, 3, 1, 2)
            x_cls = F.conv2d(x_cls, adapter_conv_weight, stride=1, padding=1)
            x_cls = x_cls.permute(0, 2, 3, 1).reshape(1, 1, self.dim)

            x_down = torch.cat([x_cls, x_patch], dim=1)

            x_down = self.act(x_down)
            x_down = self.dropout(x_down)
            x_up = self.adapter_up(x_down)
            
            x_up_all.append(x_up)
            
        x_up = torch.cat(x_up_all,dim=0)

        return x_up + input
    

    
    
 
    
    


    
    

def forward_dinov2_block_serial_V1(self, x,patch_resolution):
    
    shortcut = x 
    
    x= self.attn(self.ln1(x))
    
    x = self.adapter_attn(x)
    
    x = shortcut + x
    
    identity = x
    
    x = self.ln2(x)
    
    x = self.adapter_mlp(x)
    
    x = self.ffn(x,identity=identity)
    
    return x
    
    


def forward_sam_block_serial_V1(self, x):
    
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
        
        
    x = self.adapter_attn(x)

    x = shortcut + x
    
    identity = x
    
    x = self.norm2(x)
    
    x = self.adapter_mlp(x)
    
    x = self.mlp(x)
    
    x = identity + x
    
    
    return x

def forward_eva02_block_serial_V1(self, x,patch_resolution):
    
    inputs = x
    x = self.norm1(x)
    x = self.attn(x, patch_resolution)
    
    x = self.adapter_attn(x)
    
    x = self.drop_path(x)
    x = inputs + x



    inputs = x
    x = self.norm2(x)
    
    x = self.adapter_mlp(x)
    
    x = self.mlp(x)
    x = self.drop_path(x)
    x = inputs + x
    
    return x
    
    
    
def forward_dinov2_block_serial_V2(self, x,patch_resolution):
    
    shortcut = x 
    
    x= self.attn(self.ln1(x))
    
    x = self.adapter_attn(x)
    
    x = shortcut + x
    
    identity = x
    
    x = self.ln2(x)
    
    x = self.ffn(x,identity=identity)
        
    x = self.adapter_mlp(x)
    
    return x
    
    


def forward_sam_block_serial_V2(self, x):
    
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
        
        
    x = self.adapter_attn(x)

    x = shortcut + x
    
    identity = x
    
    x = self.norm2(x)
    
    
    x = self.mlp(x)
    x = x + identity
    
    x = self.adapter_mlp(x)
    
    # x = identity + x
    
    return x

def forward_eva02_block_serial_V2(self, x,patch_resolution):
    
    inputs = x
    x = self.norm1(x)
    x = self.attn(x, patch_resolution)
    
    x = self.adapter_attn(x)
    
    x = self.drop_path(x)
    x = inputs + x



    inputs = x
    x = self.norm2(x)
    
    
    x = self.mlp(x)
    x = self.drop_path(x)
    x = inputs + x
    
    x = self.adapter_mlp(x)
    
    return x