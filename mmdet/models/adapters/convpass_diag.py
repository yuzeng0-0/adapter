
import torch
from torch import nn
import timm
import math

from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition
from mmengine.model.base_module import ModuleList
import torch.nn.functional as F



class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    
    
class Convpass_swin_hypernet_diag(nn.Module):
    '''
        带有乘对角矩阵的hypnet swin
    '''
    def __init__(self, conv_hypernet,layer_embedding,input_dim,dim,diag_pos):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        self.diag_pos = diag_pos
            
        self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)

        if self.diag_pos == 'after_up':
            self.adapter_scale = nn.Parameter(torch.ones(input_dim))
        if self.diag_pos == 'after_down' or self.diag_pos == 'after_conv':
            self.adapter_scale = nn.Parameter(torch.ones(dim))
            
        if self.diag_pos == 'after_down_conv':
            self.adapter_scale_dwon = nn.Parameter(torch.ones(dim))
            self.adapter_scale_conv = nn.Parameter(torch.ones(dim))
        # self.adapter_diag = torch.diag(self.adapter_scale).cuda()

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
        
        
        
    def forward(self, x):
        
        assert self.diag_pos in ['after_down','after_conv','after_up','after_down_conv']
        
        if self.diag_pos =='after_down':
            return self.forward_after_down(x)
        
        if self.diag_pos =='after_conv':
            return self.forward_after_conv(x)
        
        if self.diag_pos =='after_up':
            return self.forward_after_up(x)
        
        if self.diag_pos =='after_down_conv':
            return self.forward_after_down_conv(x)
        
        

    def forward_after_conv(self, x):
        
        '''
        swin 没有 cls
        '''
        adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding)
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
        
        x_down = x_patch.permute(0, 2, 3, 1)
         # 乘以对角矩阵
        adapter_diag = torch.diag(self.adapter_scale)
        x_down = torch.matmul(x_down,adapter_diag)
        
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    def forward_after_down(self, x):
        
        
        adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding)
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        
        # 乘以对角矩阵
        adapter_diag = torch.diag(self.adapter_scale)
        x_down = torch.matmul(x_down,adapter_diag)
        
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
        
        x_down = x_patch.permute(0, 2, 3, 1)
        
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    def forward_after_up(self, x):
        
        
        adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding)
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
        
        x_down = x_patch.permute(0, 2, 3, 1)

        
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

                
        # 乘以对角矩阵
        adapter_diag = torch.diag(self.adapter_scale)
        x_up = torch.matmul(x_up,adapter_diag)
        
        return x_up
    
    
    def forward_after_down_conv(self, x):
        
        
        adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding)
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        
        # 乘以对角矩阵
        adapter_diag_down = torch.diag(self.adapter_scale_dwon)
        x_down = torch.matmul(x_down,adapter_diag_down)
        
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
        
        x_down = x_patch.permute(0, 2, 3, 1)
         # 乘以对角矩阵
        adapter_diag_conv = torch.diag(self.adapter_scale_conv)
        x_down = torch.matmul(x_down,adapter_diag_conv)
        
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    
    
    
class Convpass_swin_hypernet_sharelin_diag(nn.Module):
    '''
        带有乘对角矩阵的hypnet swin   同时share linear
    '''
    def __init__(self,conv_hypernet,layer_embedding,adapter_down,adapter_up,input_dim,dim):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        # self.diag_pos = diag_pos
            
        # self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_up.weight)

        self.adapter_down = adapter_down
        self.adapter_up = adapter_up
        
        self.adapter_scale_down = nn.Parameter(torch.ones(dim))
        self.adapter_scale_up = nn.Parameter(torch.zeros(input_dim))
        # self.adapter_diag = torch.diag(self.adapter_scale).cuda()

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim
        
        
        
    def forward(self, x):
        
        
        adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding)
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        
        # 乘以down对角矩阵
        adapter_diag_down = torch.diag(self.adapter_scale_down)
        x_down = torch.matmul(x_down,adapter_diag_down)
        
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
        
        x_down = x_patch.permute(0, 2, 3, 1)

        
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

                
        # 乘以对角矩阵
        adapter_diag_up = torch.diag(self.adapter_scale_up)
        x_up = torch.matmul(x_up,adapter_diag_up)
        
        return x_up