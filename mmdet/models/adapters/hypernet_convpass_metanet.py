import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict

class HyperNetwork_conv(nn.Module):
    # TODO add bias hypernetwork
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16,init_type='default'):
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
        self.init_type = init_type
        
        if init_type=='default':
            self.w1 = nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)),2))
            self.b1 = nn.Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)),2))

            self.w2 = nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)),2))
            self.b2 = nn.Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)),2))
            
        if init_type=='xavier':
            self.w1 = nn.Linear(self.z_dim, self.out_size*self.f_size*self.f_size)
            self.w2 = nn.Linear(self.z_dim, self.in_size*self.z_dim)
            
            nn.init.xavier_uniform_(self.w1.weight)
            nn.init.xavier_uniform_(self.w2.weight)
            nn.init.zeros_(self.w1.bias)
            nn.init.zeros_(self.w2.bias)
            
        if init_type=='kaiming':
            self.w1 = nn.Linear(self.z_dim, self.out_size*self.f_size*self.f_size)
            self.w2 = nn.Linear(self.z_dim, self.in_size*self.z_dim)
            
            nn.init.kaiming_uniform_(self.w1.weight)
            nn.init.kaiming_uniform_(self.w2.weight)
            nn.init.zeros_(self.w1.bias)
            nn.init.zeros_(self.w2.bias)
            

    def forward(self, z):
    
        if self.init_type=='default':

            h_in = torch.matmul(z, self.w2) + self.b2              # [8192]
            h_in = h_in.view(self.in_size, self.z_dim)             # [128,64]

            h_final = torch.matmul(h_in, self.w1) + self.b1         # [128,1152]
            kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)
            
        if self.init_type=='xavier' or self.init_type=='kaiming':
            h_in = self.w2(z)
            h_in = h_in.view(self.in_size, self.z_dim)
            
            h_final = self.w1(h_in)
            kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel
    
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)




class Convpass_swin_hypernet_fuse(nn.Module):
    def __init__(self, conv_hypernet,layer_embedding,input_dim,dim,fuse_type):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
            
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

        self.fuse_type = fuse_type

        
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''
        B, H,W, C = x.shape
        prompt_feat = self.meta_net(x)

        prompt_feat = torch.mean(prompt_feat,dim=(1,2))
        
        x_up_all = []
        
        for idx in range(B):
        
            if self.fuse_type=='add':
                adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding+prompt_feat[idx])
            if self.fuse_type=='avg':
                adapter_conv_weight = self.adapter_conv_hypernet((self.layer_embedding+prompt_feat[idx])/2)
            if self.fuse_type=='cat':  
                adapter_conv_weight = self.adapter_conv_hypernet(torch.cat((self.layer_embedding,prompt_feat[idx])))
                
            if self.fuse_type=='only_metafeat':
                adapter_conv_weight = self.adapter_conv_hypernet(prompt_feat[idx])
            
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

        return x_up
    
  
        

class Convpass_swin_hypernet_fusev2(nn.Module):
    
    '''
        将linear share
    '''
    
    def __init__(self, adapter_down,adapter_up,conv_hypernet,layer_embedding,input_dim,dim,fuse_type):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        
        
        self.adapter_down = adapter_down
        self.adapter_up = adapter_up
            
        # self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_up.weight)
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, 64)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(64, 64))
        ]))

        self.fuse_type = fuse_type

        
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''
        prompt_feat = self.meta_net(x)
        # NOTE 这里将两张图的feature进行了均值,有点不合理
        prompt_feat = torch.mean(prompt_feat,dim=(0,1,2))
        
        if self.fuse_type=='add':
            adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding+prompt_feat)
        if self.fuse_type=='avg':
            adapter_conv_weight = self.adapter_conv_hypernet((self.layer_embedding+prompt_feat)/2)
        if self.fuse_type=='cat':  
            adapter_conv_weight = self.adapter_conv_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
            
        if self.fuse_type=='only_metafeat':
            adapter_conv_weight = self.adapter_conv_hypernet(prompt_feat)
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
   
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    
    
class Convpass_swin_hypernet_fusev3(nn.Module):
    
    '''
        linear的参数也用hyper net生成
    '''
    
    def __init__(self, adapter_down_hypernet,adapter_up_hypernet,conv_hypernet,layer_embedding,input_dim,dim,fuse_type):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        
        
        self.adapter_down_hypernet = adapter_down_hypernet
        self.adapter_up_hypernet = adapter_up_hypernet
            
        # self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_up.weight)
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, 64)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(64, 64))
        ]))

        self.fuse_type = fuse_type

        
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''
        prompt_feat = self.meta_net(x)
        # NOTE 这里将两张图的feature进行了均值,有点不合理
        prompt_feat = torch.mean(prompt_feat,dim=(0,1,2))
        
        if self.fuse_type=='add':
            adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding+prompt_feat)
            adapter_down_lin_weight = self.adapter_down_hypernet(self.layer_embedding+prompt_feat)
            adapter_up_lin_weight = self.adapter_up_hypernet(self.layer_embedding+prompt_feat)
            
        if self.fuse_type=='avg':
            adapter_conv_weight = self.adapter_conv_hypernet((self.layer_embedding+prompt_feat)/2)
            adapter_down_lin_weight = self.adapter_down_hypernet((self.layer_embedding+prompt_feat)/2)
            adapter_up_lin_weight = self.adapter_up_hypernet((self.layer_embedding+prompt_feat)/2)
            
            
        if self.fuse_type=='cat':  
            adapter_conv_weight = self.adapter_conv_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
            
            adapter_down_lin_weight = self.adapter_down_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
            adapter_up_lin_weight = self.adapter_up_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
            
        if self.fuse_type=='only_metafeat':
            adapter_conv_weight = self.adapter_conv_hypernet(prompt_feat)
            adapter_down_lin_weight = self.adapter_down_hypernet(prompt_feat)
            adapter_up_lin_weight = self.adapter_up_hypernet(prompt_feat)
            
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        # x_down = self.adapter_down(x)
        
        x_down = F.linear(x,adapter_down_lin_weight.squeeze(-1).squeeze(-1))
        
        # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
   
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        
        x_up = F.linear(x_down,adapter_up_lin_weight.squeeze(-1).squeeze(-1))
        # x_up = self.adapter_up(x_down)

        return x_up
    
    
    

class Convpass_swin_hypernet_fuse_pos(nn.Module):
    def __init__(self, conv_hypernet,layer_embedding,input_dim,dim,fuse_type,fuse_pos):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        self.fuse_pos = fuse_pos   
            
        self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)
        
        if self.fuse_pos == 'after_down':
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(input_dim, 64)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(64, 64))
            ]))
        else:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(dim, 64)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(64, 64))
            ]))

        self.fuse_type = fuse_type

        
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward_after_conv(self,x):
        
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        
        x_up_all = []
        
        for idx in range(B):
            x_down = self.adapter_down(x[idx]).unsqueeze(0)
            prompt_feat = self.meta_net(x_down)
            prompt_feat = torch.mean(prompt_feat,dim=(1,2))
            
        
            if self.fuse_type=='add':
                adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding+prompt_feat)
            if self.fuse_type=='avg':
                adapter_conv_weight = self.adapter_conv_hypernet((self.layer_embedding+prompt_feat)/2)
            if self.fuse_type=='cat':  
                adapter_conv_weight = self.adapter_conv_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
                
            if self.fuse_type=='only_metafeat':
                adapter_conv_weight = self.adapter_conv_hypernet(prompt_feat)
                
                
            x_patch = x_down.permute(0, 3, 1, 2)
            x_patch = self.act(x_patch)
    
            x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
                
            x_down = x_patch.permute(0, 2, 3, 1)
            x_down = self.act(x_down)
            x_down = self.dropout(x_down)
            x_up = self.adapter_up(x_down)
            x_up_all.append(x_up)
            
        x_up = torch.cat(x_up_all,dim=0)
        
        return x_up



    def forward(self, x):
        
        
        if self.fuse_pos=='after_conv':
            return self.forward_after_conv(x)
        
        B, H,W, C = x.shape
        prompt_feat = self.meta_net(x)

        prompt_feat = torch.mean(prompt_feat,dim=(1,2))
        
        x_up_all = []
        
        for idx in range(B):
        
            if self.fuse_type=='add':
                adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding+prompt_feat[idx])
            if self.fuse_type=='avg':
                adapter_conv_weight = self.adapter_conv_hypernet((self.layer_embedding+prompt_feat[idx])/2)
            if self.fuse_type=='cat':  
                adapter_conv_weight = self.adapter_conv_hypernet(torch.cat((self.layer_embedding,prompt_feat[idx])))
                
            if self.fuse_type=='only_metafeat':
                adapter_conv_weight = self.adapter_conv_hypernet(prompt_feat[idx])
            
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

        return x_up
    
    
    
    

class Convpass_swin_hypernet_fusev2_pos(nn.Module):
    
    '''
        将linear share
    '''
    
    def __init__(self, adapter_down,adapter_up,conv_hypernet,layer_embedding,input_dim,dim,fuse_type,fuse_pos):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        self.fuse_pos = fuse_pos
        
        self.adapter_down = adapter_down
        self.adapter_up = adapter_up
            
        # self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_up.weight)
        
        
        
        if self.fuse_pos == 'after_down':
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(dim, 64)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(64, 64))
            ]))
        else:
            self.meta_net = nn.Sequential(OrderedDict([
                ("linear1", nn.Linear(input_dim, 64)),
                ("relu", nn.ReLU(inplace=True)),
                ("linear2", nn.Linear(64, 64))
            ]))
        self.fuse_type = fuse_type

        
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''
        if self.fuse_pos != 'after_down':
        
            prompt_feat = self.meta_net(x)
            # NOTE 这里将两张图的feature进行了均值,有点不合理
            prompt_feat = torch.mean(prompt_feat,dim=(0,1,2))
            
            if self.fuse_type=='add':
                adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding+prompt_feat)
            if self.fuse_type=='avg':
                adapter_conv_weight = self.adapter_conv_hypernet((self.layer_embedding+prompt_feat)/2)
            if self.fuse_type=='cat':  
                adapter_conv_weight = self.adapter_conv_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
                
            if self.fuse_type=='only_metafeat':
                adapter_conv_weight = self.adapter_conv_hypernet(prompt_feat)
            
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = self.adapter_down(x)
        
        if self.fuse_pos == 'after_down':
            prompt_feat = self.meta_net(x_down)
            # NOTE 这里将两张图的feature进行了均值,有点不合理
            prompt_feat = torch.mean(prompt_feat,dim=(0,1,2))
            
            if self.fuse_type=='add':
                adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding+prompt_feat)
            if self.fuse_type=='avg':
                adapter_conv_weight = self.adapter_conv_hypernet((self.layer_embedding+prompt_feat)/2)
            if self.fuse_type=='cat':  
                adapter_conv_weight = self.adapter_conv_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
                
            if self.fuse_type=='only_metafeat':
                adapter_conv_weight = self.adapter_conv_hypernet(prompt_feat)
            
        
        
        # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
   
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    
    
    
class Convpass_swin_hypernet_fusev3_pos(nn.Module):
    
    '''
        linear的参数也用hyper net生成
    '''
    
    def __init__(self, adapter_down_hypernet,adapter_up_hypernet,conv_hypernet,layer_embedding,input_dim,dim,fuse_type,fuse_pos):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        
        self.fuse_pos = fuse_pos
        self.adapter_down_hypernet = adapter_down_hypernet
        self.adapter_up_hypernet = adapter_up_hypernet
            
        # self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_up.weight)
        
        self.meta_net_down = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(input_dim, 64)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(64, 64))
        ]))
        
        
        self.meta_net_conv = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(dim, 64)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(64, 64))
        ]))
        
        self.meta_net_up = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(dim, 64)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(64, 64))
        ]))

        self.fuse_type = fuse_type

        
        
        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''

        adapter_down_lin_weight = self.get_weight(x,'down',self.fuse_type)
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        # x_down = self.adapter_down(x)
        
        x_down = F.linear(x,adapter_down_lin_weight.squeeze(-1).squeeze(-1))
        
        
        
        adapter_conv_weight = self.get_weight(x_down,'conv',self.fuse_type)
        
        # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
   
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        
        adapter_up_lin_weight = self.get_weight(x_down,'up',self.fuse_type)
        
        x_up = F.linear(x_down,adapter_up_lin_weight.squeeze(-1).squeeze(-1))
        # x_up = self.adapter_up(x_down)

        return x_up
    
    
    
    def get_weight(self,x,pos,fuse_type):
        
        
        assert pos in ['down','up','conv']
        
        if pos=='down':
            prompt_feat = self.meta_net_down(x)
            prompt_feat = torch.mean(prompt_feat,dim=(0,1,2))
            
            
            if self.fuse_type=='add':
                weight = self.adapter_down_hypernet(self.layer_embedding+prompt_feat)
                
            if self.fuse_type=='avg':
                weight = self.adapter_down_hypernet((self.layer_embedding+prompt_feat)/2)
                
                
            if self.fuse_type=='cat':  
                weight = self.adapter_down_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
                
            if self.fuse_type=='only_metafeat':
                weight = self.adapter_down_hypernet(prompt_feat)
                        
                
            
        elif pos =='conv':
            prompt_feat = self.meta_net_conv(x) 
            prompt_feat = torch.mean(prompt_feat,dim=(0,1,2))
            
            if self.fuse_type=='add':
                weight = self.adapter_conv_hypernet(self.layer_embedding+prompt_feat)
                
            if self.fuse_type=='avg':
                weight = self.adapter_conv_hypernet((self.layer_embedding+prompt_feat)/2)
                
                
            if self.fuse_type=='cat':  
                weight = self.adapter_conv_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
                
                
            if self.fuse_type=='only_metafeat':
                weight = self.adapter_conv_hypernet(prompt_feat)
            
            
            
        
        else:
            prompt_feat = self.meta_net_up(x) 
            prompt_feat = torch.mean(prompt_feat,dim=(0,1,2))
            
            if self.fuse_type=='add':
                weight = self.adapter_up_hypernet(self.layer_embedding+prompt_feat)
                
            if self.fuse_type=='avg':
                weight = self.adapter_up_hypernet((self.layer_embedding+prompt_feat)/2)
                
                
            if self.fuse_type=='cat':  
                weight = self.adapter_up_hypernet(torch.cat((self.layer_embedding,prompt_feat)))
                
                
            if self.fuse_type=='only_metafeat':
                weight = self.adapter_up_hypernet(prompt_feat)
            
        return weight
            

            
