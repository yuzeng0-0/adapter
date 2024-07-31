import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict
import torch.utils.checkpoint as cp

from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition

class HyperNetwork_conv(nn.Module):
    # TODO add bias hypernetwork
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16,init_type='default',low_dim = 8):
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
        self.low_dim  = low_dim
        
        if init_type=='default':
            self.w1 = nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size*self.f_size*self.f_size)),2))
            self.b1 = nn.Parameter(torch.fmod(torch.randn((self.out_size*self.f_size*self.f_size)),2))

            self.w2 = nn.Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size*self.z_dim)),2))
            self.b2 = nn.Parameter(torch.fmod(torch.randn((self.in_size*self.z_dim)),2))
            
        if init_type=='xavier':
            self.w1 = nn.Linear(self.low_dim, self.out_size*self.f_size*self.f_size)
            self.w2 = nn.Linear(self.z_dim, self.in_size*self.low_dim)
            
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
            h_in = h_in.view(self.in_size, self.low_dim)
            
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
    
    

class SingleHyperNetwork_conv_lora(nn.Module):
    # TODO add bias hypernetwork
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16,init_type='default',low_dim=8):
        '''
            f_size:kernel size
            out_size:out_channel
            in_size:in_channel
            z_dim:layer embedding dim
            
            This module generates the weights for the adapter conv layers
        
        '''
        super(SingleHyperNetwork_conv_lora, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.init_type = init_type
        
        self.w1 = nn.Linear(self.z_dim, low_dim)
            
        self.w2 = nn.Linear(low_dim, self.out_size*self.f_size*self.f_size*self.in_size)
            
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.zeros_(self.w1.bias)
        
                    
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
            
    
    def forward(self, z):
    
        h_final = self.w2(self.w1(z))
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel    
    
class SingleHyperNetwork_conv_polyhistor(nn.Module):
    
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16,init_type='default',low_dim=8):
        '''
            f_size:kernel size
            out_size:out_channel
            in_size:in_channel
            z_dim:layer embedding dim
            
            This module generates the weights for the adapter conv layers
        
        '''
        super(SingleHyperNetwork_conv_polyhistor, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.init_type = init_type
        self.low_dim = low_dim
        
        self.param_dict={}
        count = 0
        while count < self.f_size*self.f_size:
            
            w1_name = 'w1_' + str(count)
            w2_name = 'w2_' + str(count)
            
            self.param_dict[w1_name] = nn.Linear(self.z_dim, self.low_dim*self.in_size).cuda()
            self.param_dict[w2_name] = nn.Linear(self.z_dim, self.out_size*self.low_dim).cuda()
        
            nn.init.xavier_uniform_(self.param_dict[w1_name].weight)
            nn.init.zeros_(self.param_dict[w1_name].bias)
            
                        
            nn.init.xavier_uniform_(self.param_dict[w2_name].weight)
            nn.init.zeros_(self.param_dict[w2_name].bias)
            count = count+1
                
    
    def forward(self, z):
        
        param_list = []
        count = 0
        
        while count < self.f_size*self.f_size:
            w1_name = 'w1_' + str(count)
            w2_name = 'w2_' + str(count)
            
            h1 = self.param_dict[w1_name](z)
            h2 = self.param_dict[w2_name](z)
            
            h1 = h1.view(self.low_dim,self.in_size)
            h2 = h2.view(self.low_dim,self.out_size)
            
            h_final = torch.mm(h1.t(),h2)
            
            param_list.append(h_final)
            count = count +1
            
        kernel = torch.stack(param_list,dim=-1)
        kernel = kernel.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel       
    
class SingleHyperNetwork_conv_lorand(nn.Module):
    # TODO add bias hypernetwork
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16,init_type='default',num_branch=2,kernel_dim=8):
        '''
            f_size:kernel size
            out_size:out_channel
            in_size:in_channel
            z_dim:layer embedding dim
            
            This module generates the weights for the adapter conv layers
        
        '''
        super(SingleHyperNetwork_conv_lorand, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.init_type = init_type
        
        self.num_branch = num_branch
        
        self.param_dict={}
        count = 0
        while count < num_branch:
            # Down Projection
            S_layer_name = 'S'+str(count)
            D_layer_name = 'D'+str(count)
            self.param_dict[S_layer_name] = nn.Parameter(torch.Tensor(z_dim, kernel_dim))
            self.param_dict[D_layer_name] = nn.Parameter(torch.Tensor(kernel_dim,  self.out_size*self.f_size*self.f_size*self.in_size))
            nn.init.kaiming_uniform_(self.param_dict[S_layer_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.param_dict[D_layer_name], a=math.sqrt(5))
            # Kernel
            V_layer_name = 'V'+str(count)
            self.param_dict[V_layer_name] = nn.Parameter(torch.Tensor(kernel_dim, kernel_dim))
            nn.init.kaiming_uniform_(self.param_dict[V_layer_name], a=math.sqrt(5))
            count += 1

        self.param_dict = nn.ParameterDict(self.param_dict)
        self.bias = nn.Parameter(torch.Tensor(self.out_size*self.f_size*self.f_size*self.in_size))
        
        
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
        if self.bias is not None:
            # fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            count = 0
            weight = 0
            while count < self.num_branch:
                weight = weight +  torch.mm(torch.mm(self.param_dict['S'+str(count)], self.param_dict['V'+str(count)]),
                                            self.param_dict['D'+str(count)]).t()
                count = count + 1
            # weight = weight/self.num_branch        # avg_weight

            fan_in_D, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound_D = 1 / math.sqrt(fan_in_D)
            nn.init.uniform_(self.bias, -bound_D, bound_D)

   
        
    def forward(self, z):
        
        count = 0
        weight = 0
        while count < self.num_branch:
            weight = weight +  torch.mm(torch.mm(self.param_dict['S'+str(count)], self.param_dict['V'+str(count)]),
                                           self.param_dict['D'+str(count)]).t()
            count = count + 1
    
        h_final = F.linear(z,weight,self.bias)
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel  
    

class HyperNetwork_conv_lorandv1(nn.Module):
    # TODO add bias hypernetwork
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16,init_type='default',num_branch=2,kernel_dim = 8):
        '''
            f_size:kernel size
            out_size:out_channel
            in_size:in_channel
            z_dim:layer embedding dim
            
            This module generates the weights for the adapter conv layers
        
        '''
        '''
            v1 means only decompose w1
        '''
        super(HyperNetwork_conv_lorandv1, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.init_type = init_type
        
        self.num_branch = num_branch
        
        self.param_dict={}
        count = 0
        while count < num_branch:
            # Down Projection
            S_layer_name = 'S'+str(count)
            D_layer_name = 'D'+str(count)
            self.param_dict[S_layer_name] = nn.Parameter(torch.Tensor(z_dim, kernel_dim))
            self.param_dict[D_layer_name] = nn.Parameter(torch.Tensor(kernel_dim,  self.out_size*self.f_size*self.f_size))
            nn.init.kaiming_uniform_(self.param_dict[S_layer_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.param_dict[D_layer_name], a=math.sqrt(5))
            # Kernel
            V_layer_name = 'V'+str(count)
            self.param_dict[V_layer_name] = nn.Parameter(torch.Tensor(kernel_dim, kernel_dim))
            nn.init.kaiming_uniform_(self.param_dict[V_layer_name], a=math.sqrt(5))
            count += 1
            
            
        self.param_dict = nn.ParameterDict(self.param_dict)
        self.w1_bias = nn.Parameter(torch.Tensor(self.out_size*self.f_size*self.f_size))
        
        # self.w1 = nn.Linear(self.z_dim, self.out_size*self.f_size*self.f_size)
        self.w2 = nn.Linear(self.z_dim, self.in_size*self.z_dim)
        
        nn.init.xavier_uniform_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
            
    def forward(self, z):

        h_in = self.w2(z)
        h_in = h_in.view(self.in_size, self.z_dim)
        
        count = 0
        w1_weight = 0
        while count < self.num_branch:
            w1_weight = w1_weight +  torch.mm(torch.mm(self.param_dict['S'+str(count)], self.param_dict['V'+str(count)]),
                                           self.param_dict['D'+str(count)]).t()
            count = count + 1
    
        
        h_final = F.linear(h_in,w1_weight,self.w1_bias)
         
        # h_final = self.w1(h_in)
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel
    
    

class HyperNetwork_conv_lorandv2(nn.Module):
    # TODO add bias hypernetwork
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16,init_type='default',num_branch=2,kernel_dim = 8):
        '''
            f_size:kernel size
            out_size:out_channel
            in_size:in_channel
            z_dim:layer embedding dim
            
            This module generates the weights for the adapter conv layers
        
        '''
        '''
            v1 means decompose w1 and w2
        '''
        super(HyperNetwork_conv_lorandv2, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size
        self.init_type = init_type
        
        self.num_branch = num_branch
        
        self.param_dict={}
        count = 0
        while count < num_branch:
            # w1 Projection
            w1_S_layer_name = 'w1_S'+str(count)
            w1_D_layer_name = 'w1_D'+str(count)
            self.param_dict[w1_S_layer_name] = nn.Parameter(torch.Tensor(z_dim, kernel_dim))
            self.param_dict[w1_D_layer_name] = nn.Parameter(torch.Tensor(kernel_dim,  self.out_size*self.f_size*self.f_size))
            nn.init.kaiming_uniform_(self.param_dict[w1_S_layer_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.param_dict[w1_D_layer_name], a=math.sqrt(5))
            # Kernel
            w1_V_layer_name = 'w1_V'+str(count)
            self.param_dict[w1_V_layer_name] = nn.Parameter(torch.Tensor(kernel_dim, kernel_dim))
            nn.init.kaiming_uniform_(self.param_dict[w1_V_layer_name], a=math.sqrt(5))
            
            w2_S_layer_name = 'w2_S'+str(count)
            w2_D_layer_name = 'w2_D'+str(count)
            self.param_dict[w2_S_layer_name] = nn.Parameter(torch.Tensor(z_dim, kernel_dim))
            self.param_dict[w2_D_layer_name] = nn.Parameter(torch.Tensor(kernel_dim,  self.in_size*self.z_dim))
            nn.init.kaiming_uniform_(self.param_dict[w2_S_layer_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.param_dict[w2_D_layer_name], a=math.sqrt(5))
            # Kernel
            w2_V_layer_name = 'w2_V'+str(count)
            self.param_dict[w2_V_layer_name] = nn.Parameter(torch.Tensor(kernel_dim, kernel_dim))
            nn.init.kaiming_uniform_(self.param_dict[w2_V_layer_name], a=math.sqrt(5))
            
            
            count += 1
            
        self.param_dict = nn.ParameterDict(self.param_dict)
        self.w1_bias = nn.Parameter(torch.Tensor(self.out_size*self.f_size*self.f_size))
        self.w2_bias = nn.Parameter(torch.Tensor(self.in_size*self.z_dim))
        
        # self.w1 = nn.Linear(self.z_dim, self.out_size*self.f_size*self.f_size)
        # self.w2 = nn.Linear(self.z_dim, self.in_size*self.z_dim)
        
            
    def forward(self, z):

        # h_in = self.w2(z)
        # h_in = h_in.view(self.in_size, self.z_dim)
        
        count = 0
        w1_weight = 0
        w2_weight = 0
        while count < self.num_branch:
            w1_weight = w1_weight +  torch.mm(torch.mm(self.param_dict['w1_S'+str(count)], self.param_dict['w1_V'+str(count)]),
                                           self.param_dict['w1_D'+str(count)]).t()
            
            w2_weight = w2_weight +  torch.mm(torch.mm(self.param_dict['w2_S'+str(count)], self.param_dict['w2_V'+str(count)]),
                                           self.param_dict['w2_D'+str(count)]).t()
            
            count = count + 1
    
        h_in = F.linear(z,w2_weight,self.w2_bias)
        h_in = h_in.view(self.in_size, self.z_dim)
        
        h_final = F.linear(h_in,w1_weight,self.w1_bias)
         
        # h_final = self.w1(h_in)
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel
    
    
    
    
    
    
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)




class Convpass_swin_hypernet_mask(nn.Module):
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

        return x_up
    

class Convpass_hypernet_mask(nn.Module):
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
        
        '''
        swin 没有 cls
        '''
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
            # name = 'adapter_conv_param/adapter_conv_{}.pt'.format(int(torch.rand(1).item()*10000))
            
            # torch.save(adapter_conv_weight,name)
            
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

        return x_up
    

    
class Convpass_swintransformer_hypernet_mask(nn.Module):
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

    def forward(self, x,hw_shape):
        
        '''
        swin 没有 cls
        '''
        B, L, C = x.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        x = x.view(B,H,W,C)
        
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
        x_up = x_up.view(B,L,C)

        return x_up
    
    
 
    
    


    
    

def forward_dinov2_block_t1(self, x,patch_resolution):
    '''
        t1 means attn + mlp
    '''
    # for dino v2
    x = x + self.attn(self.ln1(x)) + self.adapter_attn(self.ln1(x),patch_resolution) * self.s
    x = self.ffn(self.ln2(x), identity=x) + self.adapter_mlp(self.ln2(x),patch_resolution) * self.s
    
    return x


def forward_sam_block_t1(self, x):
    '''
        t1 means attn + mlp
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(self.norm2(x)) * self.s
    
    return x

def forward_eva02_block_t1(self, x,patch_resolution):
    # for dino v2
    x = x + self.drop_path(self.attn(self.norm1(x),patch_resolution)) + self.drop_path(self.adapter_attn(self.norm1(x),patch_resolution)) * self.s
    
    x = x + self.drop_path(self.mlp(self.norm2(x))) + self.drop_path(self.adapter_mlp(self.norm2(x),patch_resolution)) * self.s
    return x


def forward_swin_block_t1(self, x,hw_shape):
    
    def _inner_forward(x):
        identity = x + self.adapter_attn(self.norm1(x),hw_shape) * self.s
        x = self.norm1(x)
        x = self.attn(x, hw_shape)

        x = x + identity

        identity = x
        
        x = self.norm2(x)
        x = self.ffn(x, identity=identity) + self.adapter_mlp(x,hw_shape) * self.s

        return x

    if self.with_cp and x.requires_grad:
        x = cp.checkpoint(_inner_forward, x)
    else:
        x = _inner_forward(x)

    return x

def forward_sam_block_t2(self, x):
    '''
        t2 means only adapter mlp
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
    x = x + self.mlp(self.norm2(x)) +  self.adapter_mlp(self.norm2(x)) * self.s
    
    return x

def forward_sam_block_t3(self, x):
    '''
        t3 means only adapter attn
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
    x = x + self.mlp(self.norm2(x))
    
    return x


def forward_sam_block_t4(self, x):
    
    '''
        t4 means attn + mlp + all
    '''
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