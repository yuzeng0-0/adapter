import torch.nn as nn
import torch
import torch.nn.functional as F
import math

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

class Convpass_swin_hypernet(nn.Module):
    def __init__(self, conv_hypernet,conv_bias_hypernet,layer_embedding,input_dim,dim):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        self.conv_bias_hypernet = conv_bias_hypernet
            
        self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
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
        if self.conv_bias_hypernet is not None:
            bias = self.conv_bias_hypernet(self.layer_embedding)
            x_patch = F.conv2d(x_patch, adapter_conv_weight, bias,stride=1, padding=1)
        else: 
            x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    
    
    
class Convpass_swin_hypernet_sharelinear(nn.Module):
    def __init__(self, conv_hypernet,conv_bias_hypernet,adapter_down,adapter_up,layer_embedding,input_dim,dim):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        self.conv_bias_hypernet = conv_bias_hypernet
        
        self.adapter_down = adapter_down
        self.adapter_up = adapter_up
            
        # self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_up.weight)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
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
        if self.conv_bias_hypernet is not None:
            bias = self.conv_bias_hypernet(self.layer_embedding)
            x_patch = F.conv2d(x_patch, adapter_conv_weight, bias,stride=1, padding=1)
        else: 
            x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)

        return x_up
    
    
    
class Convpass_swin_hypernet_divlinear(nn.Module):
    def __init__(self, conv_hypernet,conv_bias_hypernet,layer_embedding,num_branch,kernel_dim,input_dim,dim,linear_bias=True):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        self.conv_bias_hypernet = conv_bias_hypernet
        
        # self.adapter_down = adapter_down
        # self.adapter_up = adapter_up
            
        # self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_up.weight)
        self.num_branch = num_branch
        self.param_dict={}
        count = 0
        while count < num_branch:
            # Down Projection
            DP_layer_name = 'DP'+str(count)
            DQ_layer_name = 'DQ'+str(count)
            self.param_dict[DP_layer_name] = nn.Parameter(torch.Tensor(input_dim, kernel_dim))
            self.param_dict[DQ_layer_name] = nn.Parameter(torch.Tensor(kernel_dim, dim))
            nn.init.kaiming_uniform_(self.param_dict[DP_layer_name], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.param_dict[DQ_layer_name], a=math.sqrt(5))
            # Kernel
            K_layer_name = 'K'+str(count)
            self.param_dict[K_layer_name] = nn.Parameter(torch.Tensor(kernel_dim, kernel_dim))
            nn.init.kaiming_uniform_(self.param_dict[K_layer_name], a=math.sqrt(5))
            # Up Projection
            UP_layer_name = 'UP' + str(count)
            UQ_layer_name = 'UQ' + str(count)
            self.param_dict[UP_layer_name] = nn.Parameter(torch.Tensor(dim, kernel_dim))
            self.param_dict[UQ_layer_name] = nn.Parameter(torch.Tensor(kernel_dim, input_dim))
            # 这里需要初始化为0，来保证计算出的weight
            nn.init.zeros_(self.param_dict[UP_layer_name])
            nn.init.kaiming_uniform_(self.param_dict[UQ_layer_name], a=math.sqrt(5))
            count += 1

        self.param_dict = nn.ParameterDict(self.param_dict)
        if linear_bias:
            self.bias_D = nn.Parameter(torch.Tensor(dim))
            self.bias_U = nn.Parameter(torch.Tensor(input_dim))
            nn.init.zeros_(self.bias_D)
            nn.init.zeros_(self.bias_U)
        else:
            self.register_parameter('bias', None)
        
        

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''
        adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding)
        
        count = 0
        weight_D = 0
        weight_U = 0
        while count < self.num_branch:
            weight_D = weight_D + torch.mm(torch.mm(self.param_dict['DP' + str(count)], self.param_dict['K' + str(count)]),
                                       self.param_dict['DQ' + str(count)]).t()
            weight_U = weight_U + torch.mm(torch.mm(self.param_dict['UP' + str(count)], self.param_dict['K' + str(count)]),
                                       self.param_dict['UQ' + str(count)])
            count += 1
        
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        x_down = F.linear(x, weight_D, self.bias_D)
        # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        if self.conv_bias_hypernet is not None:
            bias = self.conv_bias_hypernet(self.layer_embedding)
            x_patch = F.conv2d(x_patch, adapter_conv_weight, bias,stride=1, padding=1)
        else: 
            x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up =  F.linear(x_down,weight_U.t(), self.bias_U)

        return x_up
    
    
    
    
class Convpass_swin_hypernet_linconv(nn.Module):
    def __init__(self, conv_hypernet,linear_hypernet,layer_embedding,lin_type_embeding):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        self.adapter_linear_hypernet = linear_hypernet
        self.lin_type_embeding = lin_type_embeding
            
        # self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        # self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        # nn.init.zeros_(self.adapter_down.bias)
        # nn.init.zeros_(self.adapter_up.bias)
        # nn.init.xavier_uniform_(self.adapter_down.weight)
        # nn.init.zeros_(self.adapter_up.weight)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        
        '''
        swin 没有 cls
        '''
        adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding)
        adapter_down_weight = self.adapter_linear_hypernet(self.layer_embedding + self.lin_type_embeding.weight[0]).squeeze(-1).squeeze(-1)
        adapter_up_weight = self.adapter_linear_hypernet(self.layer_embedding + self.lin_type_embeding.weight[1]).squeeze(-1).squeeze(-1)
        
        B, H,W, C = x.shape
        # H = int(math.sqrt(N))
        # x_down = self.adapter_down(x)
        
        x_down = F.linear(x,adapter_down_weight)
        # x_patch = x_down.reshape(B, H, H, self.dim).permute(0, 3, 1, 2)
        x_patch = x_down.permute(0, 3, 1, 2)
        x_patch = self.act(x_patch)
        
        x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
            
        x_down = x_patch.permute(0, 2, 3, 1)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        # x_up = self.adapter_up(x_down)
        x_up = F.linear(x_down,adapter_up_weight.t())

        return x_up
    
    
    

class Convpass_hypernet(nn.Module):
    def __init__(self, conv_hypernet,conv_bias_hypernet,layer_embedding,input_dim,dim):
        super().__init__()
        self.adapter_conv_hypernet = conv_hypernet
        self.layer_embedding = layer_embedding
        self.conv_bias_hypernet = conv_bias_hypernet
            
        self.adapter_down = nn.Linear(input_dim, dim)  # equivalent to 1 * 1 Conv
        self.adapter_up = nn.Linear(dim, input_dim)  # equivalent to 1 * 1 Conv
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.bias)
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_up.weight)

        self.act = QuickGELU()
        self.dropout = nn.Dropout(0.1)
        self.dim = dim

    
    
    def forward(self, x,patch_resolution):
        
        adapter_conv_weight = self.adapter_conv_hypernet(self.layer_embedding)
        B, N, C = x.shape

        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        x_down = self.act(x_down)

        patch_w,patch_h = patch_resolution[0],patch_resolution[1]   
            
        x_patch = x_down[:, 1:].reshape(B, patch_w, patch_h, self.dim).permute(0, 3, 1, 2)
        
        # x_patch = self.adapter_conv(x_patch)
        if self.conv_bias_hypernet is not None:
            bias = self.conv_bias_hypernet(self.layer_embedding)
            x_patch = F.conv2d(x_patch, adapter_conv_weight, bias,stride=1, padding=1)
        else: 
            x_patch = F.conv2d(x_patch, adapter_conv_weight, stride=1, padding=1)
        
        
        
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, patch_w * patch_h, self.dim)

        x_cls = x_down[:, :1].reshape(B, 1, 1, self.dim).permute(0, 3, 1, 2)
        
        if self.conv_bias_hypernet is not None:
            bias = self.conv_bias_hypernet(self.layer_embedding)
            x_cls = F.conv2d(x_cls, adapter_conv_weight, bias,stride=1, padding=1)
        else: 
            x_cls = F.conv2d(x_cls, adapter_conv_weight, stride=1, padding=1)
        
        x_cls = x_cls.permute(0, 2, 3, 1).reshape(B, 1, self.dim)

        x_down = torch.cat([x_cls, x_patch], dim=1)

        x_down = self.act(x_down)
        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # equivalent to 1 * 1 Conv
        return x_up
