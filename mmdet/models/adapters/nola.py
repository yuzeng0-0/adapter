
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

class Nola(nn.Module):
    r"""
        NOLA: NETWORKS AS LINEAR COMBINATION OF LOWRANK RANDOM BASIS
    """

    def __init__(self,
                 original_layer,
                 random_down,
                 random_up,
                 drop_rate: float = 0.,
                 scaling:float = 1.
                 ):
        super(Nola, self).__init__()
        # in_features = original_layer.in_features
        # out_features = original_layer.out_features

        self.nola_dropout = nn.Dropout(drop_rate)
        # self.lora_down = nn.Linear(in_features, rank, bias=False)
        # self.lora_up = nn.Linear(rank, out_features, bias=False)
        
        self.random_down = random_down
        self.random_up = random_up
        
        
        self.nola_down_scale = nn.Parameter(torch.randn(( self.random_down.shape[0])))
        self.nola_up_scale = nn.Parameter(torch.randn(( self.random_up.shape[0])))
        
        
        nn.init.xavier_uniform_(self.nola_down_scale.unsqueeze(0))
        nn.init.xavier_uniform_(self.nola_up_scale.unsqueeze(0))
        self.scaling = scaling

        # nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        # nn.init.zeros_(self.lora_up.weight)

        self.original_layer = original_layer

    def forward(self, x: torch.Tensor):
        out = self.original_layer(x)
        nola_x = self.nola_dropout(x)
        
        nola_down = torch.sum(self.random_down *  self.nola_down_scale.view(-1,1,1),dim=0)
        nola_up = torch.sum(self.random_up *  self.nola_up_scale.view(-1,1,1),dim=0)
        B,H,W,C = nola_x.shape
        nola_x = nola_x.view(-1,C)
        
        nola_out = torch.matmul(torch.matmul(nola_x,nola_down),nola_up) * self.scaling
        
        nola_out = nola_out.view(B,H,W,-1)

        return out + nola_out