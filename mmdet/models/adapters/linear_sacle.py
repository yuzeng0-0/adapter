import torch

import torch.nn as nn

class LinearScale(nn.Module):

    def __init__(self,
                 origin_layer,
               ):
        super(LinearScale, self).__init__()
        # in_features = original_layer.in_features
        # out_features = original_layer.out_features
        self.origin_layer = origin_layer
        self.adapter_scale = nn.Parameter(torch.ones(1))
        self.adapter_bias = nn.Parameter(torch.ones(1))
        
        # self.lora_dropout = nn.Dropout(drop_rate)
        # self.lora_down = nn.Linear(embed_dims, rank, bias=False)
        # self.lora_up = nn.Linear(rank, embed_dims, bias=False)
        # self.scaling = alpha / rank

        nn.init.constant_(self.adapter_scale, 1)
        nn.init.zeros_(self.adapter_bias)

        # self.original_layer = original_layer

    def forward(self, x: torch.Tensor):
        
        x = self.origin_layer(x)
        
        x = self.adapter_scale*(x + self.adapter_bias)
        
        return x