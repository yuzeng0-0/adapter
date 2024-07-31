
from .vision_transformer import VisionTransformer
from mmdet.registry import MODELS
import torch.nn as nn
import math
import torch
# from functions import reduce
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from .vision_transformer import resize_pos_embed

@MODELS.register_module()
class VisionTransformerVPT(VisionTransformer):
    
    
    def __init__(
        self,
        prompt_config:dict,
         **kwargs
    ):
        super().__init__(**kwargs)
    
        self.prompt_config = prompt_config
        self.num_vpt_token = self.prompt_config['num_token']
        self.prompt_dropout =  nn.Dropout(self.prompt_config.dropout)
        patch_size = _pair(kwargs['patch_size'])
        
                # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(
                prompt_dim,  self.embed_dims)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim =  self.embed_dims
            self.prompt_proj = nn.Identity()
            
            
        # initiate prompt:
        if self.prompt_config['initiation_type'] == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + prompt_dim))  # noqa

            # self.prompt_embeddings = nn.Parameter(torch.zeros(
            #     1, self.num_token, prompt_dim))
            self.prompt_embeddings = nn.Parameter(
                torch.zeros(1, self.num_vpt_token, prompt_dim)
            )
            # xavier_uniform initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if self.prompt_config['type']=='deep':  # noqa

                total_d_layer = len(self.blocks) - 1
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.num_vpt_token, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")
        
        
    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens)
        
        x = self.incorporate_prompt(x)
        
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)
    
    
    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        # after CLS token, all before image patches
        x = torch.cat((
                x[:, :1, :],
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B, -1, -1)),
                x[:, 1:, :]
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x
    
    
    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens+self.num_vpt_token:]
        if self.out_type == 'featmap':
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            return self.ln2(patch_token.mean(dim=1))