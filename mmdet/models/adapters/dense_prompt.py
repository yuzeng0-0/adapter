import torch.nn as nn

import torch

from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition
from mmdet.models.utils import resize_pos_embed

def sam_transformer_forward_dense_prompt(self, x: torch.Tensor,dense_prompt,prompt_cfg) -> torch.Tensor:
    B = x.shape[0]
    x,patch_resolution = self.patch_embed(x)
    
    if self.pos_embed is not None:
        x = x + resize_pos_embed(
                            self.pos_embed.reshape(1,-1,768),
                            self.patch_resolution,
                            patch_resolution,
                            mode='bicubic',
                            num_extra_tokens=0)
    x = x.reshape(B,patch_resolution[0],patch_resolution[1],768)
    
    for idx in range(len(self.blocks)):
        x = self.blocks[idx](x,dense_prompt[idx],prompt_cfg)


    x = self.neck(x.permute(0, 3, 1, 2))

    return (x,)


def forward_sam_block_dense_prompt(self,x,dense_prompt,prompt_cfg):
    
    assert prompt_cfg['pos'] == 'before_all' or prompt_cfg['pos'] == 'after_all'
    
    if prompt_cfg['pos'] == 'before_all':
        prompt_feat = torch.matmul(x,dense_prompt.t())
        prompt_feat = prompt_feat.mean(-1).unsqueeze(-1)
        x = x + prompt_feat


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
    
    x = x + self.mlp(self.norm2(x))
    
    if prompt_cfg['pos'] == 'after_all':
        prompt_feat = torch.matmul(x,dense_prompt.t())
        prompt_feat = prompt_feat.mean(-1).unsqueeze(-1)
        x = x + prompt_feat
    
    return x
