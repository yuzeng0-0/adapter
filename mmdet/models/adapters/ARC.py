import torch.nn as nn

import torch

from mmdet.models.backbones.sam_encoderv3 import window_partition,window_unpartition
from mmdet.models.utils import resize_pos_embed


class ARC_adapter(nn.Module):
    def __init__(self, adapter_dim, hidden_dim, dropout=0.0, position='att'):
        super(ARC_adapter, self).__init__()
        self.adapter_rescale = nn.Parameter(torch.empty(1, adapter_dim))
        self.adapter_bias = nn.Parameter(torch.empty(hidden_dim))
        self.dropout = nn.Dropout(dropout)

        if position == 'att':
            nn.init.zeros_(self.adapter_rescale)
        else:
            nn.init.xavier_uniform_(self.adapter_rescale)
        nn.init.zeros_(self.adapter_bias)

    def forward(self, x, down_projection, up_projection):
        adapter_output = torch.matmul(x, down_projection * self.adapter_rescale)
        adapter_output = self.dropout(adapter_output)
        adapter_output = torch.matmul(adapter_output, up_projection) + self.adapter_bias
        output = adapter_output + x

        return output
    
    


def sam_transformer_forward_arc(self, x: torch.Tensor) -> torch.Tensor:
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
    
    for blk in self.blocks:
        x = blk(x,self.att_down_projection,self.mlp_down_projection)

    x = self.neck(x.permute(0, 3, 1, 2))

    return (x,)


def forward_sam_block_arc(self,x,att_down_projection,mlp_down_projection):

    B,H, W,C = x.shape
    shortcut = x
    x = self.norm1(x)
    
    if self.adapter_attn is not None and att_down_projection is not None:
        x = x.view(B,H*W,C)
        x = self.adapter_attn(x, att_down_projection, att_down_projection.t())    
        x = x.view(B,H,W,C)
    
    # Window partition
    if self.window_size > 0:
        H, W = x.shape[1], x.shape[2]
        x, pad_hw = window_partition(x, self.window_size)

    x = self.attn(x)
    # Reverse window partition
    if self.window_size > 0:
        x = window_unpartition(x, self.window_size, pad_hw, (H, W))

    x = shortcut + x
    
    
    res_ffn_feat = x
    x = self.norm2(x)
    
    if self.adapter_mlp is not None and mlp_down_projection is not None:
        x = x.view(B,H*W,C)
        x = self.adapter_mlp(x, mlp_down_projection, mlp_down_projection.t())
        x = x.view(B,H,W,C)
        
    x = self.mlp(x)

    x = x + res_ffn_feat
    
    return x




def dinov2_transformer_forward_arc(self, x):
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
    x = self.drop_after_pos(x)

    x = self.pre_norm(x)

    outs = []
    for i, layer in enumerate(self.layers):
        
        x = layer(x,self.att_down_projection,self.mlp_down_projection)
        
        if i == len(self.layers) - 1 and self.final_norm:
            x = self.ln1(x)

        if i in self.out_indices:
            outs.append(self._format_output(x, patch_resolution))

    return tuple(outs)


def forward_dinov2_layer_arc(self,x,att_down_projection,mlp_down_projection):
    
    shortcut = x
    x = self.ln1(x)
    if self.adapter_attn is not None and att_down_projection is not None:
        x = self.adapter_attn(x, att_down_projection, att_down_projection.t())    
    
    x = shortcut + self.attn(x)
    res_ffn_feat = x
    
    x = self.ln2(x)
    
    if self.adapter_mlp is not None and mlp_down_projection is not None:
        x = self.adapter_mlp(x, mlp_down_projection, mlp_down_projection.t())
    
    x = self.ffn(x, identity=res_ffn_feat)
    
    
    return x



def eva02_transformer_forward_arc(self, x):
    B = x.shape[0]
    x, patch_resolution = self.patch_embed(x)

    if self.cls_token is not None:
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

    x = x + resize_pos_embed(
        self.pos_embed,
        self.patch_resolution,
        patch_resolution,
        mode=self.interpolate_mode,
        num_extra_tokens=self.num_extra_tokens)
    x = self.drop_after_pos(x)

    x = self.pre_norm(x)

    outs = []
    for i, layer in enumerate(self.layers):
        x = layer(x, patch_resolution,self.att_down_projection,self.mlp_down_projection)

        if i == len(self.layers) - 1 and self.final_norm:
            x = self.ln1(x)

        if i in self.out_indices:
            outs.append(self._format_output(x, patch_resolution))

    return tuple(outs)



def forward_eva02_layer_arc(self,x,patch_resolution,att_down_projection,mlp_down_projection):
    
    inputs = x
    x = self.norm1(x)
    if self.adapter_attn is not None and att_down_projection is not None:
        x = self.adapter_attn(x, att_down_projection, att_down_projection.t())    
    
    x = self.attn(x, patch_resolution)
    x = self.drop_path(x)
    x = inputs + x

    inputs = x
    x = self.norm2(x)
    
    if self.adapter_mlp is not None and mlp_down_projection is not None:
        x = self.adapter_mlp(x, mlp_down_projection, mlp_down_projection.t())
    
    x = self.mlp(x)
    x = self.drop_path(x)
    x = inputs + x

    return x


def swin_transformer_forward_arc(self, x, hw_shape):
    for block in self.blocks:
        x = block(x, hw_shape,self.att_down_projection,self.mlp_down_projection)

    if self.downsample:
        x_down, down_hw_shape = self.downsample(x, hw_shape)
        return x_down, down_hw_shape, x, hw_shape
    else:
        return x, hw_shape, x, hw_shape



def swin_layers_forward_arc(self, x,hw_shape,att_down_projection,mlp_down_projection):


    def _inner_forward(x,att_down_projection,mlp_down_projection):
        
        identity = x
        x = self.norm1(x)
        
        if self.adapter_attn is not None and att_down_projection is not None:
            x = self.adapter_attn(x, att_down_projection, att_down_projection.t())   
            
        x = self.attn(x, hw_shape)

        x = x + identity

        identity = x
        x = self.norm2(x)
        
        if self.adapter_mlp is not None and mlp_down_projection is not None:
            x = self.adapter_mlp(x, mlp_down_projection, mlp_down_projection.t())
        
        x = self.ffn(x, identity=identity)

        return x

    if self.with_cp and x.requires_grad:
        x = cp.checkpoint(_inner_forward, x)
    else:
        x = _inner_forward(x,att_down_projection,mlp_down_projection)

    return x

    
    
    