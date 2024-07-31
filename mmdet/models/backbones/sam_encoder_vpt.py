
from .sam_encoderv3 import SAMImageEncoderViTv3,resize_pos_embed,MLPBlock,window_partition,window_unpartition,LayerNorm2d
from mmdet.registry import MODELS
import torch.nn as nn
import math
import torch
# from functions import reduce
from functools import partial
from functools import reduce
from operator import mul
from torch.nn.modules.utils import _pair
from typing import Optional, Tuple, Type
from mmengine.model import BaseModule, ModuleList
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
import torch.nn.functional as F
from copy import deepcopy

@MODELS.register_module()
class SAMImageEncoderViTv3VPT(BaseModule):
    def __init__(
        self,
        prompt_config = None,
        bias_tune:bool = False,
        pos_tune:bool = False,
        pre_norm:bool = False,
        is_resize_pos_embed:bool = False,
        frozen_stages: int = 0,
        model_type: str = 'vit_b',
        checkpoint: str = None,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        init_cfg=None
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
        """
        super(SAMImageEncoderViTv3VPT,self).__init__(init_cfg)
        
        assert model_type in ['vit_h','vit_l','vit_b']
        if model_type == 'vit_h':
            embed_dim=1280
            depth=32
            num_heads=16
            global_attn_indexes=[7, 15, 23, 31]
         
        elif model_type == 'vit_l':
            embed_dim=1024
            depth=24
            num_heads=16
            global_attn_indexes=[5, 11, 17, 23]
        else:
            embed_dim=768
            depth=12
            num_heads=12
            global_attn_indexes=[2, 5, 8, 11]

        
        self.img_size = img_size
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)
        self.embed_dim = embed_dim
        self.is_resize_pos_embed = is_resize_pos_embed
        
        
        # VPT init
        self.prompt_config = prompt_config
        self.num_vpt_token = self.prompt_config['num_token']
        self.prompt_dropout =  nn.Dropout(self.prompt_config.dropout)

        
        
        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_chans,
            input_size=img_size,
            embed_dims=self.embed_dim,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            bias=not pre_norm,  # disable bias if pre_norm is used(e.g., CLIP)
        )
        _patch_cfg.update(_patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
         
        # self.patch_embed = PatchEmbed(
        #     kernel_size=(patch_size, patch_size),
        #     stride=(patch_size, patch_size),
        #     in_chans=in_chans,
        #     embed_dim=embed_dim,
        # )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = ModuleList()
        for i in range(depth):
            block = VPTBlock(
                num_vpt_token = self.num_vpt_token,
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
        
                
        
        # if project the prompt embeddings
        if self.prompt_config['project'] > -1:
            # only for prepend / add
            prompt_dim = self.prompt_config['project']
            self.prompt_proj = nn.Linear(
                prompt_dim,  self.embed_dim)
            nn.init.kaiming_normal_(
                self.prompt_proj.weight, a=0, mode='fan_out')
        else:
            prompt_dim =  self.embed_dim
            self.prompt_proj = nn.Identity()
            
        # initiate prompt:
        if self.prompt_config['initiation_type'] == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, _pair(patch_size), 1) + prompt_dim))  # noqa

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
        
        
        
        
        
        self.frozen_stages = frozen_stages
        self.pos_tune = pos_tune
        self.bias_tune = bias_tune
        if self.frozen_stages > 0:
            self._freeze_stages()
            
            
        self.checkpoint = checkpoint
        if self.checkpoint is not None:
            self.load_encoder_checkpoints(checkpoint) 
            
        
        
    def load_encoder_checkpoints(self, ckpt_path):
        
        with open(ckpt_path, "rb") as f:
            state_dict = torch.load(f)
            
        prefix = 'image_encoder'
        image_encoder_state_dict = {}
        
        for k in state_dict.keys():
            if k.startswith(prefix):
                image_encoder_state_dict[k[14:]] = state_dict[k]
                
        self.load_state_dict(image_encoder_state_dict)
        
        
    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None and self.pos_tune==False:
            self.pos_embed.requires_grad = False

        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        
        for param in self.neck.parameters():
            param.requires_grad = False
        
        if self.pos_tune:
            for i in range(1, self.frozen_stages + 1):
                m = self.blocks[i - 1]
                for name, param in m.named_parameters():
                    if 'rel_pos' in name:
                        param.requires_grad = True
                        
        if self.bias_tune:
            for name,param in self.named_parameters():
                if 'bias' in name:
                    param.requires_grad = True
            
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x,patch_resolution = self.patch_embed(x)
            
        if self.pos_embed is not None:
            x = x + resize_pos_embed(
                                self.pos_embed.reshape(1,-1,768),
                                self.patch_resolution,
                                patch_resolution,
                                mode='bicubic',
                                num_extra_tokens=0)
        # add extra vision prompt
        x = self.incorporate_prompt(x)
        
        if self.prompt_config['type']=='deep':
            
            x = self.forward_deep_prompt(x,patch_resolution)
            
        else:
            for blk in self.blocks:
                x = blk(x,patch_resolution)

            x = x[:, self.num_vpt_token:,:]
            x = x.view(B,patch_resolution[0],patch_resolution[1],-1)
            x = self.neck(x.permute(0, 3, 1, 2))

        return (x,)
    
    
    
    def incorporate_prompt(self, x):
        # combine prompt embeddings with image-patch embeddings
        B = x.shape[0]
        x = torch.cat((
                self.prompt_dropout(self.prompt_proj(self.prompt_embeddings).expand(B,-1,-1)),
                x
            ), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)

        return x
    
    def forward_deep_prompt(self,x,patch_resolution):
        B,L,C = x.shape
        
        for i, blk in enumerate(self.blocks):
            
            if i==0:
            
                x = blk(x,patch_resolution)
            else:
            
                if i<= self.deep_prompt_embeddings.shape[0]:
                        deep_prompt_emb = self.prompt_dropout(self.prompt_proj(
                        self.deep_prompt_embeddings[i-1]).expand(B, -1, -1))
                        
                        x = torch.cat((
                            deep_prompt_emb,
                            x[:, self.num_vpt_token:, :]), dim=1)
                        
                        x = blk(x,patch_resolution)

        x = x[:, self.num_vpt_token:,:]
        x = x.view(B,patch_resolution[0],patch_resolution[1],-1)
        x = self.neck(x.permute(0, 3, 1, 2))
                
        return x
        
    
    
class VPTBlock(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        num_vpt_token:int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VPTAttention(
            # num_vpt_token=0 if window_size==0 else num_vpt_token,
            num_vpt_token=num_vpt_token,
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )

        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio), act=act_layer)

        self.window_size = window_size
        self.num_vpt_token = num_vpt_token

    def forward(self, x: torch.Tensor,patch_resolution) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        
        
        # befor window partition,change the input size
        H, W = patch_resolution
        B, L, C = x.shape
        # prompt_emb = x[:, :self.num_vpt_token, :]
        # x = x[:, self.num_vpt_token:, :]
        L = L - self.num_vpt_token
        assert L == H * W, "input feature has wrong size, should be {}, got {}".format(H*W, L)
        
        # x = x.view(B, H, W, C)
        
        # Window partition
        if self.window_size > 0:
            prompt_emb = x[:, :self.num_vpt_token, :]
            x = x[:, self.num_vpt_token:, :]
            x = x.view(B, H, W, C)
            
            
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
            # num_window*B,window_size*window_size,c
            x = x.view(-1, self.window_size * self.window_size, C)
            
            # expand prompt_emb
            num_windows = int(x.shape[0] / B)
            prompt_emb = prompt_emb.unsqueeze(0)
            prompt_emb = prompt_emb.expand(num_windows, -1, -1, -1)
            prompt_emb = prompt_emb.reshape((-1, self.num_vpt_token, C))
            x = torch.cat((prompt_emb, x), dim=1)

        x = self.attn(x,patch_resolution,self.window_size)
        
        # Reverse window partition
        if self.window_size > 0:
            
            
            prompt_emb = x[:, :self.num_vpt_token, :]
            x = x[:, self.num_vpt_token:, :]
            prompt_emb = prompt_emb.view(-1, B, self.num_vpt_token, C)
            prompt_emb = prompt_emb.mean(0)
            
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
            x = x.view(B,-1,C)
            x = torch.cat((prompt_emb, x), dim=1)
            
            

        x = shortcut + x
 
        x = x + self.mlp(self.norm2(x))

        return x
    
    
    
class VPTAttention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        num_vpt_token,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.num_vpt_token = num_vpt_token
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.input_size = input_size
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor,patch_resolution,window_size) -> torch.Tensor:

        B, L, C = x.shape
        H,W = patch_resolution
        
        # B, L, C = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, L, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            if window_size>0:
                attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (self.input_size[0],self.input_size[1]), (self.input_size[0],self.input_size[1]),self.num_vpt_token)
            else:
                attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H,W), (H,W),self.num_vpt_token)
            #TODO LLaMA adapter那样分开softmax
            attn = attn.softmax(dim=-1)
            # bug ???????
            x = (attn @ v).view(B, self.num_heads, L, -1).permute(0,2,1,3).reshape(B,L,-1)
            x = self.proj(x)
            
  
                
        # #TODO LLaMA adapter那样分开softmax
        # attn = attn.softmax(dim=-1)
        # x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        # x = self.proj(x)

        return x
    
    
def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
    num_vpt_token:int,
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    
    q = q[:,num_vpt_token:,:]
    
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)


    ori_attn = attn[:,num_vpt_token:,num_vpt_token:]
    ori_attn = (
        ori_attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)
    
    attn[:,num_vpt_token:,num_vpt_token:] = ori_attn
    
    return attn

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]