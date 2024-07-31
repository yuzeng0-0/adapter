# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch

from mmdet.models.adapters.ARC import (ARC_adapter,sam_transformer_forward_arc,forward_sam_block_arc,swin_layers_forward_arc,
                                       dinov2_transformer_forward_arc,forward_dinov2_layer_arc,forward_eva02_layer_arc,eva02_transformer_forward_arc,swin_transformer_forward_arc)
import torch.nn as nn
import mmengine

@MODELS.register_module()
class ATSS_ARC(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 arc_cfg,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        if self.backbone.__class__.__name__ == 'SAMImageEncoderViTv3':
            self.backbone_embed_dim = self.backbone.embed_dim
        elif self.backbone.__class__.__name__ == 'SwinTransformer':
            pass
        else:
            self.backbone_embed_dim = self.backbone.embed_dims
        
        self.tuning_mode = arc_cfg['tuning_mode']
        self.adapter_dim = arc_cfg['adapter_dim']
        self.adapter_dropout = arc_cfg['adapter_dropout']
        
        self.apply_arc(self.backbone)
        
        self._set_arc_trainable()
        
        
    def apply_arc(self,tuning_module):
        
        
        self.backbone.att_down_projection=None
        self.backbone.mlp_down_projection=None
        
        # if self.tuning_mode == 'ARC_att':
        #     self.backbone.att_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
        #     nn.init.xavier_uniform_(self.backbone.att_down_projection)
        # elif self.tuning_mode == 'ARC':
        #     self.backbone.att_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
        #     self.backbone.mlp_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
        #     nn.init.xavier_uniform_(self.backbone.mlp_down_projection)
        #     nn.init.xavier_uniform_(self.backbone.att_down_projection)
            
            
        if self.backbone.__class__.__name__ == 'SAMImageEncoderViTv3':
            
            if self.tuning_mode == 'ARC_att':
                self.backbone.att_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                nn.init.xavier_uniform_(self.backbone.att_down_projection)
            elif self.tuning_mode == 'ARC':
                self.backbone.att_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                self.backbone.mlp_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                nn.init.xavier_uniform_(self.backbone.mlp_down_projection)
                nn.init.xavier_uniform_(self.backbone.att_down_projection)
                
            bound_method = sam_transformer_forward_arc.__get__(tuning_module, tuning_module.__class__)
            setattr(tuning_module, 'forward', bound_method)
            
        if self.backbone.__class__.__name__ == 'Dinov2VisionTransformer':       
            if self.tuning_mode == 'ARC_att':
                self.backbone.att_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                nn.init.xavier_uniform_(self.backbone.att_down_projection)
            elif self.tuning_mode == 'ARC':
                self.backbone.att_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                self.backbone.mlp_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                nn.init.xavier_uniform_(self.backbone.mlp_down_projection)
                nn.init.xavier_uniform_(self.backbone.att_down_projection)
            bound_method = dinov2_transformer_forward_arc.__get__(tuning_module, tuning_module.__class__)
            setattr(tuning_module, 'forward', bound_method)
            
        if self.backbone.__class__.__name__ == 'ViTEVA02':    
            if self.tuning_mode == 'ARC_att':
                self.backbone.att_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                nn.init.xavier_uniform_(self.backbone.att_down_projection)
            elif self.tuning_mode == 'ARC':
                self.backbone.att_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                self.backbone.mlp_down_projection = nn.Parameter(torch.empty(self.backbone_embed_dim, self.adapter_dim))
                nn.init.xavier_uniform_(self.backbone.mlp_down_projection)
                nn.init.xavier_uniform_(self.backbone.att_down_projection)   
            bound_method = eva02_transformer_forward_arc.__get__(tuning_module, tuning_module.__class__)
            setattr(tuning_module, 'forward', bound_method)
            
        if self.backbone.__class__.__name__ == 'SwinTransformer': 
            
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for stage_id,swin_block_sequence in enumerate(child_module):
                        embed_dim = swin_block_sequence.blocks[0].norm1.normalized_shape[0]
                        # attn_name = "att_down_projection_stage_{}".format(stage_id)
                        swin_block_sequence.att_down_projection = nn.Parameter(torch.empty(embed_dim, self.adapter_dim))
                        swin_block_sequence.mlp_down_projection = nn.Parameter(torch.empty(embed_dim, self.adapter_dim))
                        nn.init.xavier_uniform_(swin_block_sequence.mlp_down_projection)
                        nn.init.xavier_uniform_(swin_block_sequence.att_down_projection)  
                
                        bound_method = swin_transformer_forward_arc.__get__(swin_block_sequence, swin_block_sequence.__class__)
                        setattr(swin_block_sequence, 'forward', bound_method)
                    
            
        for child_module in tuning_module.children():
            if type(child_module) == mmengine.model.base_module.ModuleList:
                for idx,layer in enumerate(child_module):
                    if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                        if self.tuning_mode == 'ARC_att':
                            layer.adapter_mlp = None
                            layer.adapter_attn = ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='attn').to(layer.norm1.weight.device)
                        elif self.tuning_mode == 'ARC':
                            layer.adapter_attn = ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='attn').to(layer.norm1.weight.device)
                            layer.adapter_mlp =  ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='mlp').to(layer.norm1.weight.device)
                            
                        bound_method = forward_sam_block_arc.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                    if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                        if self.tuning_mode == 'ARC_att':
                            layer.adapter_mlp = None
                            layer.adapter_attn = ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='attn').to(layer.norm1.weight.device)
                        elif self.tuning_mode == 'ARC':
                            layer.adapter_attn = ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='attn').to(layer.norm1.weight.device)
                            layer.adapter_mlp =  ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='mlp').to(layer.norm1.weight.device)
                            
                        bound_method = forward_dinov2_layer_arc.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                    if tuning_module.__class__.__name__ == 'ViTEVA02':
                        if self.tuning_mode == 'ARC_att':
                            layer.adapter_mlp = None
                            layer.adapter_attn = ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='attn').to(layer.norm1.weight.device)
                        elif self.tuning_mode == 'ARC':
                            layer.adapter_attn = ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='attn').to(layer.norm1.weight.device)
                            layer.adapter_mlp =  ARC_adapter(self.adapter_dim,self.backbone_embed_dim,self.adapter_dropout,position='mlp').to(layer.norm1.weight.device)
                            
                        bound_method = forward_eva02_layer_arc.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                    if tuning_module.__class__.__name__ == 'SwinTransformer':
                        for seq_id,swin_block_sequence in enumerate(child_module):
                            for block_id,swin_block in enumerate(swin_block_sequence.blocks):
                                embed_dim = swin_block.norm1.normalized_shape[0]
                                swin_block.adapter_attn = ARC_adapter(self.adapter_dim,embed_dim,self.adapter_dropout,position='attn').to(swin_block.norm1.weight.device)
                                swin_block.adapter_mlp =  ARC_adapter(self.adapter_dim,embed_dim,self.adapter_dropout,position='mlp').to(swin_block.norm1.weight.device)
                                bound_method = swin_layers_forward_arc.__get__(swin_block, swin_block.__class__)
                                setattr(swin_block, 'forward', bound_method)
                            
                        
                        
                        
    def _set_arc_trainable(self):
        
        for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                
        for name, param in self.backbone.named_parameters():
            if 'adapter' in name or  'att_down_projection' in name or 'mlp_down_projection' in name :
                param.requires_grad = True
                

    
                
                