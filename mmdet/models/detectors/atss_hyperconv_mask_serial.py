from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
import re
import mmengine
from mmengine.runner.checkpoint import CheckpointLoader
from mmdet.models.adapters.hypernet_convpass_mask_serial import (HyperNetwork_conv,Convpass_swin_hypernet_mask_serial,
                                                        Convpass_hypernet_mask_serial)
from mmdet.models.adapters.hypernet_convpass_mask_serial import (forward_dinov2_block_serial_V1,forward_dinov2_block_serial_V2,forward_eva02_block_serial_V1,
                                                                 forward_eva02_block_serial_V2,forward_sam_block_serial_V1,forward_sam_block_serial_V2)

import torch.nn as nn

@MODELS.register_module()
class ATSSHyperConvPass_mask_serial(SingleStageDetector):
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
                 convpass,
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
        else:
            self.backbone_embed_dim = self.backbone.embed_dims
            
        self.method = convpass['method']
        
        self.dim = convpass['dim']
        self.scale = convpass['scale']
        self.xavier_init = convpass['xavier_init']
        
        self.init_type = convpass['init_type']
        self.layer_embedding_size = convpass['layer_embedding_size']
        self.num_mask = convpass['num_mask']
        
        self.serial_pos = convpass['serial_pos']
        
        self.adapter_mask_token = nn.Embedding(self.num_mask,self.layer_embedding_size)
        
        
        
    
        self.apply_convpass(self.backbone)
        self._set_convpass_trainable(self.backbone)
            
            
    def apply_convpass(self,tuning_module):
        
        # self.adapter_conv_hypetnet = HyperNetwork_conv(f_size=3,in_size=self.backbone_embed_dim,out_size=self.dim)
        
        if self.method == 'attn+mlp':

                        
            self.adapter_attn_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim).cuda()
            self.adapter_mlp_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim).cuda()

            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for idx,layer in enumerate(child_module):
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin_hypernet_mask_serial(self.adapter_attn_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_swin_hypernet_mask_serial(self.adapter_mlp_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            if self.serial_pos=='after_ffn':
                                bound_method = forward_sam_block_serial_V2.__get__(layer, layer.__class__)
                            else:
                                bound_method = forward_sam_block_serial_V1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            
                            layer.adapter_attn = Convpass_hypernet_mask_serial(self.adapter_attn_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_hypernet_mask_serial(self.adapter_mlp_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            if self.serial_pos=='after_ffn':
                                bound_method = forward_dinov2_block_serial_V2.__get__(layer, layer.__class__)
                            else:
                                bound_method = forward_dinov2_block_serial_V1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            
                            layer.adapter_attn = Convpass_hypernet_mask_serial(self.adapter_attn_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_hypernet_mask_serial(self.adapter_mlp_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            if self.serial_pos=='after_ffn':
                                bound_method = forward_eva02_block_serial_V2.__get__(layer, layer.__class__)
                            else:
                                bound_method = forward_eva02_block_serial_V1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)


              

                        
    def _set_convpass_trainable(self,tuning_module):
        
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
                
        if self.backbone.bias_tune:
            for name, param in tuning_module.named_parameters():
                if '.bias' in name:
                    param.requires_grad = True
        if self.backbone.norm_tune:
            for name, param in tuning_module.named_parameters():
                if '.norm' in name:
                    param.requires_grad = True