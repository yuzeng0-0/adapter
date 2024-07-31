# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
from mmdet.models.adapters.hypernet_convpass_mask import (HyperNetwork_conv,Convpass_swin_hypernet_mask,SingleHyperNetwork_conv_lorand,HyperNetwork_conv_lorandv1,HyperNetwork_conv_lorandv2,
                                                        Convpass_hypernet_mask,SingleHyperNetwork_conv,SingleHyperNetwork_conv_lora,SingleHyperNetwork_conv_polyhistor)
from mmdet.models.adapters.hypernet_convpass_mask import forward_sam_block_t1,forward_dinov2_block_t1,forward_eva02_block_t1,forward_swin_block_t1,Convpass_swintransformer_hypernet_mask
import torch.nn as nn
import mmengine



@MODELS.register_module()
class CascadeRCNN_hyperconvmask(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 convpass,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)


        self.method = convpass['method']
        
        self.dim = convpass['dim']
        self.scale = convpass['scale']
        self.xavier_init = convpass['xavier_init']
        
        self.init_type = convpass['init_type']
        self.layer_embedding_size = convpass['layer_embedding_size']
        self.num_mask = convpass['num_mask']
        self.hypernet_low_dim = convpass['hypernet_low_dim']

        
        self.adapter_mask_token = nn.Embedding(self.num_mask,self.layer_embedding_size)
        
        self.apply_convpass(self.backbone)
        self._set_convpass_trainable(self.backbone)
            
            
    def apply_convpass(self,tuning_module):
        
        # self.adapter_conv_hypetnet = HyperNetwork_conv(f_size=3,in_size=self.backbone_embed_dim,out_size=self.dim)
        
        if self.method == 'attn+mlp':
            
            self.adapter_attn_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type,low_dim=self.hypernet_low_dim).cuda()
            self.adapter_mlp_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type,low_dim=self.hypernet_low_dim).cuda()

            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for idx,layer in enumerate(child_module):
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin_hypernet_mask(self.adapter_attn_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_swin_hypernet_mask(self.adapter_mlp_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            
                            layer.adapter_attn = Convpass_hypernet_mask(self.adapter_attn_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_hypernet_mask(self.adapter_mlp_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            
                            layer.adapter_attn = Convpass_hypernet_mask(self.adapter_attn_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_hypernet_mask(self.adapter_mlp_conv_hypetnet,self.adapter_mask_token,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                            
                    if tuning_module.__class__.__name__ == 'SwinTransformer':
                        for seq_id,swin_block_sequence in enumerate(child_module):
                            for block_id,swin_block in enumerate(swin_block_sequence.blocks):
                                embed_dim = swin_block.norm1.normalized_shape[0]
                                swin_block.adapter_attn = Convpass_swintransformer_hypernet_mask(self.adapter_attn_conv_hypetnet,self.adapter_mask_token,embed_dim,self.dim).to(swin_block.norm1.weight.device)
                                swin_block.adapter_mlp =  Convpass_swintransformer_hypernet_mask(self.adapter_mlp_conv_hypetnet,self.adapter_mask_token,embed_dim,self.dim).to(swin_block.norm1.weight.device)
                                swin_block.s = self.scale
                                bound_method = forward_swin_block_t1.__get__(swin_block, swin_block.__class__)
                                setattr(swin_block, 'forward', bound_method)
                            
                        

              

                        
    def _set_convpass_trainable(self,tuning_module):
        
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
                
        # if self.backbone.bias_tune:
        #     for name, param in tuning_module.named_parameters():
        #         if '.bias' in name:
        #             param.requires_grad = True
        # if self.backbone.norm_tune:
        #     for name, param in tuning_module.named_parameters():
        #         if '.norm' in name:
        #             param.requires_grad = True
