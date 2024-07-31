# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch,mmengine
from mmdet.models.adapters.new_lorand import (LoRandv2,forward_sam_block_lorand,forward_dinov2_layer_lorand,
                        forward_eva02_layer_lorand,forward_swin_block_lorand)
import torch.nn as nn

@MODELS.register_module()
class ATSSLorand(SingleStageDetector):
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
                 lorand,
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

            
        self.num_branch = lorand['num_branch']    
        self.kernel_dim = lorand['kernel_dim']
        self.factor = lorand['factor']
        
        
        self.apply_lorand(self.backbone)
        self._set_lorand_trainable(self.backbone)
    
        
        
    def apply_lorand(self,tuning_module):

        
        for child_module in tuning_module.children():
            if type(child_module) == mmengine.model.base_module.ModuleList:
                for idx,layer in enumerate(child_module):
                                               
                    if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                        layer.adapter_lorand1 = LoRandv2(self.backbone_embed_dim,self.backbone_embed_dim//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(layer.norm1.weight.device)
                        layer.adapter_lorand2 =  LoRandv2(self.backbone_embed_dim,self.backbone_embed_dim//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(layer.norm1.weight.device)
                        # layer.s = self.scale
                        bound_method = forward_sam_block_lorand.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                    if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer' or 'VisionTransformer':
                        layer.adapter_lorand1 = LoRandv2(self.backbone_embed_dim,self.backbone_embed_dim//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(layer.norm1.weight.device)
                        layer.adapter_lorand2 =  LoRandv2(self.backbone_embed_dim,self.backbone_embed_dim//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(layer.norm1.weight.device)
                        # layer.s = self.scale
                        bound_method = forward_dinov2_layer_lorand.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                    if tuning_module.__class__.__name__ == 'ViTEVA02':
                        layer.adapter_lorand1 = LoRandv2(self.backbone_embed_dim,self.backbone_embed_dim//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(layer.norm1.weight.device)
                        layer.adapter_lorand2 =  LoRandv2(self.backbone_embed_dim,self.backbone_embed_dim//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(layer.norm1.weight.device)
                        # layer.s = self.scale
                        bound_method = forward_eva02_layer_lorand.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                if tuning_module.__class__.__name__ == 'SwinTransformer':
                    for seq_id,swin_block_sequence in enumerate(child_module):
                        for block_id,swin_block in enumerate(swin_block_sequence.blocks):
                            embed_dim = swin_block.norm1.normalized_shape[0]
                            swin_block.adapter_lorand1 = LoRandv2(embed_dim,embed_dim//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(swin_block.norm1.weight.device)
                            swin_block.adapter_lorand2 = LoRandv2(embed_dim,embed_dim//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(swin_block.norm1.weight.device)
                            bound_method = forward_swin_block_lorand.__get__(swin_block, swin_block.__class__)
                            setattr(swin_block, 'forward', bound_method)
                        
            
            
            
    def _set_lorand_trainable(self,tuning_module):
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
                
                
            # if name=='norm0.weight' or name=='norm0.bias' \
            #         or name=='norm1.weight' or name=='norm1.bias' \
            #         or name=='norm2.weight' or name=='norm2.bias' \
            #         or name=='norm3.weight' or name=='norm3.bias':
            #     param.requires_grad = True
            if 'norm' in name:
                param.requires_grad = True
                