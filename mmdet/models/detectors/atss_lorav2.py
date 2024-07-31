# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
from torch import nn
import re
from mmdet.models.adapters import LoRALinear
from mmengine.runner.checkpoint import CheckpointLoader
import mmengine

from mmdet.models.adapters.lorav2 import LoRALinearv2,forward_dinov2_block_type1,forward_eva02_block_type1,forward_sam_block_type1

@MODELS.register_module()
class ATSSLorav2(SingleStageDetector):
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
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 lora:dict,
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
        
        
        
        # ------------------------  Tuning -------------------------
   
        if lora is not None:
            
            self.lora_type = lora['lora_type']
            self.lora_cfg = lora['lora_cfg']
            # self.rank = lora['rank']
            # self.alpha = lora['alpha']
            # self.drop_rate = lora['drop_rate']
            
            self.apply_lora(self.backbone)
            self._set_lora_trainable(self.backbone)
            
    def apply_lora(self,tuning_module):
        
        for child_module in tuning_module.children():
            if type(child_module) == mmengine.model.base_module.ModuleList:
                for layer in child_module:
                    if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                        layer.loralinear = LoRALinearv2(tuning_module.embed_dim,**self.lora_cfg).to(layer.norm1.weight.device)
                        bound_method = forward_sam_block_type1.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                    
                    if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                        layer.loralinear = LoRALinearv2(tuning_module.embed_dims,**self.lora_cfg).to(layer.norm1.weight.device)
                        bound_method = forward_dinov2_block_type1.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                                            
                    if tuning_module.__class__.__name__ == 'ViTEVA02':
                        layer.loralinear = LoRALinearv2(tuning_module.embed_dims,**self.lora_cfg).to(layer.norm1.weight.device)
                        bound_method = forward_eva02_block_type1.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
        
    def _set_lora_trainable(self,tuning_module):
        """Set only the lora parameters trainable."""
        for name, param in tuning_module.named_parameters():
            if '.lora_' not in name:
                param.requires_grad = False
                
        if self.backbone.bias_tune:
            for name, param in tuning_module.named_parameters():
                if '.bias' in name:
                    param.requires_grad = True

            
    

        
        