# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmdet.models.adapters.adapter_former import AdapterFormer
from mmdet.models.adapters.base_adapter import inject_adapter
import re
from mmengine.logging import print_log
from torch import nn
from copy import deepcopy
import mmengine

from mmdet.models.adapters.adapter_formerv2 import AdapterFormerv2,forward_dinov2_block,forward_eva02_block,forward_sam_block


@MODELS.register_module()
class ATSSAdapterFormerv2(SingleStageDetector):
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
                 train_cfg: OptConfigType = None,
                 adapter_cfg:dict = None,
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
        

        if adapter_cfg is not None:
            self.adapter_former = True
            self.adapter_cfg = adapter_cfg
            # self.adapter_scalar = adapter_cfg['adapter_scalar']
            # self.dropout = adapter_cfg['dropout']
            # self.down_size = adapter_cfg['down_size']
                        
            # self.targets = adapter_cfg['targets']
            self.apply_adapter_former(self.backbone)
            self._set_adapter_trainable(self.backbone)
            
            
    def apply_adapter_former(self,tuning_module):

        for child_module in tuning_module.children():
            if type(child_module) == mmengine.model.base_module.ModuleList:
                for layer in child_module:
                    if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                        layer.adapter_mlp = AdapterFormerv2(tuning_module.embed_dim,**self.adapter_cfg).to(layer.norm1.weight.device)
                        # layer.s = self.adapter_cfg['adapter_scale']
                        bound_method = forward_sam_block.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                    
                    if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                        layer.adapter_mlp = AdapterFormerv2(tuning_module.embed_dims,**self.adapter_cfg).to(layer.norm1.weight.device)
                        # layer.s = self.adapter_cfg['adapter_scale']
                        bound_method = forward_dinov2_block.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                                            
                    if tuning_module.__class__.__name__ == 'ViTEVA02':
                        layer.adapter_mlp = AdapterFormerv2(tuning_module.embed_dims,**self.adapter_cfg).to(layer.norm1.weight.device)
                        # layer.s = self.adapter_cfg['adapter_scale']
                        bound_method = forward_eva02_block.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                               
    def _set_adapter_trainable(self,tuning_module):
        """Set only the lora parameters trainable."""
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
   
        if self.backbone.bias_tune:
            for name, param in tuning_module.named_parameters():
                if '.bias' in name:
                    param.requires_grad = True