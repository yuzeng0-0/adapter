# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from typing import List, Tuple, Union

from torch import Tensor
import torch.nn as nn
from mmdet.models.adapters.dense_prompt import forward_sam_block_dense_prompt,sam_transformer_forward_dense_prompt
import mmengine
from functools import reduce
from operator import mul
import math

@MODELS.register_module()
class ATSSDensePrompt(SingleStageDetector):
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
                 prompt_cfg,
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
            num_layer = len(self.backbone.blocks)
        else:
            self.backbone_embed_dim = self.backbone.embed_dims
            num_layer = len(self.backbone.layers)

        self.prompt_cfg = prompt_cfg
        self.num_prompt = prompt_cfg['num_prompt']
        self.dense_prompt = nn.Parameter(torch.zeros(num_layer,self.num_prompt,self.backbone_embed_dim))
        
        if prompt_cfg['init_type'] == 'vpt_like':
            val = math.sqrt(6. / float(3 * reduce(mul, self.backbone.patch_resolution, 1)+self.backbone_embed_dim))  # noqa
            nn.init.uniform_(self.dense_prompt.data, -val, val)
            
            
        
      
        
        
        self.apply_denseprompt(self.backbone)
        self._set_denseprompt_trainable(self.backbone)
        
        
    def apply_denseprompt(self,tuning_module):
        
        
        if self.backbone.__class__.__name__ == 'SAMImageEncoderViTv3':
            bound_method = sam_transformer_forward_dense_prompt.__get__(tuning_module, tuning_module.__class__)
            setattr(tuning_module, 'forward', bound_method)
            
        # if self.backbone.__class__.__name__ == 'Dinov2VisionTransformer':       
        #     bound_method = dinov2_transformer_forward_arc.__get__(tuning_module, tuning_module.__class__)
        #     setattr(tuning_module, 'forward', bound_method)
            
        # if self.backbone.__class__.__name__ == 'ViTEVA02':       
        #     bound_method = eva02_transformer_forward_arc.__get__(tuning_module, tuning_module.__class__)
        #     setattr(tuning_module, 'forward', bound_method)
        
        for child_module in tuning_module.children():
            if type(child_module) == mmengine.model.base_module.ModuleList:
                for idx,layer in enumerate(child_module):
                    if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                        
              
                        bound_method = forward_sam_block_dense_prompt.__get__(layer, layer.__class__)
                        setattr(layer, 'forward', bound_method)
                        
                        
                        
    def _set_denseprompt_trainable(self,tuning_module):
        
        for name, param in tuning_module.named_parameters():
            if '.dense_prompt' not in name:
                param.requires_grad = False    
                        
                        
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """

        x = self.backbone(batch_inputs,self.dense_prompt,self.prompt_cfg)
        if self.with_neck:
            x = self.neck(x)
        return x
