# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from typing import List, Tuple, Union

from torch import Tensor
import torch.nn as nn

@MODELS.register_module()
class ATSSFusion(SingleStageDetector):
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
                 cnn_backbone:ConfigType,
                 cnn_backbone_pretrain_path:str,
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
        
        self.cnn_backbone = MODELS.build(cnn_backbone)
        self.cnn_backbone_pretrain_path = cnn_backbone_pretrain_path
        
        self.load_cnn_state_dict()
        if type(self.backbone).__name__=='SAMImageEncoderViTv3':
            self.convs = nn.Conv2d(1024, self.backbone.embed_dim, 1, 1, 0, bias=True)
        else:
            self.convs = nn.Conv2d(1024, self.backbone.embed_dims, 1, 1, 0, bias=True)
        

    def load_cnn_state_dict(self):
         with open(self.cnn_backbone_pretrain_path , "rb") as f:
            state_dict = torch.load(f)
            self.cnn_backbone.load_state_dict(state_dict , False)
            
            
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        cnn_x = self.cnn_backbone(batch_inputs)
        cnn_feat = self.convs(cnn_x[2])
        fuse_feat = x[0] + cnn_feat
        fuse_feat = (fuse_feat,)
        if self.with_neck:
            fuse_feat = self.neck(fuse_feat)
        return fuse_feat