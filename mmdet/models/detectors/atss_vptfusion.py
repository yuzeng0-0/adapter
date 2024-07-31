# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from typing import List, Tuple, Union

from torch import Tensor
import torch.nn as nn

@MODELS.register_module()
class ATSSVPTFusion(SingleStageDetector):
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
                 conv_inplane,
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
        
        # 删掉原本在backbone中初始化的prompt_embeddings
        if self.backbone.prompt_config is not None:
            del self.backbone.prompt_embeddings      
            if  self.backbone.prompt_config['type']=='deep':
                del self.deep_prompt_embeddings

            
        if self.backbone.__class__.__name__ == 'SAMImageEncoderViTv3VPT':
            self.spm = SpatialPriorModule(inplanes=conv_inplane,embed_dim=self.backbone.embed_dim)
        else:
            self.spm = SpatialPriorModule(inplanes=conv_inplane,embed_dim=self.backbone.embed_dims)

        self.gen_vpt_token = nn.Linear(1, self.backbone.num_vpt_token)
            
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        
        
        # cnn_x = self.cnn_backbone(batch_inputs)
        # cnn_feat = self.convs(cnn_x[2])
        # B,C,_,_ = cnn_feat.shape
        # cnn_feat = cnn_feat.view(B,C,-1).mean(dim=-1)
        # cnn_prompt = self.extend_linear(cnn_feat).view(B,self.backbone.num_vpt_token,-1)
        
        # self.backbone.prompt_embeddings.data = self.backbone.prompt_embeddings.data.expand(B,self.backbone.num_vpt_token,C)
        
        # self.backbone.prompt_embeddings.data = cnn_prompt
        
        
        c1, c2, c3 = self.spm(batch_inputs)
        c3 = c3.mean(dim=1).unsqueeze(1).permute(0,2,1)
        self.backbone.prompt_embeddings = self.gen_vpt_token(c3).permute(0,2,1)
        
        x = self.backbone(batch_inputs)


        if self.with_neck:
            x = self.neck(x)
        return x





class SpatialPriorModule(nn.Module):
    def __init__(self, inplanes=64, embed_dim=384):
        super().__init__()

        self.stem = nn.Sequential(*[
            nn.Conv2d(3, inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.SyncBatchNorm(inplanes),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            # nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inplanes),
            # nn.SyncBatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.conv2 = nn.Sequential(*[
            nn.Conv2d(inplanes, 2 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.SyncBatchNorm(2 * inplanes),
            nn.BatchNorm2d(2*inplanes),
            nn.ReLU(inplace=True)
        ])
        self.conv3 = nn.Sequential(*[
            nn.Conv2d(2 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
            # nn.SyncBatchNorm(4 * inplanes),
            nn.BatchNorm2d(4*inplanes),
            nn.ReLU(inplace=True)
        ])
        # self.conv4 = nn.Sequential(*[
        #     nn.Conv2d(4 * inplanes, 4 * inplanes, kernel_size=3, stride=2, padding=1, bias=False),
        #     # nn.SyncBatchNorm(4 * inplanes),
        #     nn.BatchNorm2d(4*inplanes),
        #     nn.ReLU(inplace=True)
        # ])
        self.fc1 = nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        # self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        # c4 = self.conv4(c3)
        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        # c4 = self.fc4(c4)

        bs, dim, _, _ = c1.shape
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # 4s
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        # c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s

        return c1, c2, c3
