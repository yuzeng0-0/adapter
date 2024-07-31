# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, Optional, Union,Tuple

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType
from .kd_one_stagev2 import KnowledgeDistillationSingleStageDetectorV2
from mmengine.dataset import Compose
from mmengine.optim import build_optim_wrapper
from copy import deepcopy
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_flip
from mmcv.ops import batched_nms
from .base import BaseDetector
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig


@MODELS.register_module()
class Tent(BaseDetector):
    r"""Implementation of `Distilling the Knowledge in a Neural Network.
    <https://arxiv.org/abs/1503.02531>`_.

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        teacher_config (:obj:`ConfigDict` | dict | str | Path): Config file
            path or the config object of teacher model.
        teacher_ckpt (str, optional): Checkpoint path of teacher model.
            If left as None, the model will not load any weights.
            Defaults to True.
        eval_teacher (bool): Set the train mode for teacher.
            Defaults to True.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
    """

    def __init__(
        self,
        model_config: Union[ConfigType, str, Path],
        ckpt: Optional[str] = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        entropy_type = None,
        optim_wrapper = None,
        tta_cfg = None,
        episodic = True,
        adapt_all = None,
        **kwags,
    ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        if isinstance(model_config, (str, Path)):
            model_config = Config.fromfile(model_config)
        self.model = MODELS.build(model_config)
        
        
        # build optimizer
        self.optim_wrapper = None
        if optim_wrapper is not None:
            self.optim_wrapper_cfg = optim_wrapper

        self.episodic = episodic
        self.tta_cfg = tta_cfg  
        self.ckpt = ckpt
        self.adapt_all = adapt_all
        self.configure_model()
        self._init_source_model_state()
        assert entropy_type=="softmax" or entropy_type=="sigmoid"
        self.entropy_type = entropy_type
        
        # TODO: implement param_scheduler for optimizer (e.g. lr decay)
        
        
    def _init_source_model_state(self) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """
        
        if self.ckpt is not None:
            # NOTE we use the teacher model to initialize the next stage for both teacher and student

            # load_checkpoint(
            #         self.model, self.ckpt, map_location='cpu', revise_keys=[(r'^teacher\.', '')])
            load_checkpoint(
                    self.model, self.ckpt, map_location='cpu')
                
        self.source_model_state = deepcopy(self.model.state_dict())
   
        self.optim_wrapper = build_optim_wrapper(self.model, self.optim_wrapper_cfg)
        self.optim_wrapper_state = deepcopy(self.optim_wrapper.state_dict())
        self.configure_model()
        
        
    def configure_model(self):
        """Configure model for use with tent."""
        # train mode, because tent optimizes the model to minimize entropy
        self.model.train()
        # disable grad, to (re-)enable only what tent updates
        self.model.requires_grad_(False)
        # configure norm for tent updates: enable grad + force batch statisics
        
        if self.adapt_all:
            for name, param in self.model.named_parameters():
                if "norm" in name or "bn" in name:     
                # if ("norm" in name or "bn" in name) and "backbone"  not in name:     
                    param.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    param.track_running_stats = False
                    param.running_mean = None
                    param.running_var = None
        else:
              for name, param in self.model.named_parameters(): 
                if ("norm" in name or "bn" in name) and "backbone"  not in name:     
                    param.requires_grad_(True)
                    # force use of batch stats in train and eval modes
                    param.track_running_stats = False
                    param.running_mean = None
                    param.running_var = None
            


        
        
    def reset_model(self):
        
        if self.source_model_state is None:
            raise Exception("cannot reset without saved model state")
        self.model.load_state_dict(self.source_model_state, strict=True)
        self.optim_wrapper.load_state_dict(self.optim_wrapper_state)
        


    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.model.extract_feat(batch_inputs)
        # with torch.no_grad():
        #     teacher_x = self.teacher.extract_feat(batch_inputs)
        #     out_teacher = self.teacher.bbox_head(teacher_x)
        
        losses = self.model.bbox_head.loss(x, batch_data_samples)
        return losses
    

    def predict(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Return the detection results of the
            input images. The returns value is DetDataSample,
            which usually contain 'pred_instances'. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).
        """
      
        if batch_data_samples[0].frame_id==0:
            self.reset_model()
            
        with torch.enable_grad():
            outs = self.adapt(batch_inputs,batch_data_samples)
        
        return outs


    def softmax_entropy(self,x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        
        return -(x.softmax(1) * x.log_softmax(1)).sum(1)
    
    def sigmoid_entropy(self,x: torch.Tensor) -> torch.Tensor:
        """Entropy of softmax distribution from logits."""
        sigmoid = nn.Sigmoid()
        log_sigmoid = nn.LogSigmoid()
        return -(sigmoid(x) * log_sigmoid(x)).sum(1)
    
    
    
    def adapt(self,batch_inputs,batch_data_samples):
        
        det_results = self.model.predict(batch_inputs,batch_data_samples)
        
        img_feats = self.model.extract_feat(batch_inputs)
        head_inputs_dict = self.model.forward_transformer(img_feats,
                                                    batch_data_samples)
        all_layers_outputs_classes, _ = self.model.bbox_head(**head_inputs_dict)
        
        all_layers_outputs_classes = all_layers_outputs_classes[-1][0]

        if self.entropy_type=="softmax":
            loss = self.softmax_entropy(all_layers_outputs_classes).mean()
        else:
            loss = self.sigmoid_entropy(all_layers_outputs_classes).mean()
            
       
        
    
        loss.backward()
        self.optim_wrapper.step()
        
        # for name, parms in self.model.named_parameters():	
        #     if parms.grad is not None:
        #         print('-->name:', name)
        #         # print('-->para:', parms)
        #         print('-->grad_requirs:',parms.requires_grad)
        #         print('-->grad_value:',parms.grad)
        #         print("===")
                    
        self.optim_wrapper.zero_grad()
     
        return det_results
  
        
   
        
    def _forward(self, batch_inputs: Tensor,
                 batch_data_samples: SampleList) -> SampleList:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
            
        return self.model(batch_inputs,batch_data_samples)

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        return self.model.extract_feat(batch_inputs)