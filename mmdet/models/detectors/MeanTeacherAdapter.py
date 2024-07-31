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


@MODELS.register_module()
class MeanTeacherAdapter(KnowledgeDistillationSingleStageDetectorV2):
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
        consistency_loss=None,
        optim_wrapper: Optional[dict] = None,
        optim_steps: int = 0,
        skip_buffers = True,
        momentum = 0.001,
        reset_optimizer = True,
        score_thr = None,
        **kwags,
    ) -> None:
        super().__init__(**kwags)
        
        self.consistency_loss = MODELS.build(consistency_loss)
        self.score_thr = score_thr
        self.skip_buffers = skip_buffers
        # self.stu_pipeline = Compose(stu_pipeline)
        
        # build optimizer
        self.optim_wrapper = None
        if optim_wrapper is not None:
            self.optim_wrapper_cfg = optim_wrapper
        self.optim_steps = optim_steps
        self.momentum = momentum
        self.reset_optimizer = reset_optimizer
        
        self._init_source_model_state()
        
        # TODO: implement param_scheduler for optimizer (e.g. lr decay)
        
        
    def _init_source_model_state(self) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """
        
        if self.teacher_ckpt is not None:
            # NOTE we use the teacher model to initialize the next stage for both teacher and student
            load_checkpoint(
                self.teacher, self.teacher_ckpt, map_location='cpu', revise_keys=[(r'^teacher\.', '')])
            if self.init_student:
                load_checkpoint(
                    self.student, self.teacher_ckpt, map_location='cpu', revise_keys=[(r'^teacher\.', '')])
                
        self.source_model_state = deepcopy(self.teacher.state_dict())
 
        self.optim_wrapper = build_optim_wrapper(self.student, self.optim_wrapper_cfg)
        self.optim_wrapper_state = deepcopy(self.optim_wrapper.state_dict())
        
    def reset_model(self):
        
        if self.source_model_state is None:
            raise Exception("cannot reset without saved model state")
        self.teacher.load_state_dict(self.source_model_state, strict=True)
        self.student.load_state_dict(self.source_model_state, strict=True)
        


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
        x = self.student.extract_feat(batch_inputs)
        with torch.no_grad():
            teacher_x = self.teacher.extract_feat(batch_inputs)
            out_teacher = self.teacher.bbox_head(teacher_x)
        losses = self.student.bbox_head.loss(x, batch_data_samples, out_teacher)
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

        # self._reset_optimizer()
        if batch_data_samples[0].frame_id==0:
            self.reset_model()
            print("reset model!")
            
            
        with torch.enable_grad():
            for _ in range(self.optim_steps):
                outs = self.adapt(batch_inputs,batch_data_samples)
                self.momentum_update(self.momentum)
        
        if self.reset_optimizer:
            self._reset_optimizer()
        # pred_det_instances = outs[0].pred_instances.clone()
        
        return outs


    def adapt(self,batch_inputs,batch_data_samples):

        num_aug = len(batch_data_samples)
        ori_idx = 0
        num_views = num_aug -1 
  
        ori_inputs = batch_inputs[ori_idx].unsqueeze(0)
        ori_data_sample = [batch_data_samples[ori_idx]]
        aug_inputs = batch_inputs[1:]
        aug_data_sample = batch_data_samples[1:]

        # 先按照官方给的baseline中stu加aug
    
        teacher_det_results, teacher_outs = self._detect_forward(
            self.teacher, ori_inputs, ori_data_sample)
        
        _, stu_outs = self._detect_forward(
            self.student, aug_inputs, aug_data_sample)
        
        teacher_outs = dict(
            hidden_states = teacher_outs['hidden_states'][-1].repeat(num_views,1,1),
            references = teacher_outs['references'][-1].repeat(num_views,1,1)
        )
        stu_outs = dict(
            hidden_states = stu_outs['hidden_states'][-1],
            references = stu_outs['references'][-1]
        )
        
        loss = self.consistency_loss(stu_outs,teacher_outs)

        loss.backward()
        self.optim_wrapper.step()
        self.optim_wrapper.zero_grad()
        # print("loss:",loss)
        
        return teacher_det_results
        
        
    def _expand_view(self, outs: Tuple[torch.Tensor], views: int = 1):
        """Expand batch size of each element in outs to views."""
        outs =tuple((o.repeat_interleave(2, dim=1) for o in outs))
        return outs
    
    
    def _reset_optimizer(self) -> None:
        """Reset optimizer state.
        
        Args:
            model (nn.Module): detection model."""
        if self.optim_wrapper is not None:
            self.optim_wrapper.load_state_dict(self.optim_wrapper_state)
            
            
    def momentum_update(self, momentum: float) -> None:
        """Compute the moving average of the parameters using exponential
        moving average."""
        # sum = 0
        if self.skip_buffers:
            for (src_name, src_parm), (dst_name, dst_parm) in zip(
                    self.student.named_parameters(),
                    self.teacher.named_parameters()):
                dst_parm.data.mul_(1 - momentum).add_(
                    src_parm.data, alpha=momentum)
        else:
            for (src_parm,
                 dst_parm) in zip(self.student.state_dict().values(),
                                  self.teacher.state_dict().values()):
                # exclude num_tracking
                if dst_parm.dtype.is_floating_point:
                    dst_parm.data.mul_(1 - momentum).add_(
                        src_parm.data, alpha=momentum)
                    
                    
    def _detect_forward(self,detector,batch_inputs,batch_data_samples,rescale=True):
        
        
        img_feats = detector.extract_feat(batch_inputs)
        head_inputs_dict = detector.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = detector.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        det_results = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        
        return det_results,head_inputs_dict
        
        
        