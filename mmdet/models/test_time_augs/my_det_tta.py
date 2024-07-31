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
from mmengine.dataset import Compose
from mmengine.optim import build_optim_wrapper
from copy import deepcopy
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_flip
from mmcv.ops import batched_nms
from mmdet.models.detectors.base import BaseDetector

@MODELS.register_module()
class MyDetTTA(BaseDetector):
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
        model_cfg = None,
        tta_cfg=None,
        ckpt_path = None,
        **kwags,
    ) -> None:
        super().__init__(**kwags)

        
        # build optimizer
        self.model_cfg = model_cfg
        self.tta_cfg = tta_cfg  
        self.ckpt_path = ckpt_path
        self.model = MODELS.build(self.model_cfg)
        self._init_source_model_state()
                
        
    def _init_source_model_state(self) -> None:
        """Init self.source_model_state.
        
        Args:
            model (nn.Module): detection model.
        """

        load_checkpoint(
                self.model, self.ckpt_path, map_location='cpu', revise_keys=[(r'^teacher\.', '')])

                

        



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
        # x = self.student.extract_feat(batch_inputs)
        # with torch.no_grad():
        #     teacher_x = self.teacher.extract_feat(batch_inputs)
        #     out_teacher = self.teacher.bbox_head(teacher_x)
        # losses = self.student.bbox_head.loss(x, batch_data_samples, out_teacher)
        # return losses
        pass
    

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
      
        preds = self.model(batch_inputs,batch_data_samples,mode='predict')
        
        score_list = [pred.pred_instances.scores  for pred in preds]
        bbox_list = [pred.pred_instances.bboxes  for pred in preds]
        labels_list =  [pred.pred_instances.labels  for pred in preds]
        img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
        
        merged_bboxes,merged_scores = self.merge_aug_bboxes(bbox_list, score_list, img_metas)
        merged_labels = torch.cat(labels_list, dim=0)
        
        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.tta_cfg.nms)

        det_bboxes = det_bboxes[:self.tta_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.tta_cfg.max_per_img]

        results = InstanceData()
        _det_bboxes = det_bboxes.clone()
        results.bboxes = _det_bboxes[:, :-1]
        results.scores = _det_bboxes[:, -1]
        results.labels = det_labels
        det_results = preds[0]
        det_results.pred_instances = results
            

        return [det_results]


    def adapt(self,batch_inputs,batch_data_samples):
        
        num_aug = len(batch_data_samples)
        ori_idx = 0
        num_view = num_aug -1
        
        ori_inputs = batch_inputs[ori_idx].unsqueeze(0)
        ori_data_sample = [batch_data_samples[ori_idx]]
        
        aug_inputs =  batch_inputs[1:]
        aug_data_samples = batch_data_samples[1:]

        with torch.no_grad():
            # teacher加aug
            stu_preds = self.student(ori_inputs, ori_data_sample, mode='predict')
            tea_preds = self.teacher(aug_inputs,aug_data_samples,mode='predict')
        
        stu_score = stu_preds[0].pred_instances.scores
        stu_bbox = stu_preds[0].pred_instances.bboxes
            
        tea_score_list = [pred.pred_instances.scores  for pred in tea_preds]
        tea_bbox_list = [pred.pred_instances.bboxes  for pred in tea_preds]
        tea_labels_list =  [pred.pred_instances.labels  for pred in tea_preds]
        
        tea_merged_score,tea_merged_box = self.merge_aug_bboxes(tea_score_list,tea_bbox_list,aug_data_samples)
        tea_merged_label = torch.cat(tea_labels_list,dim=0)
        
        # [num_box,5]
        tea_det_bboxes, keep_idxs = batched_nms(tea_merged_box, tea_merged_score,
                                            tea_merged_label, self.tta_cfg.nms)
        tea_det_labels = tea_merged_label[keep_idxs]
        if tea_det_bboxes.shape[0] > self.tta_cfg.max_per_img:
            tea_det_bboxes = tea_det_bboxes[:self.tta_cfg.max_per_img]
            tea_det_labels = tea_det_labels[:self.tta_cfg.max_per_img]
      
        
        results = InstanceData()
        _det_bboxes = tea_det_bboxes.clone()
        results.bboxes = _det_bboxes[:, :-1]
        results.scores = _det_bboxes[:, -1]
        results.labels = tea_det_labels
        # 将teacher merged的预测结果作为stu的gt
        ori_data_sample[0].gt_instances = results
        
        
        # 计算loss
        self.student.train(True)
        loss = self.student(ori_inputs,ori_data_sample,mode='loss')

        loss  = (loss['loss_cls'] + loss['loss_bbox'] + loss['loss_iou'])/3
        loss.requires_grad_(True) 
        loss.backward()
        self.optim_wrapper.step()
        self.optim_wrapper.zero_grad()
        
    
        # 返回teacher merge后的预测结果   这里用teacher还是stu?
        self.student.train(False)
        return self.student(ori_inputs, ori_data_sample, mode='predict')
        
        
    def merge_aug_bboxes(self, aug_bboxes,aug_scores,img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            # 这里用什么shape?
            batch_input_shape = img_info['ori_shape']
            try: 
                flip = img_info['flip']
                flip_direction = img_info['flip_direction']
            except KeyError:
                flip = False
        
            if flip:
                bboxes = bbox_flip(
                    bboxes=bboxes,
                    img_shape=batch_input_shape,
                    direction=flip_direction)
            # if ori_shape!=    
                
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores        
        
        

   
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
        if self.semi_test_cfg.get('forward_on', 'teacher') == 'teacher':
            return self.teacher(
                batch_inputs, batch_data_samples, mode='tensor')
        else:
            return self.student(
                batch_inputs, batch_data_samples, mode='tensor')

    def extract_feat(self, batch_inputs: Tensor):
        """Extract features from images."""
        return self.student.extract_feat(batch_inputs)