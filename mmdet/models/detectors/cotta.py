# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import Any, Optional, Union,Tuple,List

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
from mmdet.structures import DetDataSample

@MODELS.register_module()
class Cotta(KnowledgeDistillationSingleStageDetectorV2):
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
        tta_cfg = None,
        **kwags,
    ) -> None:
        super().__init__(**kwags)
        
        self.consistency_loss = MODELS.build(consistency_loss)
        # self.stu_pipeline = Compose(stu_pipeline)
        
        # build optimizer
        self.optim_wrapper = None
        if optim_wrapper is not None:
            self.optim_wrapper_cfg = optim_wrapper
        self.optim_steps = optim_steps
        self.skip_buffers = skip_buffers
        self.momentum = momentum
        self.tta_cfg = tta_cfg  
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
      
        if batch_data_samples[0].frame_id==0:
            self.reset_model()
            
        
        outs = self.adapt(batch_inputs,batch_data_samples)
        # self.momentum_update(self.momentum)     
        
        # # restore parameters
        # for nm, m  in self.student.named_modules():
        #     for npp, p in m.named_parameters():
        #         if npp in ['weight', 'bias'] and p.requires_grad:
        #             mask = (torch.rand(p.shape)<0.01).float().cuda() 
        #             with torch.no_grad():
        #                 p.data = self.source_model_state[f"{nm}.{npp}"].to(p.data.device) * mask + p * (1.-mask)
        
        return outs


    def adapt(self,batch_inputs,batch_data_samples):
        
        num_aug = len(batch_data_samples)
        ori_idx = 0
        num_view = num_aug -1
        
        ori_inputs = batch_inputs[ori_idx].unsqueeze(0)
        ori_data_sample = [batch_data_samples[ori_idx]]
        
        aug_inputs =  batch_inputs[1:]
        aug_data_samples = batch_data_samples[1:]

   
        # stu_preds = self.student(ori_inputs, ori_data_sample, mode='predict')
        tea_preds = self.teacher(aug_inputs,aug_data_samples,mode='predict')
        
        # stu_score = stu_preds[0].pred_instances.scores
        # stu_bbox = stu_preds[0].pred_instances.bboxes
            
        tea_score_list = [pred.pred_instances.scores  for pred in tea_preds]
        tea_bbox_list = [pred.pred_instances.bboxes  for pred in tea_preds]
        tea_labels_list =  [pred.pred_instances.labels  for pred in tea_preds]
        
        # tea_merged_box,tea_merged_score = self.merge_filpaug_bboxes(tea_bbox_list,tea_score_list,aug_data_samples)
        tea_merged_box,tea_merged_score = self.merge_resizeaug_bboxes(tea_bbox_list,tea_score_list,aug_data_samples)
        tea_merged_label = torch.cat(tea_labels_list,dim=0)
        
        # # [num_box,5]
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
        
        det_results = batch_data_samples[0]
        det_results.pred_instances = results
        
        return [det_results]
        # 将teacher merged的预测结果作为stu的gt
        # ori_data_sample[0].gt_instances = results
        
        
        # # 计算loss
        # self.student.train(True)
        # loss = self.student(ori_inputs,ori_data_sample,mode='loss')

        # loss  = (loss['loss_cls'] + loss['loss_bbox'] + loss['loss_iou'])/3
        # loss.requires_grad_(True) 
        # loss.backward()
        # self.optim_wrapper.step()
        # self.optim_wrapper.zero_grad()
        
    
        # 返回teacher merge后的预测结果   这里用teacher还是stu?
        # self.student.train(False)
        # return self.student(ori_inputs, ori_data_sample, mode='predict')
        
        
    def merge_filpaug_bboxes(self, aug_bboxes,aug_scores,img_metas):
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

            batch_input_shape = img_info.ori_shape
  
            flip = img_info.flip
            flip_direction = img_info.flip_direction  
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
    def merge_resizeaug_bboxes(self, aug_bboxes,aug_scores,img_metas):
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
          
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores       
        
        

            
    def momentum_update(self, momentum: float) -> None:
        """Compute the moving average of the parameters using exponential
        moving average."""
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
                    
                    
    
                    
                    
    def test_step(self, data):
            """Get predictions of each enhanced data, a multiple predictions.

            Args:
                data (DataBatch): Enhanced data batch sampled from dataloader.

            Returns:
                MergedDataSamples: Merged prediction.
            """
            num_augs = len(data[next(iter(data))])
            data_list = [{key: value[idx]
                          for key, value in data.items()}
                         for idx in range(num_augs)]

            tea_pred = []
            stu_pred = self.student.test_step(data_list[0])
            
            for data in data_list[1:]:  # type: ignore
                tea_pred.append(self.teacher.test_step(data))
                
            self.merge_preds(tea_pred)  # type: ignore
                
            
            return self.merge_preds(list(zip(*tea_pred)))  # type: ignore


    def merge_preds(self, data_samples_list: List[List[DetDataSample]]):
        """Merge batch predictions of enhanced data.

        Args:
            data_samples_list (List[List[DetDataSample]]): List of predictions
                of all enhanced data. The outer list indicates images, and the
                inner list corresponds to the different views of one image.
                Each element of the inner list is a ``DetDataSample``.
        Returns:
            List[DetDataSample]: Merged batch prediction.
        """
        merged_data_samples = []
        for data_samples in data_samples_list:
            merged_data_samples.append(self._merge_single_sample(data_samples))
        return merged_data_samples
    
    
    def _merge_single_sample(
            self, data_samples: List[DetDataSample]) -> DetDataSample:
        """Merge predictions which come form the different views of one image
        to one prediction.

        Args:
            data_samples (List[DetDataSample]): List of predictions
            of enhanced data which come form one image.
        Returns:
            List[DetDataSample]: Merged prediction.
        """
        aug_bboxes = []
        aug_scores = []
        aug_labels = []
        img_metas = []
        # TODO: support instance segmentation TTA
        assert data_samples[0].pred_instances.get('masks', None) is None, \
            'TTA of instance segmentation does not support now.'
        for data_sample in data_samples:
            aug_bboxes.append(data_sample.pred_instances.bboxes)
            aug_scores.append(data_sample.pred_instances.scores)
            aug_labels.append(data_sample.pred_instances.labels)
            img_metas.append(data_sample.metainfo)

        # merged_bboxes, merged_scores = self.merge_resizeaug_bboxes(
        #     aug_bboxes, aug_scores, img_metas)
        
        merged_bboxes, merged_scores = self.merge_filpaug_bboxes(
            aug_bboxes, aug_scores, img_metas)
        
        merged_labels = torch.cat(aug_labels, dim=0)

        if merged_bboxes.numel() == 0:
            return data_samples[0]

        det_bboxes, keep_idxs = batched_nms(merged_bboxes, merged_scores,
                                            merged_labels, self.tta_cfg.nms)

        det_bboxes = det_bboxes[:self.tta_cfg.max_per_img]
        det_labels = merged_labels[keep_idxs][:self.tta_cfg.max_per_img]

        results = InstanceData()
        _det_bboxes = det_bboxes.clone()
        results.bboxes = _det_bboxes[:, :-1]
        results.scores = _det_bboxes[:, -1]
        results.labels = det_labels
        det_results = data_samples[0]
        det_results.pred_instances = results
        return det_results
