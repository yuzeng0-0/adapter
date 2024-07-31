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


@MODELS.register_module()
class ViDA(KnowledgeDistillationSingleStageDetectorV2):
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
        
        self.tea_injected_model = inject_trainable_Vida(self.teacher)
        self.stu_injected_model = inject_trainable_Vida(self.student)
        
        self.inject_source_model_state = deepcopy(self.tea_injected_model.state_dict())
 
        
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
        self.tea_injected_model.load_state_dict(self.inject_source_model_state, strict=True)
        self.stu_injected_model.load_state_dict(self.inject_source_model_state, strict=True)
        


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
        self.momentum_update(self.momentum)
        
        
        return outs


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
            stu_preds = self.stu_injected_model(ori_inputs, ori_data_sample, mode='predict')
            tea_preds = self.tea_injected_model(aug_inputs,aug_data_samples,mode='predict')
        
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
        self.stu_injected_model.train(True)
        loss = self.stu_injected_model(ori_inputs,ori_data_sample,mode='loss')

        loss  = (loss['loss_cls'] + loss['loss_bbox'] + loss['loss_iou'])/3
        loss.requires_grad_(True) 
        loss.backward()
        self.optim_wrapper.step()
        self.optim_wrapper.zero_grad()
        
    
        # 返回teacher merge后的预测结果   这里用teacher还是stu?
        self.stu_injected_model.train(False)
        return self.stu_injected_model(ori_inputs, ori_data_sample, mode='predict')
        
        
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
            batch_input_shape = img_info.batch_input_shape
            try: 
                flip = img_info.flip
                flip_direction = img_info.flip_direction
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
                    
                    
        
class VidaInjectedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False, r=4, r2 = 64):
        super().__init__()

        self.linearVida = nn.Linear(in_features, out_features, bias)
        self.Vida_down = nn.Linear(in_features, r, bias=False)
        self.Vida_up = nn.Linear(r, out_features, bias=False)
        self.Vida_down2 = nn.Linear(in_features, r2, bias=False)
        self.Vida_up2 = nn.Linear(r2, out_features, bias=False)
        self.scale = 1.0

        nn.init.normal_(self.Vida_down.weight, std=1 / r**2)
        nn.init.zeros_(self.Vida_up.weight)

        nn.init.normal_(self.Vida_down2.weight, std=1 / r**2)
        nn.init.zeros_(self.Vida_up2.weight)

    def forward(self, input):
        return self.linearVida(input) + self.Vida_up(self.Vida_down(input)) * self.scale + self.Vida_up2(self.Vida_down2(input)) * self.scale



def inject_trainable_Vida(
    model: nn.Module,
    # target_replace_module: List[str] = ["CrossAttention", "Attention"],
    target_replace_module: List[str] = ["MultiheadAttention", "MultiScaleDeformableAttention"],
    r: int = 4,
    r2: int = 16,
):
    """
    inject Vida into model, and returns Vida parameter groups.
    """
    inject_model = deepcopy(model)
    require_grad_params = []
    names = []

    for _module in inject_model.modules():
        if _module.__class__.__name__ in target_replace_module:

            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = VidaInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        r2,
                    )
                    _tmp.linearVida.weight = weight
                    if bias is not None:
                        _tmp.linearVida.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp

                    require_grad_params.extend(
                        list(_module._modules[name].Vida_up.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].Vida_down.parameters())
                    )
                    _module._modules[name].Vida_up.weight.requires_grad = True
                    _module._modules[name].Vida_down.weight.requires_grad = True

                    require_grad_params.extend(
                        list(_module._modules[name].Vida_up2.parameters())
                    )
                    require_grad_params.extend(
                        list(_module._modules[name].Vida_down2.parameters())
                    )
                    _module._modules[name].Vida_up2.weight.requires_grad = True
                    _module._modules[name].Vida_down2.weight.requires_grad = True                    
                    names.append(name)

    return inject_model