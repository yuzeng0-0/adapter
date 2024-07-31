



# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import torch
from mmcv.ops import batched_nms
from mmengine.model import BaseTTAModel
from mmengine.registry import MODELS
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.structures import DetDataSample
from mmdet.structures.bbox import bbox_flip
from typing import Dict, List, Optional, Union
import torch.nn as nn
from mmengine.optim import OptimWrapper


@MODELS.register_module()
class DetTentModel(BaseTTAModel):
    """Merge augmented detection results, only bboxes corresponding score under
    flipping and multi-scale resizing can be processed now.

    Examples:
        >>> tta_model = dict(
        >>>     type='DetTTAModel',
        >>>     tta_cfg=dict(nms=dict(
        >>>                     type='nms',
        >>>                     iou_threshold=0.5),
        >>>                     max_per_img=100))
        >>>
        >>> tta_pipeline = [
        >>>     dict(type='LoadImageFromFile',
        >>>          backend_args=None),
        >>>     dict(
        >>>         type='TestTimeAug',
        >>>         transforms=[[
        >>>             dict(type='Resize',
        >>>                  scale=(1333, 800),
        >>>                  keep_ratio=True),
        >>>         ], [
        >>>             dict(type='RandomFlip', prob=1.),
        >>>             dict(type='RandomFlip', prob=0.)
        >>>         ], [
        >>>             dict(
        >>>                 type='PackDetInputs',
        >>>                 meta_keys=('img_id', 'img_path', 'ori_shape',
        >>>                         'img_shape', 'scale_factor', 'flip',
        >>>                         'flip_direction'))
        >>>         ]])]
    """

    def __init__(self, tta_cfg=None, **kwargs):
        super().__init__(**kwargs)
        self.tta_cfg = tta_cfg

    def merge_aug_bboxes(self, aug_bboxes: List[Tensor],
                         aug_scores: List[Tensor],
                         img_metas: List[str]) -> Tuple[Tensor, Tensor]:
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
            ori_shape = img_info['ori_shape']
            flip = img_info['flip']
            flip_direction = img_info['flip_direction']
            if flip:
                bboxes = bbox_flip(
                    bboxes=bboxes,
                    img_shape=ori_shape,
                    direction=flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

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
   

        det_results = data_samples[0]   # 没有flip的

        return det_results
    
    def test_step(self, data):
        """Get predictions of each enhanced data, a multiple predictions.

        Args:
            data (DataBatch): Enhanced data batch sampled from dataloader.

        Returns:
            MergedDataSamples: Merged prediction.
        """
        data_list: Union[List[dict], List[list]]
        if isinstance(data, dict):
            num_augs = len(data[next(iter(data))])
            data_list = [{key: value[idx]
                          for key, value in data.items()}
                         for idx in range(num_augs)]
        elif isinstance(data, (tuple, list)):
            num_augs = len(data[0])
            data_list = [[_data[idx] for _data in data]
                         for idx in range(num_augs)]
        else:
            raise TypeError('data given by dataLoader should be a dict, '
                            f'tuple or a list, but got {type(data)}')
            
        # self.module.train()
        max_per_img = 300
        predictions = []
        # self.module.eval()
        criterion = nn.CrossEntropyLoss()
        # training_parma = self.get_training_params()
        param_list = []
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                if param.requires_grad and ("norm" in name or "bn" in name):         # 只更新norm和bn的参数
                    param_list.append(param)
                # print (name)
                else:
                    param.requires_grad=False
        
        optimizer = torch.optim.Adam(param_list, lr=0.00006/8, betas=(0.9, 0.999))

        optim_wrapper=OptimWrapper(optimizer=optimizer)
         
        for data in data_list:  # type: ignore

            
            device = self.module.parameters().__next__().device
            data = self.module.data_preprocessor(data, False)
            inputs = data['inputs'][0].unsqueeze(0).float()
            predict =self.module(inputs.to(device),data['data_samples'])
            
            result = self.module._run_forward(data, mode='predict')
            
            cls_score = predict[0][-1][0].sigmoid()
            bbox_pred = predict[0][-1][0]
        
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.module.bbox_head.num_classes
            bbox_index = indexes // self.module.bbox_head.num_classes
            cls_score = cls_score[bbox_index]
            bbox_pred = bbox_pred[bbox_index]
        
            loss = criterion(cls_score, det_labels)
            loss.requires_grad = True
        
            optim_wrapper.update_params(loss)

        return result
        # return pred
    
    
    def get_training_params(self):
        
        param_list = []
        for name, param in self.module.named_parameters():
            if param.requires_grad:
                if param.requires_grad and ("norm" in name or "bn" in name):         # 只更新norm和bn的参数
                    param_list.append(param)
                    print (name)
                else:
                    param.requires_grad=False
                    
        return param_list
