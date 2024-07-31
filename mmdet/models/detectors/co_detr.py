import warnings

import torch
from torch import Tensor
import torch.nn as nn

# from mmdet.core import bbox2result
# from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.base import BaseDetector
from mmdet.registry import MODELS

from mmdet.models.detectors import DINO
from mmdet.structures import DetDataSample, OptSampleList, SampleList
from typing import Dict, List, Tuple, Union,Optional
from .deformable_detr import DeformableDETR
from mmdet.models.layers import CdnQueryGenerator,SinePositionalEncoding,DeformableDetrTransformerEncoder,DinoTransformerDecoder

from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,bbox_overlaps
from ..utils import multi_apply
import torch.nn.functional as F
from mmdet.models.layers.transformer.utils import inverse_sigmoid
from mmdet.models.layers.transformer.co_transformer import CoDinoTransformerDecoder
from mmdet.utils import reduce_mean

@MODELS.register_module()
class CoDETR(DeformableDETR):
    def __init__(self,
                *args, 
                #  backbone,
                #  neck=None,
                 query_head=None,
                 rpn_head=None,
                 roi_head=[None],
                 co_bbox_head=[None],
                 train_cfg=[None, None],
                 test_cfg=[None, None],
                 pretrained=[None, None],
                 init_cfg=None,
                 with_pos_coord=True,
                 with_attn_mask=True,
                 eval_module='detr',
                 eval_index=0,
                 dn_cfg = None,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.with_pos_coord = with_pos_coord
        self.with_attn_mask = with_attn_mask
        # Module for evaluation, ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module
        # Module index for evaluation
        self.eval_index = eval_index
        # self.backbone = MODELS.build(backbone)

        head_idx = 0
        
        if query_head is not None:
            query_head.update(train_cfg=train_cfg[head_idx] if (train_cfg is not None and train_cfg[head_idx] is not None) else None)
            query_head.update(test_cfg=test_cfg[head_idx])
            
            query_head['share_pred_layer'] = not kwargs['with_box_refine']
            query_head['num_pred_layer'] = (kwargs['decoder']['num_layers'] + 1) \
                if self.as_two_stage else kwargs['decoder']['num_layers']
            query_head['as_two_stage'] = kwargs['as_two_stage']
            
            self.query_head = MODELS.build(query_head)
            self.query_head.init_weights()
            head_idx += 1

        if rpn_head is not None:
            rpn_train_cfg = train_cfg[head_idx].rpn if (train_cfg is not None and train_cfg[head_idx] is not None) else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg[head_idx].rpn)
            self.rpn_head = MODELS.build(rpn_head_)
            self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                rcnn_train_cfg = train_cfg[i+head_idx].rcnn if (train_cfg and train_cfg[i+head_idx] is not None) else None
                roi_head[i].update(train_cfg=rcnn_train_cfg)
                roi_head[i].update(test_cfg=test_cfg[i+head_idx].rcnn)
                self.roi_head.append(MODELS.build(roi_head[i]))
                self.roi_head[-1].init_weights()

        self.co_bbox_head = nn.ModuleList()
        for i in range(len(co_bbox_head)):
            if co_bbox_head[i]:
                co_bbox_head[i].update(train_cfg=train_cfg[i+head_idx+len(self.roi_head)] if (train_cfg and train_cfg[i+head_idx+len(self.roi_head)] is not None) else None)
                co_bbox_head[i].update(test_cfg=test_cfg[i+head_idx+len(self.roi_head)])
                self.co_bbox_head.append(MODELS.build(co_bbox_head[i]))  
                self.co_bbox_head[-1].init_weights() 
                
        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.query_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
                

        self.head_idx = head_idx
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        
        
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = CoDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'


        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)
        
        # self.pos_trans_fc = nn.Linear(self.embed_dims * 2,
        #                                   self.embed_dims * 2)
        self.pos_trans_fc = None
        # additional 
   
        self.downsample = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, self.embed_dims)
        )

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_query_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'query_head') and self.query_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None and len(self.roi_head)>0)
                or (hasattr(self, 'co_bbox_head') and self.co_bbox_head is not None and len(self.co_bbox_head)>0))

    def extract_feat(self, img, img_metas=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x


    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        
        
        outs = []
        if self.with_rpn:
        # for rpn head
            memory = encoder_outputs_dict['memory']
            num_level = len(img_feats)
            start = 0
            for lvl in range(num_level):
                bs, c, h, w = img_feats[lvl].shape
                end = start + h*w
                # NOTE this is a bug，we should change h*w to the last dim
                feat = memory[:,start:end].permute(0,2,1).contiguous()
                # feat = memory[:,start:end].contiguous()
                start = end
                outs.append(feat.reshape(bs, c, h, w))
            outs.append(self.downsample(outs[-1]))
            
        # return head_inputs_dict,outs
        return head_inputs_dict


    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.query_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.query_head.cls_branches[
            self.decoder.num_layers](
                output_memory)
        enc_outputs_coord_unact = self.query_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(self,
                        query: Tensor,
                        memory: Tensor,
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.query_head.reg_branches)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        losses = dict()
        
        def upd_loss(losses, idx, weight=1):
            new_losses = dict()
            for k,v in losses.items():
                new_k = '{}{}'.format(k,idx)
                if isinstance(v,list) or isinstance(v,tuple):
                    new_losses[new_k] = [i*weight for i in v]
                else:new_losses[new_k] = v*weight
            return new_losses
        
        img_feats = self.extract_feat(batch_inputs)
        # head_inputs_dict,new_img_feats = self.forward_transformer(img_feats,
        #                                             batch_data_samples)
        new_img_feats = img_feats
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        if self.with_query_head:
            bbox_losses= self.query_head.loss(
                **head_inputs_dict, batch_data_samples=batch_data_samples)
            losses.update(bbox_losses)
            
        if self.with_rpn:
            proposal_cfg = self.train_cfg[self.head_idx].get('rpn_proposal',
                                                self.test_cfg[self.head_idx].rpn)
            rpn_losses, proposal_list = self.rpn_head.loss_and_predict(new_img_feats,batch_data_samples,proposal_cfg)
            losses.update(rpn_losses)

        # for roi_head,convert bbox type
        img_metas = []
        gt_bboxes = []
        gt_labels = []
        # gt_bboxes_ignore = []
        # gt_masks = []
        for idx in range(len(batch_data_samples)):
            img_metas.append(batch_data_samples[idx].metainfo)
            gt_bboxes.append(batch_data_samples[idx].gt_instances.bboxes)
            gt_labels.append(batch_data_samples[idx].gt_instances.labels)
    
        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].forward_train(new_img_feats,img_metas,proposal_list,gt_bboxes,gt_labels,
                                                    gt_bboxes_ignore=None, gt_masks=None)
            
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))
            else: 
                if 'pos_coords' in roi_losses.keys():
                    tmp = roi_losses.pop('pos_coords')     
            roi_losses = upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)
            
            
        for i in range(len(self.co_bbox_head)):
            bbox_losses = self.co_bbox_head[i].forward_train(new_img_feats, img_metas, gt_bboxes,
                                                        gt_labels, gt_bboxes_ignore=None)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    tmp = bbox_losses.pop('pos_coords')          
            bbox_losses = upd_loss(bbox_losses, idx=i+len(self.roi_head))
            losses.update(bbox_losses)
            
        if self.with_pos_coord and len(positive_coords)>0:
            for i in range(len(positive_coords)):
                bbox_losses = self.forward_train_aux(new_img_feats, img_metas, gt_bboxes,
                                                            gt_labels, positive_coords[i], i,gt_bboxes_ignore=None)
                bbox_losses = upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)                    


        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        # 因为self.forward_transformer比原本的多return了一个new img feat，因此这里head_inputs_dict由dict 变为了tuple
        results_list = self.query_head.predict(
            **head_inputs_dict[0],
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.co_bbox_head.forward(**head_inputs_dict)
        return results
    
    
    
    
    def forward_train_aux(self,
                          x,
                          img_metas,
                          gt_bboxes,
                          gt_labels=None,
                          pos_coords=None,
                          head_idx=0,
                          gt_bboxes_ignore=None,
                          **kwargs):
        """Forward function for training mode.

        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        aux_targets = self.get_aux_targets(pos_coords, img_metas, x, head_idx)
        outs = self.forward_aux(x[:-1], img_metas, aux_targets, head_idx)
        outs = outs + aux_targets
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, img_metas)
        losses = self.loss_aux(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses


    def loss_aux(self,
                 all_cls_scores,
                 all_bbox_preds,
                 enc_cls_scores,
                 enc_bbox_preds,
                 aux_coords, 
                 aux_labels, 
                 aux_targets, 
                 aux_label_weights, 
                 aux_bbox_weights,
                 aux_feats,
                 attn_masks,
                 gt_bboxes_list,
                 gt_labels_list,
                 img_metas,
                 gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and shape
                [nb_dec, bs, num_query, 4].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 4). Only be
                passed when as_two_stage is True, otherwise is None.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # gt_bboxes_ignore = None
        # assert gt_bboxes_ignore is None, \
        #     f'{self.__class__.__name__} only supports ' \
        #     f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_labels = [aux_labels for _ in range(num_dec_layers)]
        all_label_weights = [aux_label_weights for _ in range(num_dec_layers)]
        all_bbox_targets = [aux_targets for _ in range(num_dec_layers)]
        all_bbox_weights = [aux_bbox_weights for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_single_aux, all_cls_scores, all_bbox_preds,
            all_labels, all_label_weights, all_bbox_targets, 
            all_bbox_weights, img_metas_list, all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.

        # loss from the last decoder layer
        loss_dict['loss_cls_aux'] = losses_cls[-1]
        loss_dict['loss_bbox_aux'] = losses_bbox[-1]
        loss_dict['loss_iou_aux'] = losses_iou[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_iou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_aux'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_aux'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.loss_iou_aux'] = loss_iou_i
            num_dec_layer += 1
        return loss_dict


    def get_aux_targets(self, pos_coords, img_metas, mlvl_feats, head_idx):
        coords, labels, targets = pos_coords[:3]
        head_name = pos_coords[-1]
        bs, c = len(coords), mlvl_feats[0].shape[1]
        max_num_coords = 0
        all_feats = []
        for i in range(bs):
            label = labels[i]
            feats = [feat[i].reshape(c, -1).transpose(1, 0) for feat in mlvl_feats]
            feats = torch.cat(feats, dim=0)
            bg_class_ind = self.query_head.num_classes
            pos_inds = ((label >= 0)
                        & (label < bg_class_ind)).nonzero().squeeze(1)  
            max_num_coords = max(max_num_coords, len(pos_inds))
            all_feats.append(feats)
        max_num_coords = min(self.query_head.max_pos_coords, max_num_coords)
        max_num_coords = max(9, max_num_coords)

        if self.query_head.use_zero_padding:
            attn_masks = []
            label_weights = coords[0].new_zeros([bs, max_num_coords])
        else:
            attn_masks = None
            label_weights = coords[0].new_ones([bs, max_num_coords])
        bbox_weights = coords[0].new_zeros([bs, max_num_coords, 4])

        aux_coords, aux_labels, aux_targets, aux_feats = [], [], [], []

        for i in range(bs):
            coord, label, target = coords[i], labels[i], targets[i]
            feats = all_feats[i]
            if 'rcnn' in head_name:
                feats = pos_coords[-2][i]
                num_coords_per_point = 1
            else:
                num_coords_per_point = coord.shape[0] // feats.shape[0]
            feats = feats.unsqueeze(1).repeat(1, num_coords_per_point, 1)
            feats = feats.reshape(feats.shape[0]*num_coords_per_point, feats.shape[-1])
            img_meta = img_metas[i]
            # img_h, img_w, _ = img_meta['img_shape']
            img_h, img_w = img_meta['img_shape']
            factor = coord.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0)
            bg_class_ind = self.query_head.num_classes
            pos_inds = ((label >= 0)
                        & (label < bg_class_ind)).nonzero().squeeze(1)
            neg_inds = ((label == bg_class_ind)).nonzero().squeeze(1)
            if pos_inds.shape[0] > max_num_coords:
                indices = torch.randperm(pos_inds.shape[0])[:max_num_coords].cuda()
                pos_inds = pos_inds[indices]

            coord = bbox_xyxy_to_cxcywh(coord[pos_inds] / factor)
            label = label[pos_inds]
            target = bbox_xyxy_to_cxcywh(target[pos_inds] / factor)
            feat = feats[pos_inds]

            if self.query_head.use_zero_padding:
                label_weights[i][:len(label)] = 1
                bbox_weights[i][:len(label)] = 1
                attn_mask = torch.zeros([max_num_coords, max_num_coords,]).bool().to(coord.device)
            else:
                bbox_weights[i][:len(label)] = 1

            if coord.shape[0] < max_num_coords:
                padding_shape = max_num_coords-coord.shape[0]
                if self.query_head.use_zero_padding:
                    padding_coord = coord.new_zeros([padding_shape, 4])
                    padding_label = label.new_ones([padding_shape]) * self.query_head.num_classes
                    padding_target = target.new_zeros([padding_shape, 4])
                    padding_feat = feat.new_zeros([padding_shape, c])
                    attn_mask[coord.shape[0] :, 0 : coord.shape[0],] = True
                    attn_mask[:, coord.shape[0] :,] = True
                else:
                    indices = torch.randperm(neg_inds.shape[0])[:padding_shape].cuda()
                    neg_inds = neg_inds[indices]
                    padding_coord = bbox_xyxy_to_cxcywh(coords[i][neg_inds] / factor)
                    padding_label = labels[i][neg_inds]
                    padding_target = bbox_xyxy_to_cxcywh(targets[i][neg_inds] / factor)
                    padding_feat = feats[neg_inds]
                coord = torch.cat((coord, padding_coord), dim=0)
                label = torch.cat((label, padding_label), dim=0)
                target = torch.cat((target, padding_target), dim=0)
                feat = torch.cat((feat, padding_feat), dim=0)
            if self.query_head.use_zero_padding:
                attn_masks.append(attn_mask.unsqueeze(0))
            aux_coords.append(coord.unsqueeze(0))
            aux_labels.append(label.unsqueeze(0))
            aux_targets.append(target.unsqueeze(0))
            aux_feats.append(feat.unsqueeze(0))

        if self.query_head.use_zero_padding:
            attn_masks = torch.cat(attn_masks, dim=0).unsqueeze(1).repeat(1, 8, 1, 1)
            attn_masks = attn_masks.reshape(bs*8, max_num_coords, max_num_coords)
        else:
            attn_mask = None

        aux_coords = torch.cat(aux_coords, dim=0)
        aux_labels = torch.cat(aux_labels, dim=0)
        aux_targets = torch.cat(aux_targets, dim=0)
        aux_feats = torch.cat(aux_feats, dim=0)
        aux_label_weights = label_weights
        aux_bbox_weights = bbox_weights
        return (aux_coords, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks)
    
    
    
    
    def forward_aux(self, mlvl_feats, img_metas, aux_targets, head_idx):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h). \
                Shape [nb_dec, bs, num_query, 4].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 4). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """
        aux_coords, aux_labels, aux_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks = aux_targets
        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            # img_h, img_w, _ = img_metas[img_id]['img_shape'] 
            img_h, img_w = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        query_embeds = None
        query, inter_references = self.decoder.forward_aux(
                    mlvl_feats,
                    mlvl_masks,
                    query_embeds,
                    mlvl_positional_encodings,
                    aux_coords,
                    pos_feats=aux_feats,
                    reg_branches=self.query_head.reg_branches if self.with_box_refine else None,  # noqa:E501
                    cls_branches=self.query_head.cls_branches if self.as_two_stage else None,  # noqa:E501
                    return_encoder_output=True,
                    attn_masks=attn_masks,
                    head_idx=head_idx
            )

        outputs_classes = []
        outputs_coords = []

        for lvl in range(query.shape[0]):
            reference = inter_references[lvl]
            reference = inverse_sigmoid(reference, eps=1e-3)
            outputs_class = self.query_head.cls_branches[lvl](query[lvl])
            tmp = self.query_head.reg_branches[lvl](query[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        return outputs_classes, outputs_coords, \
                None, None
                
                
    
    def loss_single_aux(self,
                        cls_scores,
                        bbox_preds,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        img_metas,
                        gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        num_q = cls_scores.size(1)
        try:
            labels = labels.reshape(num_imgs * num_q)
            label_weights = label_weights.reshape(num_imgs * num_q)
            bbox_targets = bbox_targets.reshape(num_imgs * num_q, 4)
            bbox_weights = bbox_weights.reshape(num_imgs * num_q, 4)
        except:
            return cls_scores.mean()*0, cls_scores.mean()*0, cls_scores.mean()*0

        bg_class_ind = self.query_head.num_classes
        num_total_pos = len(((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1))
        num_total_neg = num_imgs*num_q - num_total_pos

        # classification loss
        cls_scores = cls_scores.reshape(-1,  self.query_head.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.query_head.bg_cls_weight
        if self.query_head.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        bg_class_ind = self.query_head.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero().squeeze(1)
        scores = label_weights.new_zeros(labels.shape)
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
        pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
        pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
        scores[pos_inds] = bbox_overlaps(
            pos_decode_bbox_pred.detach(),
            pos_decode_bbox_targets,
            is_aligned=True)
        
        # loss_cls = self.query_head.loss_cls(
        #         cls_scores, labels, label_weights, avg_factor=cls_avg_factor)
        loss_cls = self.query_head.loss_cls(
            cls_scores, (labels, scores),
            weight=label_weights,
            avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(img_metas, bbox_preds):
            # img_h, img_w, _ = img_meta['img_shape']
            img_h, img_w = img_meta['img_shape']
            factor = bbox_pred.new_tensor([img_w, img_h, img_w,
                                           img_h]).unsqueeze(0).repeat(
                                               bbox_pred.size(0), 1)
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.query_head.loss_iou(
            bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos)

        # regression L1 loss
        loss_bbox = self.query_head.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos)
        return loss_cls*self.query_head.lambda_1, loss_bbox*self.query_head.lambda_1, loss_iou*self.query_head.lambda_1
