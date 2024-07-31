# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence, Union,Any
from typing import Any, List, Optional, Sequence, Union
from mmengine.logging import print_log
from torch import Tensor
import logging
import numpy as np
import torch
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable
from mmengine.structures import BaseDataElement
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from ..functional import eval_recalls
from .coco_metric import CocoMetric
from mmengine.dist import (broadcast_object_list, collect_results,
                           is_main_process)


@METRICS.register_module()
class BDDMetric(CocoMetric):
    
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        
        self.clear_weather = ['clear','partly cloudy']
        self.good_day = ['daytime']
        
        self.good_condition_results: List[Any] = []
        self.bad_condition_results: List[Any] = []
        # self.loop_back_results: List[Any] = []

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            
            attributes = data_sample['attributes']
            if attributes['weather'] in self.clear_weather or attributes['timeofday'] in self.good_day:
                self.good_condition_results.append((gt, result))
            else:
                self.bad_condition_results.append((gt,result))          
            # add converted result to the results list
            self.results.append((gt, result))
  

    def compute_metrics(self, results: list,good_condition_results: list,bad_condition_results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        all_results = {'good_condition':good_condition_results,'bad_condition':bad_condition_results,'all':results}
        all_eval_results = {}
        
        for key,value in all_results.items():
            
            # logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
            #                     f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
            #                     f'{ap[4]:.3f} {ap[5]:.3f}')
            logger.info('Eval {} mAP .....'.format(key))
            
            # split gt and prediction list
            gts, preds = zip(*value)

            tmp_dir = None
            if self.outfile_prefix is None:
                tmp_dir = tempfile.TemporaryDirectory()
                outfile_prefix = osp.join(tmp_dir.name, 'results')
            else:
                outfile_prefix = self.outfile_prefix

            if self._coco_api is None:
                # use converted gt json file to initialize coco api
                logger.info('Converting ground truth to coco format...')
                coco_json_path = self.gt_to_coco_json(
                    gt_dicts=gts, outfile_prefix=outfile_prefix)
                self._coco_api = COCO(coco_json_path)

            # handle lazy init
            if self.cat_ids is None:
                self.cat_ids = self._coco_api.get_cat_ids(
                    cat_names=self.dataset_meta['classes'])
            
            # if self.img_ids is None:
            #     self.img_ids = self._coco_api.get_img_ids()
            #     # self.img_ids = self.img_ids[:2]
            
            # 计算当前的eval的所有的img_id
            # if self.img_ids is None:
            img_ids = []
            for idx in range(len(gts)):
                img_ids.append(gts[idx]['img_id'])
            self.img_ids = img_ids
            
            print("gts:",len(gts))
            print("img_ids",len(self.img_ids))
            
            assert len(gts)==len(self.img_ids)
            # convert predictions to coco format and dump to json file
            result_files = self.results2json(preds, outfile_prefix)

            eval_results = OrderedDict()
            if self.format_only:
                logger.info('results are saved in '
                            f'{osp.dirname(outfile_prefix)}')
                return eval_results

            for metric in self.metrics:
                logger.info(f'Evaluating {metric}...')

                # TODO: May refactor fast_eval_recall to an independent metric?
                # fast eval recall
                if metric == 'proposal_fast':
                    ar = self.fast_eval_recall(
                        preds, self.proposal_nums, self.iou_thrs, logger=logger)
                    log_msg = []
                    for i, num in enumerate(self.proposal_nums):
                        eval_results[f'AR@{num}'] = ar[i]
                        log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                    log_msg = ''.join(log_msg)
                    logger.info(log_msg)
                    continue

                # evaluate proposal, bbox and segm
                iou_type = 'bbox' if metric == 'proposal' else metric
                if metric not in result_files:
                    raise KeyError(f'{metric} is not in results')
                try:
                    predictions = load(result_files[metric])
                    if iou_type == 'segm':
                        # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                        # When evaluating mask AP, if the results contain bbox,
                        # cocoapi will use the box area instead of the mask area
                        # for calculating the instance area. Though the overall AP
                        # is not affected, this leads to different
                        # small/medium/large mask AP results.
                        for x in predictions:
                            x.pop('bbox')
                    coco_dt = self._coco_api.loadRes(predictions)

                except IndexError:
                    logger.error(
                        'The testing results of the whole dataset is empty.')
                    break

                coco_eval = COCOeval(self._coco_api, coco_dt, iou_type)

                coco_eval.params.catIds = self.cat_ids
                coco_eval.params.imgIds = self.img_ids
                coco_eval.params.maxDets = list(self.proposal_nums)
                coco_eval.params.iouThrs = self.iou_thrs

                # mapping of cocoEval.stats
                coco_metric_names = {
                    'mAP': 0,
                    'mAP_50': 1,
                    'mAP_75': 2,
                    'mAP_s': 3,
                    'mAP_m': 4,
                    'mAP_l': 5,
                    'AR@100': 6,
                    'AR@300': 7,
                    'AR@1000': 8,
                    'AR_s@1000': 9,
                    'AR_m@1000': 10,
                    'AR_l@1000': 11
                }
                metric_items = self.metric_items
                if metric_items is not None:
                    for metric_item in metric_items:
                        if metric_item not in coco_metric_names:
                            raise KeyError(
                                f'metric item "{metric_item}" is not supported')

                if metric == 'proposal':
                    coco_eval.params.useCats = 0
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    if metric_items is None:
                        metric_items = [
                            'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                            'AR_m@1000', 'AR_l@1000'
                        ]

                    for item in metric_items:
                        val = float(
                            f'{coco_eval.stats[coco_metric_names[item]]:.3f}')
                        eval_results[item] = val
                else:
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    coco_eval.summarize()
                    if self.classwise:  # Compute per-category AP
                        # Compute per-category AP
                        # from https://github.com/facebookresearch/detectron2/
                        precisions = coco_eval.eval['precision']
                        # precision: (iou, recall, cls, area range, max dets)
                        assert len(self.cat_ids) == precisions.shape[2]

                        results_per_category = []
                        for idx, cat_id in enumerate(self.cat_ids):
                            t = []
                            # area range index 0: all area ranges
                            # max dets index -1: typically 100 per image
                            nm = self._coco_api.loadCats(cat_id)[0]
                            precision = precisions[:, :, idx, 0, -1]
                            precision = precision[precision > -1]
                            if precision.size:
                                ap = np.mean(precision)
                            else:
                                ap = float('nan')
                            t.append(f'{nm["name"]}')
                            t.append(f'{round(ap, 3)}')
                            eval_results[f'{nm["name"]}_precision'] = round(ap, 3)

                            # indexes of IoU  @50 and @75
                            for iou in [0, 5]:
                                precision = precisions[iou, :, idx, 0, -1]
                                precision = precision[precision > -1]
                                if precision.size:
                                    ap = np.mean(precision)
                                else:
                                    ap = float('nan')
                                t.append(f'{round(ap, 3)}')

                            # indexes of area of small, median and large
                            for area in [1, 2, 3]:
                                precision = precisions[:, :, idx, area, -1]
                                precision = precision[precision > -1]
                                if precision.size:
                                    ap = np.mean(precision)
                                else:
                                    ap = float('nan')
                                t.append(f'{round(ap, 3)}')
                            results_per_category.append(tuple(t))

                        num_columns = len(results_per_category[0])
                        results_flatten = list(
                            itertools.chain(*results_per_category))
                        headers = [
                            'category', 'mAP', 'mAP_50', 'mAP_75', 'mAP_s',
                            'mAP_m', 'mAP_l'
                        ]
                        results_2d = itertools.zip_longest(*[
                            results_flatten[i::num_columns]
                            for i in range(num_columns)
                        ])
                        table_data = [headers]
                        table_data += [result for result in results_2d]
                        table = AsciiTable(table_data)
                        logger.info('\n' + table.table)

                    if metric_items is None:
                        metric_items = [
                            'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                        ]

                    for metric_item in metric_items:
                        key = f'{metric}_{metric_item}'
                        val = coco_eval.stats[coco_metric_names[metric_item]]
                        eval_results[key] = float(f'{round(val, 3)}')

                    ap = coco_eval.stats[:6]
                    logger.info(f'{metric}_mAP_copypaste: {ap[0]:.3f} '
                                f'{ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                                f'{ap[4]:.3f} {ap[5]:.3f}')

            if tmp_dir is not None:
                tmp_dir.cleanup()
            
            all_eval_results[key] = eval_results
            
        return eval_results


    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset. When batch
                size > 1, the dataloader may pad some data samples to make
                sure all ranks have the same length of dataset slice. The
                ``collect_results`` function will drop the padded data based on
                this size.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        if len(self.results) == 0:
            print_log(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.',
                logger='current',
                level=logging.WARNING)

        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
            
            good_condition_results = collect_results(
                self.good_condition_results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
            bad_condition_results = collect_results(
                self.bad_condition_results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            collect_results(self.results, size, self.collect_device)
            collect_results(self.good_condition_results, size, self.collect_device)
            collect_results(self.bad_condition_results, size, self.collect_device)

        if is_main_process():
            # cast all tensors in results list to cpu
            results = _to_cpu(results)
            _metrics = self.compute_metrics(results,good_condition_results,bad_condition_results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        self.good_condition_results.clear()
        self.bad_condition_results.clear()
        return metrics[0]
    
    
def _to_cpu(data: Any) -> Any:
    """transfer all tensors and BaseDataElement to cpu."""
    if isinstance(data, (Tensor, BaseDataElement)):
        return data.to('cpu')
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, tuple):
        return tuple(_to_cpu(d) for d in data)
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data