# Copyright (c) OpenMMLab. All rights reserved.

from mmcv.transforms import LoadImageFromFile

from mmdet.datasets.transforms import LoadAnnotations, LoadPanopticAnnotations
from mmdet.registry import TRANSFORMS


def get_loading_pipeline(pipeline):
    """Only keep loading image and annotations related configuration.

    Args:
        pipeline (list[dict]): Data pipeline configs.

    Returns:
        list[dict]: The new pipeline list with only keep
            loading image and annotations related configuration.

    Examples:
        >>> pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True),
        ...    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
        ...    dict(type='RandomFlip', flip_ratio=0.5),
        ...    dict(type='Normalize', **img_norm_cfg),
        ...    dict(type='Pad', size_divisor=32),
        ...    dict(type='DefaultFormatBundle'),
        ...    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ...    ]
        >>> expected_pipelines = [
        ...    dict(type='LoadImageFromFile'),
        ...    dict(type='LoadAnnotations', with_bbox=True)
        ...    ]
        >>> assert expected_pipelines ==\
        ...        get_loading_pipeline(pipelines)
    """
    loading_pipeline_cfg = []
    for cfg in pipeline:
        obj_cls = TRANSFORMS.get(cfg['type'])
        # TODO：use more elegant way to distinguish loading modules
        if obj_cls is not None and obj_cls in (LoadImageFromFile,
                                               LoadAnnotations,
                                               LoadPanopticAnnotations):
            loading_pipeline_cfg.append(cfg)
    assert len(loading_pipeline_cfg) == 2, \
        'The data pipeline in your config file must include ' \
        'loading image and annotations related pipeline.'
    return loading_pipeline_cfg



"""Dataset filtering utils."""
from typing import Any, Dict, List, Optional, Union


def _check_attributes(
    attributes: Union[bool, float, str],
    allowed_attributes: Union[bool, float, str, List[float], List[str]],
) -> bool:
    """Check if attributes are allowed.
    Args:
        attributes: Attributes of current frame.
        allowed_attributes: Attributes allowed.
    Return:
        boolean, whether frame attributes are allowed.
    """
    if isinstance(allowed_attributes, list):
        # assert frame_attributes not in allowed_attributes
        return attributes in allowed_attributes
    return attributes == allowed_attributes


def check_attributes(attributes, allowed_attributes=None):
    """Check if a dictionary of attributes is allowed.
    Args:
        attributes (Dict[str, str]): attributes to check
        allowed_attributes (Dict[str, List[str]]): allowed attributes
    Return:
        boolean, whether frame attributes are allowed.
    """
    check = True
    if allowed_attributes:
        for key in allowed_attributes:
            allowed_attribute = allowed_attributes[key]
            check = check and _check_attributes(
                attributes[key], allowed_attribute
            )
            if not check:
                return check            
    return check