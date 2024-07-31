# Copyright (c) OpenMMLab. All rights reserved.
from .det_inferencer import DetInferencer
from .inference import (async_inference_detector, inference_detector,
                        init_detector)

from .tta_test import single_gpu_tent

__all__ = [
    'init_detector', 'async_inference_detector', 'inference_detector',
    'DetInferencer','single_gpu_tent'
]
