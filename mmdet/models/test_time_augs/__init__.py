# Copyright (c) OpenMMLab. All rights reserved.
from .det_tta import DetTTAModel
from .merge_augs import (merge_aug_bboxes, merge_aug_masks,
                         merge_aug_proposals, merge_aug_results,
                         merge_aug_scores)
from .det_tent import DetTentModel
from .det_cotta import DetCoTTAModel

from .my_det_tta import MyDetTTA
from .ms_det_tta import MSDetTTAModel

__all__ = [
    'merge_aug_bboxes', 'merge_aug_masks', 'merge_aug_proposals',
    'merge_aug_scores', 'merge_aug_results', 'DetTTAModel','DetTentModel','DetCoTTAModel','MyDetTTA','MSDetTTAModel'
]
