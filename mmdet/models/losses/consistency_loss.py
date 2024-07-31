import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.nn import CrossEntropyLoss

# from shift_tta.registry import MODELS
from mmengine.registry import MODELS

@MODELS.register_module()
class ConsistencyLoss(nn.Module):
    """YOLOXConsistencyLoss
    Args:
        weight (float, optional): Weight of the loss. Default to 1.0.
        obj_weight (float, optional): Weight of the objectness consistency loss.
            Default to 1.0.
        reg_weight (float, optional): Weight of the regression consistency loss.
            Default to 1.0.
        cls_weight (float, optional): Weight of the classification consistency loss.
            Default to 1.0.
    """

    def __init__(self,
                #  weight=1.0,
                #  reg_weight=1.0,
                #  cls_weight=1.0,
                #  only_last_layer = True,
                loss_l1 = None,
        ):
        super(ConsistencyLoss, self).__init__()
        # self.weight = weight
        # self.reg_weight = reg_weight
        # self.cls_weight = cls_weight
        # self.only_last_layer = only_last_layer
        self.loss_l1 = MODELS.build(loss_l1)

    
    def forward(self, stu_out, tea_out, **kwargs):

        """Forward pass.
        Args:
            inputs: Dictionary of classification scores and bounding box
            refinements for the sampled proposals. For cls scores, the shape is
            (b*n) * (cats + 1), where n is sampled proposal in each image, cats
            is the total number of categories without the background. For bbox
            preds, the shape is (b*n) * (4*cats)
            targets: Same output by bbox_head from the teacher output.
        Returns:
            The YOLOX consistency loss.
        """
        # num_query = teacher_reg.shape[1]
        stu_hidden_states = stu_out['hidden_states']
        stu_references = stu_out['references']
        
        tea_hidden_states = tea_out['hidden_states']
        tea_references = tea_out['references']
        
        loss_hidden = self.loss_l1(stu_hidden_states,tea_hidden_states)
        loss_ref =  self.loss_l1(stu_references,tea_references)

        return (loss_hidden + loss_ref)/2