# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector
import mmengine

from mmdet.models.adapters.new_lorand import LoRandv2,forward_swin_block_lorand,forward_swin_block_lorandv2

@MODELS.register_module()
class CascadeRCNN_lorand(TwoStageDetector):
    r"""Implementation of `Cascade R-CNN: Delving into High Quality Object
    Detection <https://arxiv.org/abs/1906.09756>`_"""

    def __init__(self,
                 lorand,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)


        self.num_branch = lorand['num_branch']    
        self.kernel_dim = lorand['kernel_dim']
        self.factor = lorand['factor']
        self.after_ffn = lorand['after_ffn'] 

        self.apply_lorand(self.backbone)
        self._set_lorand_trainable(self.backbone)
        
    def apply_lorand(self,tuning_module):
        
        for child_module in tuning_module.children():
            if type(child_module) == mmengine.model.base_module.ModuleList:
                for idx,swin_block_sequence in enumerate(child_module):
                    for idx,swin_block in enumerate(swin_block_sequence.blocks):
                        swin_block.adapter_lorand1 = LoRandv2(swin_block.attn.w_msa.embed_dims,swin_block.attn.w_msa.embed_dims//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(swin_block.norm1.weight.device)
                        swin_block.adapter_lorand2 = LoRandv2(swin_block.ffn.embed_dims,swin_block.ffn.embed_dims//self.factor,num_branch=self.num_branch, kernel_dim=self.kernel_dim).to(swin_block.norm1.weight.device)
                        if self.after_ffn:
                            bound_method = forward_swin_block_lorandv2.__get__(swin_block, swin_block.__class__)
                        else:
                            bound_method = forward_swin_block_lorand.__get__(swin_block, swin_block.__class__)
                        setattr(swin_block, 'forward', bound_method)
                 


    def _set_lorand_trainable(self,tuning_module):
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
                
                
            # if name=='norm0.weight' or name=='norm0.bias' \
            #         or name=='norm1.weight' or name=='norm1.bias' \
            #         or name=='norm2.weight' or name=='norm2.bias' \
            #         or name=='norm3.weight' or name=='norm3.bias':
            #     param.requires_grad = True
            if 'norm' in name:
                param.requires_grad = True