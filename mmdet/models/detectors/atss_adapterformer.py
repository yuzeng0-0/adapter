# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmdet.models.adapters.adapter_former import AdapterFormer
from mmdet.models.adapters.base_adapter import inject_adapter
import re
from mmengine.logging import print_log
from torch import nn
from copy import deepcopy

@MODELS.register_module()
class ATSSAdapterFormer(SingleStageDetector):
    """Implementation of `ATSS <https://arxiv.org/abs/1912.02424>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of ATSS. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of ATSS. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 adapter_cfg:dict = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        

        if adapter_cfg is not None:
            self.adapter_former = True
            self.targets = adapter_cfg['targets']
            # self.apply_adapter_former(self.backbone)
            
            self._set_adapter_trainable(self.backbone)
            
            
    def apply_adapter_former(self,tuning_module):
        """Apply LoRA to target layers."""
        '''
            tuning_module:被替换的部分
        '''
        module_names = [k for k, _ in tuning_module.named_modules()]
        for module_name in module_names:
            for target in self.targets:
                target_name = target['type']
                target_param_cfg = deepcopy(target)
                del target_param_cfg['type']
                
                if type(self.backbone).__name__=='SAMImageEncoderViTv3':
                    target_param_cfg['embed_dims'] = self.backbone.embed_dim
                else:
                    target_param_cfg['embed_dims'] = self.backbone.embed_dims
                    
                if re.fullmatch(target_name, module_name) or \
                        module_name.endswith(target_name):
                    '''
                        因为dinov2 里面的layernorm和后面的mlp并不在同一个模块里面,因此将layernorm和mlp写在一个nn.Sequential中作为origin module传入adapter中
                        将原本的layernorm替换为nn.identity,将原本的ffn替换为adapter,相当于ln的操作放到了adapter中做
                    '''
                    current_ffn_module = tuning_module.get_submodule(module_name)
                    if type(self.backbone).__name__=='SAMImageEncoderViTv3':
                        ln_name = '.'.join(module_name.split('.')[:-1])+'.'+'norm2'
                    else:
                        ln_name = '.'.join(module_name.split('.')[:-1])+'.'+'ln2'
                    current_ln_module = tuning_module.get_submodule(ln_name)
                    original_module = nn.Sequential(current_ln_module,current_ffn_module)
                    
                    self._replace_module(tuning_module,module_name,original_module,target_param_cfg)
                            
                            
    def _replace_module(self, tuning_module,module_name,original_module,target_param_cfg):
        """Replace target layer with LoRA linear layer in place."""
        # for name,_child_module in current_module.named_modules():
        #     if _child_module.__class__.__name__ == "Linear":
            
        target_ffn_module = AdapterFormer(original_layer=original_module,**target_param_cfg).to(original_module[0].weight.device)
        # target_module = LoRALinear(_child_module, alpha, rank, drop_rate)
        
        target_ffn_name = module_name.split('.')[-1]
        if type(self.backbone).__name__=='SAMImageEncoderViTv3':
            target_ln_name = 'norm2'
        else:
            target_ln_name = 'ln2'
        parent_module_name = '.'.join(module_name.split('.')[:-1])
        
        parent_module = tuning_module.get_submodule(parent_module_name)
        
        target_ln_module = nn.Identity()
        
        setattr(parent_module, target_ffn_name, target_ffn_module)
        setattr(parent_module, target_ln_name, target_ln_module)
        
        
    def _set_adapter_trainable(self,tuning_module):
        """Set only the lora parameters trainable."""
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
   
        
        
    def init_weights(self):
        super().init_weights()
           
        if self.adapter_former:
            self.apply_adapter_former(self.backbone)
            # self._set_lora_trainable(self.backbone)
