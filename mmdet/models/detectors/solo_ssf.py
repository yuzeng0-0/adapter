# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage_instance_seg import SingleStageInstanceSegmentor

import torch
from mmengine.logging import print_log
from torch import nn
import re
from mmdet.models.adapters.ssf import SSF_adapter


@MODELS.register_module()
class SOLOSSF(SingleStageInstanceSegmentor):
    """`SOLO: Segmenting Objects by Locations
    <https://arxiv.org/abs/1912.04488>`_

    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 mask_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        
        
        
        self.changed_module_names = []
    
        self.apply_ssf(self.backbone)
        self._set_lora_trainable(self.backbone)
            

    def apply_ssf(self,tuning_module):
        """Apply LoRA to target layers."""
        '''
            tuning_module:被替换的部分
        '''
        # module_names = [k for k, _ in tuning_module.named_modules()]
        # for module_name in module_names:


            # current_module = tuning_module.get_submodule(module_name)
            
        for name,_child_module in tuning_module.named_modules():
            
            if  _child_module.__class__.__name__ == "Linear" or _child_module.__class__.__name__ == "LayerNorm":
                
                print_log(
                f'Set SSF for {name} ',
                logger='current')
                
                self._replace_module(tuning_module,name, _child_module)
                self.changed_module_names.append(name)
                
            if name ==  "patch_embed.projection":
                
                print_log(
                f'Set SSF for {name} ',
                logger='current')
                
                self._replace_module(tuning_module,name, _child_module)
                self.changed_module_names.append(name)
                
        
        


        self.backbone._register_load_state_dict_pre_hook(self._load_original_layer_weight)
          
    def _replace_module(self, tuning_module,full_module_name: str, _child_module: nn.Module):
        """Replace target layer with LoRA linear layer in place."""
        # for name,_child_module in current_module.named_modules():
        #     if _child_module.__class__.__name__ == "Linear":
            
        target_module = SSF_adapter(_child_module).to(_child_module.weight.device)
        
        target_name = full_module_name.split('.')[-1]
        parent_module_name = '.'.join(full_module_name.split('.')[:-1])
        
        parent_module = tuning_module.get_submodule(parent_module_name)
        
        setattr(parent_module, target_name, target_module)
        
    def _set_lora_trainable(self,tuning_module):
        """Set only the lora parameters trainable."""
        for name, param in tuning_module.named_parameters():
            param.requires_grad = False

            if 'scale' in name or 'shift' in name:
                param.requires_grad = True
                
            
                
                
    def _load_original_layer_weight(self, state_dict, prefix, *args, **kwargs):
        
        state_dict_key_list = list(state_dict.keys())
        
        
        for changed_name in self.changed_module_names:
            
            old_name_weight = changed_name + '.weight'
            if old_name_weight in state_dict_key_list:
                new_name_weight = changed_name + '.original_module' + '.weight'
                
                print_log(
                            f'change {old_name_weight} '
                            f'to: {new_name_weight},',
                            logger='current')
            
                state_dict[new_name_weight] = state_dict[old_name_weight]
                
            old_name_bias = changed_name + '.bias'
            if old_name_bias in state_dict_key_list:
                new_name_bias = changed_name + '.original_module' + '.bias'
                
                
                print_log(
                            f'change {old_name_bias} '
                            f'to: {new_name_bias},',
                            logger='current')
                
                state_dict[new_name_bias] = state_dict[old_name_bias]