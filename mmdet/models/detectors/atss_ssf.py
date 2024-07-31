# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
from torch import nn
import re
from mmdet.models.adapters.ssf import SSF_adapter
from mmengine.runner.checkpoint import CheckpointLoader
from copy import deepcopy


@MODELS.register_module()
class ATSSSSF(SingleStageDetector):
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
            
                
            
        
        
        # if self.backbone.__class__.__name__ == 'SAMImageEncoderViTv3':
            
        #     for name in state_dict_key_list:
        #         if 'lin' in name:
        #             index = name.find('lin') + 4    # +4是因为index的位置是起始位置
        #             new_name = name[:index] + '.original_module' + name[index:]
        #             state_dict[new_name] = state_dict[name]
        #             print_log(
        #                         f'change {name} '
        #                         f'to: {new_name},',
        #                         logger='current')
                    
        # if self.backbone.__class__.__name__ == 'Dinov2VisionTransformer':
            
        #     for name in state_dict_key_list:
                
        #         if 'ffn.layers.0.0' in name or 'ffn.layers.1' in name:
        #             final_word = name.split('.')[-1]
        #             split_name = '.'.join(name.split('.')[:-1])
        #             new_name = split_name + '.original_module.' + final_word
        #             state_dict[new_name] = state_dict[name]
        #             print_log(
        #                         f'change {name} '
        #                         f'to: {new_name},',
        #                         logger='current')
                    
        # if self.backbone.__class__.__name__ == 'ViTEVA02':
            
        #     for name in state_dict_key_list:
                
        #         if 'mlp.w12' in name or 'mlp.w3' in name:
        #             final_word = name.split('.')[-1]
        #             split_name = '.'.join(name.split('.')[:-1])
        #             new_name = split_name + '.original_module.' + final_word
        #             state_dict[new_name] = state_dict[name]
        #             print_log(
        #                         f'change {name} '
        #                         f'to: {new_name},',
        #                         logger='current')
                    
                    
                    
                    
                    
            
            
    
    # def init_weights(self):
    #     super().init_weights()
        
    #     # #  先load 
    #     # if self.backbone.init_cfg is not None:
    #     #     ckpt_path = self.backbone.init_cfg['checkpoint']
    #     #     ckpt = CheckpointLoader.load_checkpoint(
    #     #        ckpt_path , map_location='cpu')
    #     #     if 'state_dict' in ckpt:
    #     #         _state_dict = ckpt['state_dict']
    #     #     self.load_state_dict(_state_dict, False)
        
        
        
    #     if self.lora_enable:
    #         self.apply_lora(self.backbone)
    #         # self._set_lora_trainable(self.backbone)

        
        