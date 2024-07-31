# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
from torch import nn
import re
from mmdet.models.adapters.conv_lora_new import LoRALinear
from mmengine.runner.checkpoint import CheckpointLoader
from copy import deepcopy


@MODELS.register_module()
class ATSS_Conv_Lora(SingleStageDetector):
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
                 lora:dict,
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
        
        
        
        # ------------------------  Tuning -------------------------
        self.lora_enable = False
        
        
        if lora is not None:
            assert type(lora['targets'])==list
            self.lora_enable = True
            self.targets = lora['targets']
            
            self.rank = lora['rank']
            self.alpha = lora['alpha']
            self.drop_rate = lora['drop_rate']
            
            self.apply_lora(self.backbone)
            self._set_lora_trainable(self.backbone)
            

    def apply_lora(self,tuning_module):
        """Apply LoRA to target layers."""
        '''
            tuning_module:被替换的部分
        '''
        module_names = [k for k, _ in tuning_module.named_modules()]
        for module_name in module_names:
            for target in self.targets:
                target_name = target['type']
                target_alpha = target.get('alpha', self.alpha)
                target_rank = target.get('rank', self.rank)
                target_drop_rate = target.get('drop_rate', self.drop_rate)

                if re.fullmatch(target_name, module_name) or \
                        module_name.endswith(target_name):
                    current_module = tuning_module.get_submodule(module_name)
                    
                    for name,_child_module in current_module.named_modules():
                        
                        if  _child_module.__class__.__name__ == "Linear":
                            full_module_name = module_name+'.'+name
                            print_log(
                            f'Set LoRA for {full_module_name} '
                            f'with alpha: {target_alpha}, '
                            f'rank: {target_rank}, '
                            f'drop rate: {target_drop_rate}',
                            logger='current')
                            self._replace_module(tuning_module,full_module_name, _child_module,
                                                        target_alpha, target_rank,
                                                        target_drop_rate)

        self.backbone._register_load_state_dict_pre_hook(self._load_original_layer_weight)
          
    def _replace_module(self, tuning_module,full_module_name: str, _child_module: nn.Module,
                        alpha: int, rank: int, drop_rate: float):
        """Replace target layer with LoRA linear layer in place."""
        # for name,_child_module in current_module.named_modules():
        #     if _child_module.__class__.__name__ == "Linear":
            
        target_module = LoRALinear(_child_module, alpha, rank, drop_rate).to(_child_module.weight.device)
        # target_module = LoRALinear(_child_module, alpha, rank, drop_rate)
        
        target_name = full_module_name.split('.')[-1]
        parent_module_name = '.'.join(full_module_name.split('.')[:-1])
        
        parent_module = tuning_module.get_submodule(parent_module_name)
        
        setattr(parent_module, target_name, target_module)
        
    def _set_lora_trainable(self,tuning_module):
        """Set only the lora parameters trainable."""
        for name, param in tuning_module.named_parameters():
            param.requires_grad = False

        for name, param in tuning_module.named_parameters():
            if '.lora_' in name:
                param.requires_grad = True
            if 'expert' in name:
                param.requires_grad = True
            if 'gate' in name:
                param.requires_grad = True

    def _load_original_layer_weight(self, state_dict, prefix, *args, **kwargs):
        
        state_dict_key_list = list(state_dict.keys())
        
        if self.backbone.__class__.__name__ == 'SAMImageEncoderViTv3':
            
            for name in state_dict_key_list:
                if 'lin' in name:
                    index = name.find('lin') + 4    # +4是因为index的位置是起始位置
                    new_name = name[:index] + '.original_layer' + name[index:]
                    state_dict[new_name] = state_dict[name]
                    print_log(
                                f'change {name} '
                                f'to: {new_name},',
                                logger='current')
                    
        if self.backbone.__class__.__name__ == 'Dinov2VisionTransformer_Conv_Lora':
            
            for name in state_dict_key_list:
                
                if 'ffn.layers.0.0' in name or 'ffn.layers.1' in name:
                    final_word = name.split('.')[-1]
                    split_name = '.'.join(name.split('.')[:-1])
                    new_name = split_name + '.original_layer.' + final_word
                    state_dict[new_name] = state_dict[name]
                    print_log(
                                f'change {name} '
                                f'to: {new_name},',
                                logger='current')
                if 'attn.qkv' in name or 'attn.proj' in name:
                    final_word = name.split('.')[-1]
                    split_name = '.'.join(name.split('.')[:-1])
                    new_name = split_name + '.original_layer.' + final_word
                    state_dict[new_name] = state_dict[name]
                    print_log(
                                f'change {name} '
                                f'to: {new_name},',
                                logger='current')
                    
                    
                    
        if self.backbone.__class__.__name__ == 'SwinTransformer':
            
            for name in state_dict_key_list:
                
                if 'ffn.layers' in name:
                    
                    final_word = name.split('.')[-1]
                    split_name = '.'.join(name.split('.')[:-1])
                    new_name = split_name + '.original_layer.' + final_word
                    state_dict[new_name] = state_dict[name]
                    print_log(
                                f'change {name} '
                                f'to: {new_name},',
                                logger='current')
                    
                    
                    
            
            
    
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

        
        