# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
from torch import nn
import re
from mmdet.models.adapters.lora_fft import LoRAFFT, LoRAFFTv2,forward_sam_block_lorafft_mlp 
from mmengine.runner.checkpoint import CheckpointLoader
from copy import deepcopy
import mmengine



@MODELS.register_module()
class ATSSLoraFFTv2(SingleStageDetector):
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
                 lorafft:dict,
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
    
        
        if self.backbone.__class__.__name__ == 'SAMImageEncoderViTv3':
            self.backbone_embed_dim = self.backbone.embed_dim
        else:
            self.backbone_embed_dim = self.backbone.embed_dims
            
        self.method = lorafft['method']
        self.rank = lorafft['rank']
        self.alpha = lorafft['alpha']
        self.drop_rate = lorafft['drop_rate']
        self.dcn_stage = lorafft['dcn_stage']
        
        self.apply_lora(self.backbone)
        self._set_lora_trainable(self.backbone)
            

    def apply_lora(self,tuning_module):
        """Apply LoRA to target layers."""
        '''
            tuning_module:被替换的部分
        '''
                            
        if self.method == 'mlp':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_mlp = LoRAFFTv2(self.backbone_embed_dim,self.rank, self.drop_rate,self.dcn_stage).to(layer.norm1.weight.device)
                            layer.s = self.alpha / self.rank
                            bound_method = forward_sam_block_lorafft_mlp.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            layer.adapter_mlp = LoRAFFTv2(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t2.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                                                
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            layer.adapter_mlp = LoRAFFTv2(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t2.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
        
    def _set_lora_trainable(self,tuning_module):
        """Set only the lora parameters trainable."""
        for name, param in tuning_module.named_parameters():
            if '.lora_' not in name:
                param.requires_grad = False
                
                
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
                    
        if self.backbone.__class__.__name__ == 'Dinov2VisionTransformer':
            
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
                    
        if self.backbone.__class__.__name__ == 'ViTEVA02':
            
            for name in state_dict_key_list:
                
                if 'mlp.w12' in name or 'mlp.w3' in name:
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

        
        