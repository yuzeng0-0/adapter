from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
import re
# from mmdet.models.adapters.convpass import Convpass,forward_block_attn,forward_dinov2_block,forward_sam_block,Convpass_swin,forward_eva02_block
from mmdet.models.adapters.linear_sacle import LinearScale
import mmengine
from mmengine.runner.checkpoint import CheckpointLoader


@MODELS.register_module()
class ATSSLinearScale(SingleStageDetector):
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
                 linear_scale,
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
        
        if linear_scale is not None:

            self.apply_linearscale(self.backbone)
            self.backbone._register_load_state_dict_pre_hook(self._prepare_linear_scale)
            
            self._set_linearscale_trainable(self.backbone)
            
            
    def apply_linearscale(self,tuning_module):
        

        for child_module in tuning_module.children():
            if type(child_module) == mmengine.model.base_module.ModuleList:
                for layer in child_module:
                    module_names = [k for k, _ in layer.named_modules()]
                    modules = [k for _, k in layer.named_modules()]
                    
                    for module_name,module in zip(module_names,modules):   
                        if  module.__class__.__name__ == "Linear":
                            linearscale = LinearScale(module)
                            parent_module_name = '.'.join(module_name.split('.')[:-1])
                            parent_module = layer.get_submodule(parent_module_name)
                            target_name = module_name.split('.')[-1]
                            setattr(parent_module, target_name, linearscale)
                    
    def _prepare_linear_scale(self,state_dict, prefix, *args, **kwargs):
        
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        for name,_ in self.backbone.named_parameters():
            if 'origin_layer' in name:
                
                name_parts  = name.split('.')
                ori_name = '.'.join(name_parts[:-2]) + '.'+name_parts[-1]
                
                if ori_name in state_dict.keys():
                    # print("1111")
                    
                    logger.info(
                        f'Replace {ori_name} '
                        f'to {name}.')
                    state_dict[name] = state_dict[ori_name]
        
        
    def _set_linearscale_trainable(self,tuning_module):
        
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
                
        if self.backbone.bias_tune:
            for name, param in tuning_module.named_parameters():
                if '.bias' in name:
                    param.requires_grad = True
            