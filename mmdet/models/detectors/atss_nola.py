# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
from torch import nn
import re
from mmdet.models.adapters.nola import Nola

@MODELS.register_module()
class ATSSNola(SingleStageDetector):
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
                 nola:dict,
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
            num_layer = len(self.backbone.blocks)
        else:
            self.backbone_embed_dim = self.backbone.embed_dims
        
        
        if nola is not None:
            assert type(nola['targets'])==list
            self.targets = nola['targets']
            
            self.rank = nola['rank']
            self.scaling = nola['scaling']
            self.drop_rate = nola['drop_rate']
            
            self.down_l = nola['down_l']  # down矩阵由几个随机矩阵所组成
            self.up_l = nola['up_l']  # up矩阵由几个随机矩阵所组成
            
            
            self.apply_nola(self.backbone)
            self._set_nola_trainable(self.backbone)
            

    def apply_nola(self,tuning_module):
        """Apply LoRA to target layers."""
        '''
            tuning_module:被替换的部分
        '''
  
            
        random_down_shape_lin1 = (self.down_l,self.backbone_embed_dim,self.rank)
        random_up_shape_lin1 =  (self.down_l,self.rank,self.backbone_embed_dim*4)
        
        random_down_shape_lin2 = (self.down_l,self.backbone_embed_dim*4,self.rank)
        random_up_shape_lin2 = (self.up_l,self.rank,self.backbone_embed_dim)
        
    
        self.random_dwon_lin1 = torch.randn(random_down_shape_lin1).cuda()
        self.random_up_ln1 = torch.randn(random_up_shape_lin1).cuda()
        
        self.random_dwon_lin2 = torch.randn(random_down_shape_lin2).cuda()
        self.random_up_ln2 = torch.randn(random_up_shape_lin2).cuda()
        
        # self.random_dwon_lin1.requires_grad = False
        # self.random_up_ln1.requires_grad = False
        
        # self.random_dwon_lin2.requires_grad = False
        # self.random_up_ln2.requires_grad = False
        
        # self.nola_up_scale_ln1 = nn.Parameter(torch.randn((self.up_l)))
        # self.nola_down_scale_ln1 = nn.Parameter(torch.randn((self.down_l)))
        
        # self.nola_up_scale_ln2 = nn.Parameter(torch.randn((self.up_l)))
        # self.nola_down_scale_ln2 = nn.Parameter(torch.randn((self.down_l)))
        
        # nn.init.xavier_uniform_(self.nola_up_scale_ln1.unsqueeze(0))
        # nn.init.xavier_uniform_(self.nola_down_scale_ln1.unsqueeze(0))
        # nn.init.xavier_uniform_(self.nola_up_scale_ln2.unsqueeze(0))
        # nn.init.xavier_uniform_(self.nola_down_scale_ln2.unsqueeze(0))
        
        
        module_names = [k for k, _ in tuning_module.named_modules()]
        for module_name in module_names:
            for target in self.targets:
                target_name = target['type']
                target_scaling = target.get('scaling', self.scaling)
                # target_rank = target.get('rank', self.rank)
                target_drop_rate = target.get('drop_rate', self.drop_rate)
                target_down_l = target.get('down_l', self.drop_rate)
                target_up_l = target.get('up_l', self.drop_rate)

                if re.fullmatch(target_name, module_name) or \
                        module_name.endswith(target_name):
                    current_module = tuning_module.get_submodule(module_name)
                    
                    for name,_child_module in current_module.named_modules():
                        
                        if  _child_module.__class__.__name__ == "Linear":
                            full_module_name = module_name+'.'+name
                            print_log(
                            f'Set NOLA for {full_module_name} '
                            f'with scaling: {target_scaling}, '
                            f'rank: {self.rank}, '
                            f'drop rate: {target_drop_rate},'
                            f'down_l: {target_down_l},'
                            f'up_l: {target_up_l}',
                            logger='current')
                            self._replace_module(tuning_module,full_module_name, _child_module,
                                                        target_scaling, self.rank,
                                                        target_drop_rate)

        self.backbone._register_load_state_dict_pre_hook(self._load_original_layer_weight)
          
    def _replace_module(self, tuning_module,full_module_name: str, _child_module: nn.Module,
                        scaling: int, rank: int, drop_rate: float):
        """Replace target layer with LoRA linear layer in place."""
        # for name,_child_module in current_module.named_modules():
        #     if _child_module.__class__.__name__ == "Linear":
        target_name = full_module_name.split('.')[-1]
        if target_name=='lin1':
            target_module = Nola(_child_module, self.random_dwon_lin1, self.random_up_ln1,drop_rate,scaling).to(_child_module.weight.device)
        else:
            target_module = Nola(_child_module, self.random_dwon_lin2, self.random_up_ln2,drop_rate,scaling).to(_child_module.weight.device)
            
        # target_module = LoRALinear(_child_module, alpha, rank, drop_rate)
        
        parent_module_name = '.'.join(full_module_name.split('.')[:-1])
        
        parent_module = tuning_module.get_submodule(parent_module_name)
        
        setattr(parent_module, target_name, target_module)
        
    def _set_nola_trainable(self,tuning_module):
        """Set only the lora parameters trainable."""
        for name, param in tuning_module.named_parameters():
            if '.nola' not in name:
                param.requires_grad = False
                
    def _load_original_layer_weight(self, state_dict, prefix, *args, **kwargs):
        
        state_dict_key_list = list(state_dict.keys())
        
        for name in state_dict_key_list:
            if 'lin' in name:
                index = name.find('lin') + 4    # +4是因为index的位置是起始位置
                new_name = name[:index] + '.original_layer' + name[index:]
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

        
        