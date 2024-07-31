from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
import re
# from mmdet.models.adapters.convpass import Convpass,forward_block_attn,forward_dinov2_block_t1,forward_sam_block_t1,Convpass_swin,forward_eva02_block_t1
from mmdet.models.adapters.conv_lora import *
from mmdet.models.adapters.lorav2 import LoRALinearv2
import mmengine
from mmengine.runner.checkpoint import CheckpointLoader

@MODELS.register_module()
class ATSSConvPassLora(SingleStageDetector):
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
                 adapter_cfg,
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
        
        if adapter_cfg is not None:
            # assert type(convpass['targets'])==list
            self.adapter_type = adapter_cfg['type']
            self.lora_cfg = adapter_cfg['lora_cfg'] 
            # self.targets = convpass['targets']
            self.convpass = adapter_cfg['convpass']
            
            self.dim =  self.convpass['dim']
            self.scale =  self.convpass['scale']
            self.xavier_init =  self.convpass['xavier_init']
            self.linear_init =  self.convpass['linear_init']
            self.dw_conv = self.convpass['dw_conv']
            
            
            self.apply_convpassLora(self.backbone)
            self._set_convpass_trainable(self.backbone)
            
            
    def apply_convpassLora(self,tuning_module):
        
        if self.adapter_type == 'type1':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin(tuning_module.embed_dim,self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass_swin(tuning_module.embed_dim,self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.loralinear = LoRALinearv2(tuning_module.embed_dim,**self.lora_cfg).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                                                
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
            
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformerVPT':
                            layer.adapter_attn = ConvpassVPT(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_mlp = ConvpassVPT(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_vpt.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                                    
                        if tuning_module.__class__.__name__ == 'ViTEVA02VPT':
                            layer.adapter_attn = ConvpassVPT(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = ConvpassVPT(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_vpt.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3VPT':
                            layer.adapter_attn = ConvpassVPT_swin(tuning_module.embed_dim,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = ConvpassVPT_swin(tuning_module.embed_dim,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_vpt.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                            
                            
        if self.adapter_type == 'type2':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin(tuning_module.embed_dim,self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.loralinear = LoRALinearv2(tuning_module.embed_dim,**self.lora_cfg).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t2.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t2.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                                                
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t2.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
        if self.adapter_type == 'type3':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_mlp = Convpass_swin(tuning_module.embed_dim,self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.loralinear = LoRALinearv2(tuning_module.embed_dim,**self.lora_cfg).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t3.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t3.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                                                
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t3.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
           
           
           
        if self.adapter_type == 'type4':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_all = Convpass_swin(tuning_module.embed_dim, self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.loralinear = LoRALinearv2(tuning_module.embed_dim,**self.lora_cfg).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_all = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                                                
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_all = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
        if self.adapter_type == 'type5':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin(tuning_module.embed_dim,self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass_swin(tuning_module.embed_dim,self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.loralinear = LoRALinearv2(tuning_module.embed_dim,**self.lora_cfg).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t5.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_all = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                                                
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_all = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
        if self.adapter_type == 'type6':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin(tuning_module.embed_dim,self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass_swin(tuning_module.embed_dim,self.dw_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.loralinear = LoRALinearv2(tuning_module.embed_dim,**self.lora_cfg).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t6.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.adapter_all = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.ln1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                                                
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            layer.adapter_attn = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_all = Convpass(tuning_module.embed_dims,tuning_module.patch_resolution,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        
                        
    def _set_convpass_trainable(self,tuning_module):
        
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
                
        if self.backbone.bias_tune:
            for name, param in tuning_module.named_parameters():
                if '.bias' in name:
                    param.requires_grad = True
            