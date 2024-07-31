from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
import re
# from mmdet.models.adapters.convpass import Convpass,forward_block_attn,forward_dinov2_block_t1,forward_sam_block_t1,Convpass_swin,forward_eva02_block_t1
from mmdet.models.adapters.convpass import *
import mmengine
from mmengine.runner.checkpoint import CheckpointLoader
from mmdet.models.adapters.hom_norm import *

@MODELS.register_module()
class ATSSHomNorm(SingleStageDetector):
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
                 convpass,
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
        
        self.method = 'attn+mlp'
        self.dim = convpass['dim']
        self.scale =  convpass['scale']

        self.apply_hom_norm(self.backbone)
        self._set_convpass_trainable(self.backbone)
            
            
    def apply_hom_norm(self,tuning_module):
        
        if self.method == 'attn+mlp':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Hom_norm(tuning_module.embed_dim,self.dim, False,'xavier').to(layer.norm1.weight.device)
                            layer.adapter_mlp = Hom_norm(tuning_module.embed_dim,self.dim, False,'xavier').to(layer.norm1.weight.device)
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
                        
                            
                            
        if self.method == 'mlp':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_mlp = Convpass_swin(tuning_module.embed_dim,self.num_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
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
                            
        if self.method == 'attn':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin(tuning_module.embed_dim,self.num_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
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
           
           
           
        if self.method == 'all+mlp+attn':
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin(tuning_module.embed_dim, self.num_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass_swin(tuning_module.embed_dim, self.num_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_all = Convpass_swin(tuning_module.embed_dim, self.num_conv,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
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
                            
        if self.method == 'mlp+attn+share':   
            
            shared_conv = nn.Conv2d(self.dim, self.dim, 3, 1, 1)
            if self.xavier_init:
                nn.init.xavier_uniform_(shared_conv.weight)
            else:
                nn.init.zeros_(shared_conv.weight)
                shared_conv.weight.data[:, :, 1, 1] += torch.eye(self.dim, dtype=torch.float)
                nn.init.zeros_(shared_conv.bias)
            
            
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for layer in child_module:
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin_share(shared_conv,tuning_module.embed_dim,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.adapter_mlp = Convpass_swin_share(shared_conv,tuning_module.embed_dim,self.dim, self.xavier_init,self.linear_init).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                                
                                

                        
    def _set_convpass_trainable(self,tuning_module):
        
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
                
        if self.backbone.bias_tune:
            for name, param in tuning_module.named_parameters():
                if '.bias' in name:
                    param.requires_grad = True
            