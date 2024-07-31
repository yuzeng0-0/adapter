from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch
from mmengine.logging import print_log
import re
import mmengine
from mmengine.runner.checkpoint import CheckpointLoader
from mmdet.models.adapters.hypernet_convpass_metanet import Convpass_swin_hypernet_fuse,HyperNetwork_conv,Convpass_swin_hypernet_fusev2,Convpass_swin_hypernet_fusev3,Convpass_swin_hypernet_fuse_pos,Convpass_swin_hypernet_fusev2_pos,Convpass_swin_hypernet_fusev3_pos
from mmdet.models.adapters.convpass import forward_sam_block_t1,forward_dinov2_block_t1,forward_eva02_block_t1
import torch.nn as nn

@MODELS.register_module()
class ATSSHyperConvPass_metanet_pos(SingleStageDetector):
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
        
        
        if self.backbone.__class__.__name__ == 'SAMImageEncoderViTv3':
            self.backbone_embed_dim = self.backbone.embed_dim
            num_layer = len(self.backbone.blocks)
        else:
            self.backbone_embed_dim = self.backbone.embed_dims
            num_layer = len(self.backbone.layers)
            
        self.method = convpass['method']
        
        self.dim = convpass['dim']
        self.scale = convpass['scale']
        self.xavier_init = convpass['xavier_init']
        self.layer_embedding_size = convpass['layer_embedding_size']
        self.fuse_type = convpass['fuse_type']
    
        self.layer_embedding = nn.Embedding(num_layer, self.layer_embedding_size).cuda()
        
        if self.fuse_type == 'only_metafeat':
            self.layer_embedding.requires_grad = False

            
        self.init_type = convpass['init_type']
        self.hyper_linear = convpass['hyper_linear']
        
        if self.fuse_type=='cat':
            self.layer_embedding_size = self.layer_embedding_size*2
            
        self.share_linear = convpass['share_linear']
        
        
        self.fuse_pos = convpass['fuse_pos']
        
        
    
        self.apply_convpass(self.backbone)
        self._set_convpass_trainable(self.backbone)
            
            
    def apply_convpass(self,tuning_module):
        
        # self.adapter_conv_hypetnet = HyperNetwork_conv(f_size=3,in_size=self.backbone_embed_dim,out_size=self.dim)
        
        if self.method == 'attn+mlp':
            
            self.adapter_attn_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type).cuda()
            self.adapter_mlp_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type).cuda()
            
            if self.share_linear:
                self.adapter_down = nn.Linear(self.backbone_embed_dim, self.dim)  # equivalent to 1 * 1 Conv
                self.adapter_up = nn.Linear(self.dim, self.backbone_embed_dim)  # equivalent to 1 * 1 Conv
                nn.init.zeros_(self.adapter_down.bias)
                nn.init.zeros_(self.adapter_up.bias)
                nn.init.xavier_uniform_(self.adapter_down.weight)
                nn.init.zeros_(self.adapter_up.weight)
                
            if self.hyper_linear:
                self.adapter_attn_down_lin_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=1,in_size=self.backbone_embed_dim,out_size=self.dim,init_type=self.init_type).cuda()
                self.adapter_attn_up_lin_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=1,in_size=self.dim,out_size=self.backbone_embed_dim,init_type=self.init_type).cuda()
                
                
                self.adapter_mlp_down_lin_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=1,in_size=self.backbone_embed_dim,out_size=self.dim,init_type=self.init_type).cuda()
                self.adapter_mlp_up_lin_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=1,in_size=self.dim,out_size=self.backbone_embed_dim,init_type=self.init_type).cuda()

                
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for idx,layer in enumerate(child_module):
                        if self.share_linear:
                            layer.adapter_attn = Convpass_swin_hypernet_fusev2_pos(self.adapter_down,self.adapter_up,self.adapter_attn_conv_hypetnet,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim,self.fuse_type,self.fuse_pos).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_swin_hypernet_fusev2_pos(self.adapter_down,self.adapter_up,self.adapter_mlp_conv_hypetnet,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim,self.fuse_type,self.fuse_pos).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                        elif self.hyper_linear:
                            layer.adapter_attn = Convpass_swin_hypernet_fusev3_pos( self.adapter_attn_down_lin_hypetnet,self.adapter_attn_up_lin_hypetnet,self.adapter_attn_conv_hypetnet,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim,self.fuse_type,self.fuse_pos).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_swin_hypernet_fusev3_pos(self.adapter_mlp_down_lin_hypetnet,self.adapter_mlp_up_lin_hypetnet,self.adapter_mlp_conv_hypetnet,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim,self.fuse_type,self.fuse_pos).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                            
                        else:
                            layer.adapter_attn = Convpass_swin_hypernet_fuse_pos(self.adapter_attn_conv_hypetnet,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim,self.fuse_type,self.fuse_pos).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_swin_hypernet_fuse_pos(self.adapter_mlp_conv_hypetnet,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim,self.fuse_type,self.fuse_pos).to(layer.norm1.weight.device)
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
            