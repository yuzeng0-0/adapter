# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector
import torch,mmengine
from mmdet.models.adapters.hypernet_convpass import HyperNetwork_conv,Convpass_swin_hypernet,Convpass_swin_hypernet_divlinear,Convpass_hypernet
from mmdet.models.adapters.convpass import forward_sam_block_t1,forward_sam_block_t5,forward_sam_block_t6,forward_sam_block_t4,forward_dinov2_block_t1,forward_eva02_block_t1
import torch.nn as nn

@MODELS.register_module()
class ATSSHyperConvpass(SingleStageDetector):
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

        self.layer_embedding = nn.Embedding(num_layer, self.layer_embedding_size).cuda()
        self.layer_embedding.weight.requires_grad = False
        self.init_type = convpass['init_type']
        self.conv_bias = convpass['conv_bias']
        
        
        
        self.div_linear = convpass['div_linear']
        # self.diff_dim = convpass['diff_dim']
        
        if self.div_linear:
            self.num_branch = convpass['num_branch']
            self.kernel_dim = convpass['kernel_dim']         # 论文中的β
            
        # if self.diff_dim:
        #     self.diff_layer_dim = [8,8,16,16,32,32,64,64,128,128,256,256]    
        # else:
        #      self.diff_layer_dim = [self.dim]*12
   
            

        self.apply_convpass(self.backbone)
        self._set_convpass_trainable(self.backbone)
    
        
        
    def apply_convpass(self,tuning_module):
        
        # self.adapter_conv_hypetnet = HyperNetwork_conv(f_size=3,in_size=self.backbone_embed_dim,out_size=self.dim)
        
        
        if self.method == 'attn+mlp':
            
            self.adapter_attn_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type).cuda()
            self.adapter_mlp_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type).cuda()
            if self.conv_bias:
                self.adapter_attn_conv_bias_hypetner = nn.Linear(self.layer_embedding_size,self.dim)
                self.adapter_mlp_conv_bias_hypetner = nn.Linear(self.layer_embedding_size,self.dim)
            else:
                self.adapter_attn_conv_bias_hypetner = None
                self.adapter_mlp_conv_bias_hypetner = None
                
            
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for idx,layer in enumerate(child_module):
                        if tuning_module.__class__.__name__ == 'SAMImageEncoderViTv3':
                            layer.adapter_attn = Convpass_swin_hypernet(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_swin_hypernet(self.adapter_mlp_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                        if tuning_module.__class__.__name__ == 'Dinov2VisionTransformer':
                            layer.adapter_attn = Convpass_hypernet(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_hypernet(self.adapter_mlp_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_dinov2_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                        if tuning_module.__class__.__name__ == 'ViTEVA02':
                            layer.adapter_attn = Convpass_hypernet(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_hypernet(self.adapter_mlp_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_eva02_block_t1.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                        
                            
        if self.method == 'attn+all':
            
            self.adapter_attn_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type).cuda()
            self.adapter_mlp_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type).cuda()
            if self.conv_bias:
                self.adapter_attn_conv_bias_hypetner = nn.Linear(self.layer_embedding_size,self.dim)
                self.adapter_mlp_conv_bias_hypetner = nn.Linear(self.layer_embedding_size,self.dim)
            else:
                self.adapter_attn_conv_bias_hypetner = None
                self.adapter_mlp_conv_bias_hypetner = None
                
            
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for idx,layer in enumerate(child_module):
           
                
                        if self.div_linear:
                            layer.adapter_attn = Convpass_swin_hypernet_divlinear(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,
                                                                                    self.layer_embedding.weight[idx],self.num_branch,self.kernel_dim,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_all =  Convpass_swin_hypernet_divlinear(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,
                                                                                    self.layer_embedding.weight[idx],self.num_branch,self.kernel_dim,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t5.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                        else:
                            layer.adapter_attn = Convpass_swin_hypernet(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_all =  Convpass_swin_hypernet(self.adapter_mlp_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t5.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
        if self.method == 'mlp+all':
            
            self.adapter_attn_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type).cuda()
            self.adapter_mlp_conv_hypetnet = HyperNetwork_conv(z_dim=self.layer_embedding_size,f_size=3,in_size=self.dim,out_size=self.dim,init_type=self.init_type).cuda()
            if self.conv_bias:
                self.adapter_attn_conv_bias_hypetner = nn.Linear(self.layer_embedding_size,self.dim)
                self.adapter_mlp_conv_bias_hypetner = nn.Linear(self.layer_embedding_size,self.dim)
            else:
                self.adapter_attn_conv_bias_hypetner = None
                self.adapter_mlp_conv_bias_hypetner = None
                
            
            for child_module in tuning_module.children():
                if type(child_module) == mmengine.model.base_module.ModuleList:
                    for idx,layer in enumerate(child_module):
           
                
                        if self.div_linear:
                            layer.adapter_all = Convpass_swin_hypernet_divlinear(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,
                                                                                    self.layer_embedding.weight[idx],self.num_branch,self.kernel_dim,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_swin_hypernet_divlinear(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,
                                                                                    self.layer_embedding.weight[idx],self.num_branch,self.kernel_dim,self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            
                        else:
                            layer.adapter_all = Convpass_swin_hypernet(self.adapter_attn_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.adapter_mlp =  Convpass_swin_hypernet(self.adapter_mlp_conv_hypetnet,self.adapter_attn_conv_bias_hypetner,self.layer_embedding.weight[idx],self.backbone_embed_dim,self.dim).to(layer.norm1.weight.device)
                            layer.s = self.scale
                            bound_method = forward_sam_block_t4.__get__(layer, layer.__class__)
                            setattr(layer, 'forward', bound_method)
                            

                        
                        
    def _set_convpass_trainable(self,tuning_module):
        for name, param in tuning_module.named_parameters():
            if '.adapter' not in name:
                param.requires_grad = False
                