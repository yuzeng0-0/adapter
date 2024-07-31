import torch
import torch.nn as nn

from mmdet.registry import MODELS

@MODELS.register_module()
class BaseAdapterBlock(nn.Module):
    def __init__(self, mainmodule, adapter=None):
        super(BaseAdapterBlock, self).__init__()
        self.mainmodule = mainmodule
        if isinstance(adapter, dict):
            adapter = MODELS.build(adapter)
        assert isinstance(adapter, nn.Module)
        self.adapter = adapter

    def forward(self, x):
        main_x = self.mainmodule(x)
        
        adapter_x = self.adapter(x)
        
        return main_x + adapter_x


from typing import Callable, Dict, List, Optional, Tuple


def inject_adapter(
    model: nn.Module,
    adapter, # nn.Module or dict
    target_replacemodule, # List[str] = ["conv_after_body"]
    same_mid_size: bool = True,    # 
):
    """
    inject any adapter into model.
    """
    if not isinstance(target_replacemodule, list):
        target_replacemodule = [target_replacemodule]
    name_mode = False; module_mode = False
    if all([isinstance(sub_module, str) for sub_module in target_replacemodule]):
        name_mode = True
    elif all([isinstance(sub_module, nn.Module) for sub_module in target_replacemodule]):
        module_mode = True
        target_replacemodule = [sub_module.__class__.__name__ for sub_module in target_replacemodule]
    elif all([isinstance(sub_module, type) for sub_module in target_replacemodule]):
        module_mode = True
        target_replacemodule = [sub_module.__name__ for sub_module in target_replacemodule]
    else:
        raise ValueError(f"only one type(`nn.Module` or `str`) in `target_replacemodule`.")
    print(f"target replaced module: {target_replacemodule}")
    
    module_config = dict()
    # almost adapters have the same input and output dims 
    module_config['same_mid_size'] = same_mid_size
    print(f"module config: {module_config}")
    
    module_list = [[module_name, module] for module_name, module in model.named_modules()]
    for module_name, module in module_list:
        print(f"Module Name: {module_name}, Module Type: {module.__class__.__name__}")
        if module_mode:
            if module.__class__.__name__ in target_replacemodule:
                build_adapter_block(model, module_name, adapter, module_config)
                continue
        if name_mode:
            if module_name in target_replacemodule:
                build_adapter_block(model, module_name, adapter, module_config)
                continue

def build_adapter_block(model, module_name, adapter, module_config):
    # get main module from model
    main_module = model.get_submodule(module_name)
    
    if module_config['same_mid_size']:
        if main_module.__class__ in [nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            in_dims, out_dims = main_module.in_channels, main_module.out_channels
        elif main_module.__class__ in [nn.Linear]:
            in_dims, out_dims = main_module.in_features, main_module.out_features
        if in_dims != out_dims:
            return
    # create adapter block
    adapter_block = BaseAdapterBlock(mainmodule=main_module, adapter=adapter)
    # replace module by adapter_block
    setattr(model, module_name, adapter_block)