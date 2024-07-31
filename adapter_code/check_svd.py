import torch
import numpy as np

import matplotlib.pyplot as plt

ckpt_path = '/home/chenz/code/PEFT/work_dirs/atss_dinov2_small_1x_cityscapes_bitfit/epoch_1.pth'
with open(ckpt_path, "rb") as f:
    state_dict = torch.load(f)
    
    
state_dict = state_dict['state_dict']

all_svd = []

for key,value in state_dict.items():
    
    if 'ffn' in key or 'attn' in key and 'weight' in key:
        if value.ndim == 1:
            continue
        
        _,s,_v = torch.svd(value)
        all_svd.append(s)
        
all_svd = torch.cat(all_svd)

all_svd = all_svd.numpy()


plt.hist(all_svd,bins=50,range=(0,1))
plt.savefig('svd_hist.jpg')