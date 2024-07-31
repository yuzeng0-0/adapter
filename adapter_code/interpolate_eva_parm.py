import torch
import torch.nn.functional as F

checkpoint_path = 'checkpoints/changed_eva02-small-p14_pre_in21k_20230505-3175f463.pth'
dst_path = 'checkpoints/changed_eva02-small-p14to16_pre_in21k_20230505-3175f463.pth'

data = torch.load(checkpoint_path)

patch_embed_weight = data['state_dict']['patch_embed.projection.weight']

src_shape = patch_embed_weight.shape
dst_shape = (16,16)
mode = 'bicubic'

dst_weight = F.interpolate(
        patch_embed_weight.float(), size=dst_shape, align_corners=False, mode=mode)


data['state_dict']['patch_embed.projection.weight'] = dst_weight

torch.save(data,dst_path)

a = 1+1