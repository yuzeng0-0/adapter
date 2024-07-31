
import torch
ckpt_path = 'checkpoints/vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth'
save_path = 'checkpoints/changed_vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth'


with open(ckpt_path, "rb") as f:
    state_dict = torch.load(f)
        
    new_state_dict = {}
    for k in state_dict['state_dict']:
        new_state_dict[k[9:]] = state_dict['state_dict'][k]
    del new_state_dict['mask_token']
    
    state_dict['state_dict'] = new_state_dict
    
torch.save(state_dict,'checkpoints/changed_vit-small-p14_dinov2-pre_3rdparty_20230426-5641ca5a.pth')