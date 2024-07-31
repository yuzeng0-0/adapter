import torch
from mmpretrain import get_model

model = get_model('vit-small-p14_dinov2-pre_3rdparty', pretrained=True)
inputs = torch.rand(1, 3, 224, 224)
out = model(inputs)
print(type(out))
# To extract features.
feats = model.extract_feat(inputs)
print(type(feats))