import timm
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)
