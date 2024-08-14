import timm
import torch.nn as nn
from transformers import BertModel


class VisionTransformer(nn.Module):
    """
    Generate Vision embeddings using Vision Transformer models
    """

    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)


class TextTransformer(nn.Module):
    """
    Generate text embeddings using Bert models
    """

    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask)
        return output.last_hidden_state[:, 0, :]
