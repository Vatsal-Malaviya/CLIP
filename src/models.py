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


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning
    """

    def __init__(self, embedding_dim, projection_dim) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, projection_dim)

    def forward(self, x):
        return self.fc(x)


class CLIP(nn.Module):
    """
    CLIP model
    """

    def __init__(self, image_encoder, text_encoder, projection_dim) -> None:
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.image_projection = ProjectionHead(
            image_encoder.model.embed_dim, projection_dim
        )
        self.text_projection = ProjectionHead(
            text_encoder.model.config.hidden_size, projection_dim
        )

    def forward(self, image, input_ids, attention_mask):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(input_ids, attention_mask)
        image_projection = self.image_projection(image_features)
        text_projection = self.text_projection(text_features)
        return image_projection, text_projection
