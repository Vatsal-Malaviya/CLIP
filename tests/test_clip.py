import os
import sys

import pytest
import torch
from transformers import BertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.clip import CLIP
from src.clip import ProjectionHead
from src.clip import TextTransformer
from src.clip import VisionTransformer


@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def image_encoder():
    return VisionTransformer(model_name="vit_base_patch16_224")


@pytest.fixture
def text_encoder():
    return TextTransformer(model_name="bert-base-uncased")


@pytest.fixture
def clip_model(image_encoder, text_encoder):
    projection_dim = 256  # Example projection dimension
    return CLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        projection_dim=projection_dim,
    )


def test_vit_init(image_encoder):
    assert isinstance(image_encoder, VisionTransformer)
    assert hasattr(image_encoder, "model")
    assert isinstance(image_encoder.model.head, torch.nn.Identity)


def test_text_transformer_init(text_encoder):
    assert isinstance(text_encoder, TextTransformer)
    assert hasattr(text_encoder, "model")
    assert hasattr(text_encoder.model, "config")


def test_projection_head():
    embedding_dim = 768
    projection_dim = 256
    projection_head = ProjectionHead(
        embedding_dim=embedding_dim, projection_dim=projection_dim
    )

    dummy_input = torch.randn(
        10, embedding_dim
    )  # Batch of 10, each with embedding_dim features
    output = projection_head(dummy_input)

    assert output.shape == (
        10,
        projection_dim,
    )  # Batch of 10, each with projection_dim features


def test_clip_forward_pass(clip_model, tokenizer):
    sample_text = [
        "A man is riding a horse.",
        "A group of people are sitting on a bench.",
    ]
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    dummy_image = torch.randn(2, 3, 224, 224)

    image_projection, text_projection = clip_model(
        dummy_image, input_ids, attention_mask
    )

    # Assertions to check the shapes of the outputs
    assert image_projection.shape == (2, 256)  # 2 images, 256-dimensional projection
    assert text_projection.shape == (
        2,
        256,
    )  # 2 text sequences, 256-dimensional projection
