import os
import sys

import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import VisionTransformer


@pytest.fixture
def model():
    return VisionTransformer()


def test_vit_init(model):
    assert isinstance(model, VisionTransformer)
    assert hasattr(model, "model")
    assert isinstance(model.model.head, torch.nn.Identity)


def test_vit_forward(model):
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 768)


def test_vit_device_moving(model):
    if torch.cuda.is_available():
        model = model.to("cuda")
        dummy_input = torch.randn(1, 3, 224, 224).to("cuda")
        output = model(dummy_input)
        assert output.is_cuda

    model = model.to("cpu")
    dummy_input = torch.randn(1, 3, 224, 224).to("cpu")
    output = model(dummy_input)
    assert not output.is_cuda
