import os
import sys

import pytest
import torch
from transformers import BertTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models import TextTransformer


@pytest.fixture
def tokenizer(model_name="bert-base-uncased"):
    return BertTokenizer.from_pretrained(model_name)


@pytest.fixture
def model(model_name="bert-base-uncased"):
    return TextTransformer(model_name)


def test_text_transformer_output_shape(tokenizer, model):
    sample_text = ["A man is eating food.", "A dog is running.", "A cat is sleeping."]
    inputs = tokenizer(sample_text, return_tensors="pt", padding=True, truncation=True)
    text_embeddings = model(inputs["input_ids"], inputs["attention_mask"])
    assert text_embeddings.shape == torch.Size([len(sample_text), 768])
