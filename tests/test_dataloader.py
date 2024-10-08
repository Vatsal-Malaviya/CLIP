import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import hydra
import pytest
from torch.utils.data import DataLoader

from src.dataloader import Flickr30K
from src.dataloader import create_transform


@pytest.fixture
def dataloader():
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs"))
    with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = hydra.compose(config_name="config.yaml")
    transform = create_transform(cfg)
    dataset = Flickr30K(
        data_dir=cfg.dataset.data_dir,
        annotation_file=cfg.dataset.annotation_file,
        transform=transform,
    )
    return cfg, DataLoader(dataset, batch_size=cfg.dataset.batch_size, shuffle=True)


def test_dataloader(dataloader):
    cfg, dataloader = dataloader
    data_iter = iter(dataloader)
    batch = next(data_iter)
    assert "image" in batch and "caption" in batch
    assert len(batch["image"]) == cfg.dataset.batch_size
    assert len(batch["caption"]) == cfg.dataset.batch_size


def test_dataloader_content(dataloader):
    _, dataloader = dataloader
    data_iter = iter(dataloader)
    batch = next(data_iter)
    assert batch["image"].shape[1:] == (3, 224, 224)  # Check image size after transform
    assert isinstance(batch["caption"][0], str)  # Check caption is a string
