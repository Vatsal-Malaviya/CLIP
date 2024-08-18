import os
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional

import pandas as pd
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Flickr30K(Dataset):
    """
    PyTorch Dataset for the Flickr30K dataset.

    Args:
        data_dir (str): Directory where the images are stored.
        annotation_file (str): Path to the annotation file (CSV) containing image file
            names and captions.
        transform (Optional[Callable]): Optional transform to be applied on a sample.
    """

    def __init__(
        self, data_dir: str, annotation_file: str, transform: Optional[Callable] = None
    ) -> None:
        self.data_dir = data_dir
        self.data = pd.read_csv(annotation_file, sep="|", dtype=str)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_path = os.path.join(self.data_dir, "images", self.data.iloc[idx, 0])
        image = Image.open(image_path).convert("RGB")
        caption = self.data["comment"].iloc[idx]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}


def create_transform(cfg: DictConfig) -> Callable:
    """
    Creates a transform pipeline for the images.

    Args:
        resize (tuple): Size to which images will be resized (height, width).
        mean (tuple): Mean for normalization.
        std (tuple): Standard deviation for normalization.

    Returns:
        Callable: Transform pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize(cfg.image_encoder.transform.resize),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=cfg.image_encoder.transform.mean,
                std=cfg.image_encoder.transform.std,
            ),
        ]
    )
