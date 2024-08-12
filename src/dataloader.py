import os

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Flicker30K(Dataset):
    def __init__(self, data_dir, annotation, transform=None):
        self.data_dir = data_dir
        self.data = pd.read_csv(annotation, sep="|")
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.data_dir, "images", self.data.iloc[idx, 0])
        ).convert("RGB")
        caption = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return {"image": image, "caption": caption}


def create_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
