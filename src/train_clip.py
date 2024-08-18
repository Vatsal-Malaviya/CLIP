import logging
import os
import shutil
import sys

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer

from clip import CLIP
from clip import TextTransformer
from clip import VisionTransformer
from dataloader import Flickr30K
from dataloader import create_transform
from loss import clip_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs"))


@hydra.main(config_path=config_dir, config_name="config", version_base="1.1")
def train(cfg: DictConfig):
    """
    Train CLIP model
    """
    logger = logging.getLogger(__name__)

    transform = create_transform(cfg)
    dataset = Flickr30K(
        data_dir=os.path.join(hydra.utils.get_original_cwd(), cfg.dataset.data_dir),
        annotation_file=os.path.join(
            hydra.utils.get_original_cwd(), cfg.dataset.annotation_file
        ),
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_encoder = VisionTransformer()
    text_encoder = TextTransformer()

    clip_model = CLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
    ).to(device)

    optimizer = torch.optim.Adam(clip_model.parameters(), lr=cfg.training.learning_rate)
    tokenizer = BertTokenizer.from_pretrained(cfg.text_encoder.model_name)

    model_weights_dir = os.path.join(os.getcwd(), "model_weights")
    os.makedirs(model_weights_dir, exist_ok=True)

    best_loss = float("inf")
    best_epoch = 0
    for epoch in tqdm(range(cfg.training.epochs), desc="Epochs"):
        clip_model.train()
        running_loss = 0.0
        for batch in tqdm(dataloader, desc="Batches", leave=False):
            optimizer.zero_grad()
            image = batch["image"].to(device)
            text = batch["caption"]

            inputs = tokenizer(
                text, return_tensors="pt", padding=True, truncation=True
            ).to(device)

            image_embeddings, text_embeddings = clip_model(
                image, inputs["input_ids"], inputs["attention_mask"]
            )
            loss = clip_loss(image_embeddings, text_embeddings)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}, Loss: {avg_loss}")

        model_path = os.path.join(model_weights_dir, f"clip_model_epoch_{epoch+1}.pth")
        torch.save(clip_model.state_dict(), model_path)

        if avg_loss < best_loss:
            best_epoch = epoch + 1
            best_loss = avg_loss
            logger.info(f"New Best Model at epoch {epoch+1}, Loss: {best_loss}")
    shutil.copy(
        os.path.join(model_weights_dir, f"clip_model_epoch_{best_epoch}.pth"),
        os.path.join(os.getcwd(), "best_clip_model.pth"),
    )
    logger.info(f"Best Model at epoch {best_epoch}, Loss: {best_loss}")


if __name__ == "__main__":
    train()
