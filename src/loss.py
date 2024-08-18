import os

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../configs"))
with hydra.initialize_config_dir(config_dir=config_dir, version_base="1.1"):
    cfg = hydra.compose(config_name="config")
    clip = hydra.compose(config_name="clip")
    cfg = OmegaConf.merge(cfg, clip)


def clip_loss(image_embeddings, text_embeddings, temperature=cfg.training.temperature):
    # Normalize the embeddings
    image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

    # Compute cosine similarity and scale by temperature
    logits_per_image = image_embeddings @ text_embeddings.T / temperature
    logits_per_text = text_embeddings @ image_embeddings.T / temperature

    # Create labels for the contrastive loss
    batch_size = image_embeddings.shape[0]
    labels = torch.arange(batch_size, device=image_embeddings.device)

    # Calculate cross-entropy loss
    loss_i2t = F.cross_entropy(logits_per_image, labels)
    loss_t2i = F.cross_entropy(logits_per_text, labels)

    # Final loss is the average of the two
    loss = (loss_i2t + loss_t2i) / 2
    return loss
