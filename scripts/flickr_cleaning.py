import pandas as pd
import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def clean_data(cfg: DictConfig) -> None:
    # Load the data
    data_file = hydra.utils.to_absolute_path(cfg.dataset.data_file)
    flickr = pd.read_csv(data_file, sep="|")

    # Clean the data
    flickr.columns = [col.strip() for col in flickr.columns]
    for col in flickr.columns:
        flickr[col] = flickr[col].str.strip()

    # Save the cleaned data
    flickr.to_csv(data_file, sep="|", index=False)
    print(f"Data cleaned and saved to {data_file}")


if __name__ == "__main__":
    clean_data()
