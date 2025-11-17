import tomllib
from pathlib import Path
from typing import List

import requests
import tokenizers as tk
import torch
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel
from torch.nn.utils.rnn import pad_sequence


class ConfigData(BaseModel):
    artifacts_dir: str
    assets_dir: str
    filename: str
    url: str


class ConfigModel(BaseModel):
    embedding_dim: int
    hidden_dim: int
    num_layers: int
    vocab_size: int


class ConfigTraining(BaseModel):
    batch_size: int
    epochs: int
    factor: float
    learning_rate: float
    min_lr: float
    patience: int


class Config(BaseModel):
    data: ConfigData
    model: ConfigModel
    training: ConfigTraining


class ShiftedDataset:
    def __init__(self, sequences: torch.Tensor):
        self.X = sequences[:, :-1]
        self.y = sequences[:, 1:]

    def to(self, device: torch.device):
        self.X = self.X.to(device)
        self.y = self.y.to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

    def __repr__(self):
        return f"ShiftedDataset {self.X.shape}"


def get_config() -> Config:
    config_file = Path("config.toml").resolve()
    with config_file.open(mode="rb") as file:
        config = tomllib.load(file)
    return Config(**config)


def get_data(filename: Path, url: str) -> List[str]:
    logger.info(f"Getting data from {url}")

    processed_names = []
    current_url = url

    while True:
        response = requests.get(current_url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Find all anchors with class .category-page__member-link
        links = soup.find_all("a", class_="category-page__member-link")
        names = [link.get_text(strip=True) for link in links]
        processed_names.extend(names)
        logger.info(f"Found {len(names)} names on {current_url}")

        # Find the next button
        next_btn = soup.find("a", class_="category-page__pagination-next")
        if next_btn and next_btn.get("href"):
            href = next_btn["href"]
            if isinstance(href, list):
                href = href[0]
            current_url = href
        else:
            break

    # Remove all entries starting with "Category:"
    processed_names = [
        name for name in processed_names if not name.startswith("Category:")
    ]

    # Part of the entry within parantheses
    cleaned_names = []
    for name in processed_names:
        if "(" in name and ")" in name:
            name = name[: name.index("(")].strip()
        cleaned_names.append(name)

    # Remove duplicates
    processed_names = list(set(cleaned_names))

    # Replace ’ with '
    processed_names = [name.replace("’", "'") for name in processed_names]

    # Sort names
    processed_names.sort()

    # add start and end tokens to each name / sentence
    processed_names = ["<s>" + name + "</s>" for name in processed_names if name]

    # Save the processed names to a file
    logger.info(f"Saving processed names to {filename}")
    with open(filename, "w", encoding="utf-8") as file:
        for name in processed_names:
            file.write(name + "\n")

    return processed_names


def load_data(filename: Path, url: str) -> List[str]:
    if not filename.exists():
        logger.info(f"File {filename} not found. donwloading from {url}")
        names = get_data(filename, url)
    else:
        logger.info(f"Loading processed names from {filename}")
        with open(filename, "r", encoding="utf-8") as file:
            names = [line.strip() for line in file]
    logger.info(f"Loaded {len(names)} names")
    return names


def preprocess(corpus: List[str], tokenizer: tk.Tokenizer) -> torch.Tensor:
    encoded_sequences = [tokenizer.encode(word).ids for word in corpus]
    padded_sequences = pad_sequence(
        [torch.tensor(seq) for seq in encoded_sequences], batch_first=True
    )
    return padded_sequences
