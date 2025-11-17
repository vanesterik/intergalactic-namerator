import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from intergalactic_namerator.models import NameratorRNN, build_tokenizer
from intergalactic_namerator.utils import (
    Config,
    ConfigModel,
    ShiftedDataset,
    get_config,
    load_data,
    preprocess,
)

LOG_FILEPATH = Path("logs/main.log")

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(LOG_FILEPATH, level="INFO", rotation="1 MB")


def main():
    # Get config
    config = get_config()

    # Load data
    datafile = Path(config.data.assets_dir) / config.data.filename
    url = config.data.url
    processed_names = load_data(datafile, url)

    # Preprocess to DataLoader
    tokenizer = build_tokenizer(
        corpus=processed_names,
        vocab_size=config.model.vocab_size,
    )
    padded_sequences = preprocess(processed_names, tokenizer)
    dataset = ShiftedDataset(padded_sequences)
    loader = DataLoader(
        dataset=dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
    )

    # Train model
    model, history = train(loader, tokenizer.get_vocab_size(), config)

    # Save tokenizer and model to artifacts folder
    artifacts_dir = Path(config.data.artifacts_dir)
    tokenizer_file = artifacts_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    torch.save(model.state_dict(), artifacts_dir / "model.pth")

    # save config to artifacts folder
    with open(artifacts_dir / "config.json", "w") as file:
        file.write(config.model_dump_json(indent=4))
    logger.info(f"Model and tokenizer saved to {artifacts_dir}")

    # Save training history to artifacts folder
    history_file = artifacts_dir / "history.txt"
    with open(history_file, "w") as file:
        file.write("\n".join(map(str, history)))
    logger.info(f"Training history saved to {history_file}")


def train(loader: DataLoader, vocab_size: int, config: Config):
    model = NameratorRNN(
        ConfigModel(
            embedding_dim=config.model.embedding_dim,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            vocab_size=vocab_size,
        )
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        params=model.parameters(),
        lr=config.training.learning_rate,
    )
    scheduler = ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=config.training.factor,
        patience=config.training.patience,
        min_lr=config.training.min_lr,
    )
    epochs = config.training.epochs
    history = []
    last_lr = 0.0

    for epoch in range(epochs):
        loss = 0.0

        for x, y in loader:
            optimizer.zero_grad()
            hidden = model.init_hidden(x)
            output, hidden = model(x, hidden)
            loss += loss_fn(output.view(-1, vocab_size), y.view(-1))

        loss.backward()  # type: ignore
        optimizer.step()
        scheduler.step(loss)
        history.append(loss.item())  # type: ignore
        curr_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")  # type: ignore
            if last_lr != curr_lr:
                last_lr = curr_lr
                logger.info(f"Current learning rate: {curr_lr}")

    return model, history


if __name__ == "__main__":
    main()
