import json
import sys
from pathlib import Path

import tokenizers as tk
import torch
import uvicorn
from fastapi import FastAPI
from loguru import logger

from intergalactic_namerator.models import NameratorRNN
from intergalactic_namerator.utils import Config

ARTIFACTS_DIR = Path("artifacts").resolve()
LOG_FILEPATH = Path("logs/app.log")

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(LOG_FILEPATH, level="INFO", rotation="1 MB")


def load_artifacts() -> tuple[NameratorRNN, tk.Tokenizer]:
    logger.info(f"Loading model and tokenizer from {ARTIFACTS_DIR}")

    # Load tokenizer artifact
    tokenizer_file = str(ARTIFACTS_DIR / "tokenizer.json")
    tokenizer = tk.Tokenizer.from_file(tokenizer_file)

    # Load config
    with (ARTIFACTS_DIR / "config.json").open("r") as file:
        config_data = json.load(file)
        config = Config.model_validate(config_data)

    # Load model
    model = NameratorRNN(config=config.model)
    model_path = str(ARTIFACTS_DIR / "model.pth")
    model.load_state_dict(torch.load(model_path, weights_only=False))
    logger.info("Model and tokenizer loaded successfully")

    return model, tokenizer


app = FastAPI()

model, tokenizer = load_artifacts()


# @app.get("/generate")
# async def generate_words(num_words: int = 10, temperature: float = 1.0):
#     try:
#         words = new_words(num_words, temperature)
#         return words
#     except Exception as e:
#         logger.exception(f"Error generating words: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def read_index():
    return {"message": "Welcome to the Intergalactic Namerator API!"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=443)
