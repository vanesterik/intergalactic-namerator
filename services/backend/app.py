import json
import sys
from pathlib import Path
from typing import List, Tuple

import tokenizers as tk
import torch
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel

from intergalactic_namerator.models import NameratorRNN
from intergalactic_namerator.utils import Config


def load_artifacts(artifacts_dir: Path) -> Tuple[NameratorRNN, tk.Tokenizer]:
    logger.info(f"Loading model and tokenizer from {artifacts_dir}")

    # Load tokenizer artifact
    tokenizer_file = str(artifacts_dir / "tokenizer.json")
    tokenizer = tk.Tokenizer.from_file(tokenizer_file)

    # Load config
    with (artifacts_dir / "config.json").open("r") as file:
        config_data = json.load(file)
        config = Config.model_validate(config_data)

    # Load model
    model = NameratorRNN(config=config.model)
    model_path = str(artifacts_dir / "model.pth")
    model.load_state_dict(torch.load(model_path, weights_only=False))
    logger.info("Model and tokenizer loaded successfully")

    return model, tokenizer


def generate_name(
    model: NameratorRNN,
    tokenizer: tk.Tokenizer,
    max_length: int = 20,
    temperature: float = 1.0,
) -> str:
    start_token_idx = tokenizer.encode("<s>").ids[0]
    input_seq = torch.tensor([[start_token_idx]], dtype=torch.long)
    generated_name = []

    model.eval()
    hidden = model.init_hidden(input_seq)

    for _ in range(max_length - 1):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)

        logits = output.squeeze(0)[-1, :]
        scaled_logits = logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        next_token = torch.multinomial(probabilities, 1).item()

        if next_token == tokenizer.token_to_id("<pad>"):
            break

        generated_name.append(next_token)
        input_seq = torch.tensor([generated_name], dtype=torch.long)

    return tokenizer.decode(generated_name)


def generate_names(
    model: NameratorRNN,
    tokenizer: tk.Tokenizer,
    n: int,
    max_length: int = 20,
    temperature: float = 1.0,
) -> List[str]:
    names = []

    for _ in range(n):
        name = generate_name(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
            temperature=temperature,
        )
        names.append(name)

    return names


class Payload(BaseModel):
    n: int
    max_length: int
    temperature: float


ARTIFACTS_DIR = Path("artifacts").resolve()
LOG_FILEPATH = Path("logs/app.log")

logger.remove()
logger.add(sys.stderr, level="INFO", colorize=True)
logger.add(LOG_FILEPATH, level="INFO", rotation="1 MB")


app = FastAPI()

model, tokenizer = load_artifacts(ARTIFACTS_DIR)


@app.post("/generate")
async def generate(payload: Payload):
    try:
        return generate_names(
            model=model,
            tokenizer=tokenizer,
            n=payload.n,
            max_length=payload.max_length,
            temperature=payload.temperature,
        )
    except Exception as e:
        logger.exception(f"Error generating words: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def read_index():
    return {"message": "Welcome to the Intergalactic Namerator API!"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=80)
