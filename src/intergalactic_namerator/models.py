from typing import List

import tokenizers as tk
import torch
import torch.nn as nn

from intergalactic_namerator.utils import ConfigModel


def build_tokenizer(corpus: List[str], vocab_size: int) -> tk.Tokenizer:
    tokenizer = tk.Tokenizer(tk.models.BPE(unk_token="<unk>"))
    trainer = tk.trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"],
    )

    # Handle spaces better by removing the prefix space
    tokenizer.pre_tokenizer = tk.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tk.decoders.ByteLevel()

    # Train the BPE model
    tokenizer.train_from_iterator(corpus, trainer)
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    return tokenizer


class NameratorRNN(nn.Module):
    def __init__(self, config: ConfigModel):
        super(NameratorRNN, self).__init__()
        self.config = config
        self.num_layers = config.num_layers

        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
        )
        self.project = nn.Linear(
            config.embedding_dim,
            config.hidden_dim,
        )
        self.rnn = nn.RNN(
            config.embedding_dim,
            config.hidden_dim,
            batch_first=True,
            dropout=0.2,
            num_layers=self.num_layers,
        )
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.activation = nn.GELU()
        self.linear = nn.Linear(
            config.hidden_dim,
            config.vocab_size,
        )

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        identity = self.project(embedded.clone())
        x, hidden = self.rnn(embedded, hidden)
        x = self.norm(x + identity)
        x = self.activation(x)
        x = self.linear(x)
        return x, hidden

    def init_hidden(self, x):
        return torch.zeros(
            self.num_layers,
            x.shape[0],
            self.config.hidden_dim,
        )
