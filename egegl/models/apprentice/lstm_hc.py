"""
LSTM-based Generator class

Copyright (c) 2021 Elix, Inc.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn

from egegl.models.apprentice.abstract_generator import AbstractGenerator


class LSTMGenerator(AbstractGenerator):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        n_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)
        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            batch_first=True,
            num_layers=n_layers,
            dropout=dropout,
        )

        self.init_weights()

    def init_weights(self) -> None:
        # Init the encoder/decoder weights and biases
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0.0)

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
                r_gate = param[int(0.25 * len(param)) : int(0.5 * len(param))]  # type: ignore
                nn.init.constant_(r_gate, 1)

    def forward(  # type: ignore
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embeds = self.encoder(x)
        output, hidden = self.lstm(embeds, hidden)  # type: ignore
        output = self.decoder(output)
        return output, hidden

    def config(self) -> Dict:
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )
