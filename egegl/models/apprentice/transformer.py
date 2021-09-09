"""
Transformer-based Neural apprentice class

Copyright (c) 2021 Elix, Inc.
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn

from egegl.models.apprentice import AbstractGenerator


class PositionalEncoding(nn.Module):
    def __init__(self, n_embed: int, dropout: float = 0.2, max_len=120) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, n_embed)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embed, 2).float() * (-math.log(10000.0) / n_embed)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        x = x + self.pe[: x.size(0), :]  # type: ignore
        return self.dropout(x)


class TransformerGenerator(AbstractGenerator):
    def __init__(
        self,
        n_token: int,
        n_embed: int,
        n_head: int,
        n_hidden: int,
        n_layers: int,
        dropout: int,
    ) -> None:
        super().__init__()

        self.n_token = n_token
        self.n_embed = n_embed
        self.n_head = n_head
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.dropout = dropout

        encoder_layers = nn.TransformerEncoderLayer(n_embed, n_head, n_hidden, dropout)  # type: ignore
        decoder_layers = nn.TransformerDecoderLayer(n_embed, n_head, n_hidden, dropout)  # type: ignore
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)  # type: ignore
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, n_layers)  # type: ignore

        self.pos_encoder = PositionalEncoding(n_embed, dropout)
        self.embed = nn.Embedding(n_token, n_embed)
        self.dense = nn.Linear(n_embed, n_token)

        self.src_mask: Optional[torch.Tensor] = None
        self.tgt_mask: Optional[torch.Tensor] = None

        self.init_weights()

    def forward(  # type: ignore
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        use_src_mask: bool = False,
        use_tgt_mask: bool = True,
    ) -> torch.Tensor:  # type: ignore
        context = self.encode(src=src, use_src_mask=use_src_mask)
        output = self.decode(tgt=tgt, context=context, use_tgt_mask=use_tgt_mask)
        return output

    def encode(self, src: torch.Tensor, use_src_mask: bool) -> torch.Tensor:
        if use_src_mask:
            self.src_mask = self._generate_square_subsequent_mask(src.shape[1]).to(
                src.device
            )

        src_embed = self.embed(src) * math.sqrt(self.n_embed)
        src_encoded = self.pos_encoder(src_embed.transpose(0, 1))
        context = self.transformer_encoder(src_encoded, mask=self.src_mask)
        return context

    def decode(
        self, tgt: torch.Tensor, context: torch.Tensor, use_tgt_mask: bool
    ) -> torch.Tensor:
        if use_tgt_mask:
            self.tgt_mask = self._generate_square_subsequent_mask(tgt.shape[1]).to(
                tgt.device
            )

        tgt_embed = self.embed(tgt) * math.sqrt(self.n_embed)
        tgt_encoded = self.pos_encoder(tgt_embed.transpose(0, 1))
        output = self.transformer_decoder(
            tgt_encoded, memory=context, tgt_mask=self.tgt_mask
        )
        logits = self.dense(output)
        return logits.transpose(0, 1)

    def config(self) -> Dict:
        return dict(
            n_token=self.n_token,
            n_embed=self.n_embed,
            n_head=self.n_head,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

    def init_weights(self) -> None:
        initrange = 0.1
        nn.init.uniform_(self.embed.weight, -initrange, initrange)
        nn.init.zeros_(self.dense.bias)
        nn.init.uniform_(self.dense.weight, -initrange, initrange)

    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
