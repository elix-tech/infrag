"""
Abstract Generator-handler class

Copyright (c) 2021 Elix, Inc.
"""

from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Type

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from egegl.data.char_dict import SmilesCharDictionary
from egegl.models.apprentice import (
    AbstractGenerator,
    LSTMGenerator,
    TransformerGenerator,
)
from egegl.utils.smiles import smiles_to_actions


class AbstractGeneratorHandler(metaclass=ABCMeta):
    def __init__(
        self,
        model: AbstractGenerator,
        optimizer: Adam,
        char_dict: SmilesCharDictionary,
        max_sampling_batch_size: int,
    ):
        self.model = model
        self.optimizer = optimizer
        self.char_dict = char_dict
        self.max_sampling_batch_size = max_sampling_batch_size
        self.max_seq_length = self.char_dict.max_smi_len + 1
        self.criterion = nn.NLLLoss()

    @abstractmethod
    def train_on_batch(self, smiles: List[str], device: torch.device) -> float:
        raise NotImplementedError

    @abstractmethod
    def train_on_action_batch(self, actions: torch.Tensor, device=torch.device):
        raise NotImplementedError

    @abstractmethod
    def sample(self, num_samples: int, device: torch.device, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _sample_action_batch(self, batch_size: int, device: torch.device):
        raise NotImplementedError

    @abstractmethod
    def sample_action(
        self, num_samples: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def save(self, save_dir: str) -> None:
        self.model.save(save_dir=save_dir)

    def _get_start_token_vector(self, batch_size, device):
        return (
            torch.LongTensor(batch_size, 1).fill_(self.char_dict.begin_idx).to(device)
        )
