"""
LSTM-handler class

Copyright (c) 2021 Elix, Inc.
"""

from typing import List, Tuple, Type

import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

from egegl.data.char_dict import SmilesCharDictionary
from egegl.models.apprentice import (
    AbstractGenerator,
    LSTMGenerator,
)
from egegl.models.handlers.abstract_handler import AbstractGeneratorHandler
from egegl.utils.smiles import smiles_to_actions


class LSTMGeneratorHandler(AbstractGeneratorHandler):
    def __init__(
        self,
        model: LSTMGenerator,
        optimizer: Adam,
        char_dict: SmilesCharDictionary,
        max_sampling_batch_size: int,
    ):
        super().__init__(model, optimizer, char_dict, max_sampling_batch_size)

    def train_on_batch(self, smiles: List[str], device: torch.device) -> float:
        actions, _ = smiles_to_actions(self.char_dict, smiles)
        actions_ten = torch.LongTensor(actions)  # type: ignore
        loss = self.train_on_action_batch(actions=actions_ten, device=device)
        return loss

    def train_on_action_batch(
        self, actions: torch.Tensor, device=torch.device
    ) -> float:
        batch_size = actions.size(0)
        batch_seq_length = actions.size(1)

        actions = actions.to(device)

        start_token_vector = self._get_start_token_vector(batch_size, device)
        input_actions = torch.cat([start_token_vector, actions[:, :-1]], dim=1)
        target_actions = actions

        # Put everything on the device
        input_actions = input_actions.to(device)
        target_actions = target_actions.to(device)

        output, _ = self.model(input_actions, hidden=None)

        # Transpose the output to have N, C, S shape
        log_probs = torch.log_softmax(output.transpose(1, 2), dim=1)
        loss = self.criterion(log_probs, target_actions)

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def sample(self, num_samples: int, device: torch.device, **kwargs):
        action, log_prob, seq_length = self.sample_action(
            num_samples=num_samples, device=device
        )
        smiles = self.char_dict.matrix_to_smiles(action, seq_length - 1)
        return smiles, action, log_prob, seq_length

    def sample_action(
        self, num_samples: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        number_batches = (
            num_samples + self.max_sampling_batch_size - 1
        ) // self.max_sampling_batch_size
        remaining_samples = num_samples

        action = torch.LongTensor(num_samples, self.max_seq_length).to(device)  # type: ignore
        log_prob = torch.FloatTensor(num_samples, self.max_seq_length).to(device)  # type: ignore
        seq_length = torch.LongTensor(num_samples).to(device)  # type: ignore

        batch_start = 0

        for i in range(number_batches):
            batch_size = min(self.max_sampling_batch_size, remaining_samples)
            batch_end = batch_start + batch_size

            action_batch, log_prob_batch, seq_length_batch = self._sample_action_batch(
                batch_size, device
            )
            action[batch_start:batch_end, :] = action_batch
            log_prob[batch_start:batch_end, :] = log_prob_batch
            seq_length[batch_start:batch_end] = seq_length_batch

            batch_start += batch_size
            remaining_samples -= batch_size

        return action, log_prob, seq_length

    def _sample_action_batch(self, batch_size: int, device: torch.device):
        hidden = None
        inp = self._get_start_token_vector(batch_size, device)
        action = torch.zeros((batch_size, self.max_seq_length), dtype=torch.long).to(
            device
        )
        log_prob = torch.zeros((batch_size, self.max_seq_length), dtype=torch.float).to(
            device
        )
        seq_length = torch.zeros(batch_size, dtype=torch.long).to(device)
        ended = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for step in range(self.max_seq_length):
            output, hidden = self.model(inp, hidden)

            prob = torch.softmax(output, dim=2)
            distribution = Categorical(probs=prob)
            action_t = distribution.sample()
            log_prob_t = distribution.log_prob(action_t)
            inp = action_t

            action[~ended, step] = action_t.squeeze(dim=1)[~ended]
            log_prob[~ended, step] = log_prob_t.squeeze(dim=1)[~ended]

            seq_length += (~ended).long()
            ended = ended | (action_t.squeeze(dim=1) == self.char_dict.end_idx).bool()

            if ended.all():
                break

        return action, log_prob, seq_length

