"""
Explainer handler class

Copyright (c) 2021 Elix, Inc.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Batch

from egegl.models.attribution import (
    AbstractExplainer,
    DirectedMessagePassingNetwork,
    GraphConvNetwork,
)


class ExplainerHandler:
    def __init__(
        self,
        model: AbstractExplainer,
        optimizer=Adam,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.MSELoss()

    def train_on_graph_batch(self, batch: Batch, device=torch.device) -> float:
        pred = self.generate_preds(batch.to(device))
        loss = self.criterion(pred.squeeze(), batch.y.to(device))

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def save(self, save_dir: str) -> None:
        self.model.save(save_dir)

    def generate_preds(
        self, batch: Batch, return_activations: bool = False
    ) -> torch.Tensor:
        """
        Interface to generate predictions depending on the model instance from Batch data
        """
        if isinstance(self.model, DirectedMessagePassingNetwork):
            preds = self.model(
                batch.x,
                batch.edge_attr,
                batch.edge_index,
                batch.batch,
                return_activations,
            )
        elif isinstance(self.model, GraphConvNetwork):
            preds = self.model(
                batch.x, batch.edge_index, batch.batch, return_activations
            )
        else:
            raise ValueError(f"The explainer class {type(self.model)} is not supported")
        return preds

    def generate_attributions(self, batch: Batch) -> torch.Tensor:
        with torch.no_grad():
            activations = self.generate_preds(batch, return_activations=True)
            cam = torch.matmul(
                activations, torch.transpose(self.model.final_dense.weight.data, 0, 1)  # type: ignore
            )
            return cam
