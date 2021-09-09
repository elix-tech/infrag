"""
Explainer pretrainer class

Copyright (c) 2021 Elix, Inc.
"""

from typing import List

import torch
from rdkit import Chem
from torch_geometric.data import DataLoader
from tqdm import tqdm

from egegl.data import GraphDataset
from egegl.logger.abstract_logger import AbstractLogger
from egegl.models.handlers import ExplainerHandler
from egegl.utils.featurizer import CanonicalFeaturizer


class ExplainerPreTrainer:
    def __init__(
        self,
        canon_smiles: List[str],
        canon_scores: List[float],
        explainer_handler: ExplainerHandler,
        num_epochs: int,
        batch_size: int,
        save_dir: str,
        num_workers: int,
        device: torch.device,
        logger: AbstractLogger,
    ):
        self.explainer_handler = explainer_handler
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.device = device
        self.logger = logger

        featurizer = CanonicalFeaturizer()
        graph_dataset = GraphDataset(
            canon_smiles=canon_smiles,
            canon_scores=canon_scores,
            featurizer=featurizer,
        )

        self.dataset_loader = DataLoader(
            graph_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

    def pretrain(self) -> None:
        for epoch in tqdm(range(self.num_epochs)):
            for batch in tqdm(self.dataset_loader):
                loss = self.explainer_handler.train_on_graph_batch(
                    batch=batch, device=self.device
                )
                self.logger.log_metric("explainer_loss", loss)
            self.explainer_handler.save(self.save_dir)
