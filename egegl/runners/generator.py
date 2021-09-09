"""
Generator class for optimization

Copyright (c) 2021 Elix, Inc.
"""

from typing import List, Optional, Union

import torch
from guacamol.goal_directed_generator import GoalDirectedGenerator
from guacamol.scoring_function import ScoringFunction
from joblib import Parallel
from tqdm import tqdm

from egegl.memory import Recorder
from egegl.runners.trainer import Trainer


class Generator(GoalDirectedGenerator):
    def __init__(
        self,
        trainer: Trainer,
        recorder: Recorder,
        num_steps: int,
        device: torch.device,
        scoring_num_list: List[int],
        num_jobs: int,
        dataset_type: Optional[str] = None,
    ) -> None:
        self.trainer = trainer
        self.recorder = recorder
        self.num_steps = num_steps
        self.device = device
        self.scoring_num_list = scoring_num_list
        self.dataset_type = dataset_type

        self.pool = Parallel(n_jobs=num_jobs)

    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        self.trainer.init(
            scoring_function=scoring_function, device=self.device, pool=self.pool
        )
        for step in tqdm(range(self.num_steps)):
            smiles, scores = self.trainer.step(
                scoring_function=scoring_function, device=self.device, pool=self.pool
            )

            self.recorder.add_list(smiles=smiles, scores=scores)
            current_score = self.recorder.get_and_log_score()

            if self.dataset_type == "guacamol":
                if current_score == 1.0:
                    break

        self.recorder.log_final()
        self.trainer.log_fragments()
        best_smiles, best_scores = self.recorder.get_topk(top_k=number_molecules)
        return best_smiles

