"""
Module with class to handle sampling budget between the genetic experts

Copyright (c) 2021 Elix, Inc.
"""

from copy import deepcopy
from typing import List

import numpy as np


class SamplingHandler:
    def __init__(
        self,
        num_experts: int,
        sampling_budget: int,
        sampling_strategy: str,
    ):
        self.num_experts = num_experts
        self.sampling_budget = sampling_budget
        self.sampling_strategy = sampling_strategy

        self.uniform_distribution = [1.0 / num_experts] * num_experts
        self.previous_expert_memory_ratio = deepcopy(self.uniform_distribution)
        self.previous_probability_distribution = deepcopy(self.uniform_distribution)

    def calculate_partial_query_size(
        self,
        expert_contribution: List[int],
    ) -> List[int]:
        if self.num_experts == 1:
            return [self.sampling_budget]

        preferences = [0] * self.num_experts
        unique_ids, counts = np.unique(
            np.array(expert_contribution), return_counts=True
        )

        for idx, count in zip(unique_ids, counts):
            preferences[idx] = count

        current_expert_memory_ratio = preferences / np.sum(preferences)

        if self.sampling_strategy == "softmax":
            probability_distribution = self._softmax_strategy(
                current_expert_memory_ratio
            )
        elif self.sampling_strategy == "rebalancing":
            probability_distribution = self._rebalancing_strategy(
                current_expert_memory_ratio
            )
        elif self.sampling_strategy == "fixed":
            probability_distribution = np.array(self.uniform_distribution)
        else:
            raise ValueError(
                "Sampling strategy '{}' is not a valid sampling strategy. Please choose either 'fixed', 'softmax' or 'rebalancing'".format(
                    self.sampling_strategy
                )
            )

        # Save the current distribution states for future calculations
        self.previous_probability_distribution = probability_distribution
        self.previous_expert_memory_ratio = current_expert_memory_ratio

        query_size = probability_distribution * self.sampling_budget
        query_size_int = query_size.astype(int)
        return query_size_int.tolist()

    def _softmax_strategy(self, expert_ratio):
        probability_distribution = softmax(expert_ratio)
        return probability_distribution

    def _rebalancing_strategy(self, expert_ratio):
        expert_improvement = expert_ratio - self.previous_expert_memory_ratio
        updated_distribution = (
            self.previous_probability_distribution + expert_improvement
        )
        rebalanced_distribution = updated_distribution + 0.2 * (
            self.uniform_distribution - updated_distribution
        )
        return rebalanced_distribution


def softmax(x_arr: np.ndarray) -> np.ndarray:
    y_arr = np.exp(x_arr - np.max(x_arr))
    prob = y_arr / y_arr.sum(0)
    return prob
