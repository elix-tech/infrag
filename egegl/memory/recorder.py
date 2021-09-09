"""
Base class for best molecules Recorder

Copyright (c) 2021 Elix, Inc.
"""

from functools import total_ordering
from typing import Any, List, Tuple

import numpy as np

from egegl.logger.abstract_logger import AbstractLogger
from egegl.utils.filters import RDFilter


@total_ordering
class RecorderElement:
    def __init__(self, smile, score):
        self.smile = smile
        self.score = score

    def __eq__(self, other):
        return np.isclose(self.score, other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __hash__(self):
        return hash(self.smile)


def unravel_elements(elements: List[RecorderElement]) -> Tuple[List[str], List[float]]:
    return tuple(  # type: ignore
        map(list, zip(*[(element.smile, element.score) for element in elements]))
    )


class Recorder:
    def __init__(
        self,
        scoring_num_list: List[int],
        logger: AbstractLogger,
        record_filtered: bool = True,
    ) -> None:
        self.elements: List[Any] = []
        self.filtered_elements: List[Any] = []
        self.seen_smiles: set = set()

        self.record_filtered = record_filtered
        if self.record_filtered:
            self.rd_filter = RDFilter()

        self.scoring_num_list = scoring_num_list
        self.logger = logger
        self.max_size = max(scoring_num_list)

    def __len__(self) -> int:
        return len(self.elements)

    def add_list(self, smiles: List[str], scores: List[float]) -> None:
        new_elements = [
            RecorderElement(smile=smile, score=score)
            for smile, score in zip(smiles, scores)
        ]
        new_elements = list(set(new_elements))
        new_elements = list(
            filter(lambda element: element.smile not in self.seen_smiles, new_elements)
        )
        self.seen_smiles = self.seen_smiles.union(smiles)
        self.elements.extend(new_elements)
        self.elements = list(set(self.elements))

        if len(self.elements) > self.max_size:
            self.elements = sorted(self.elements, reverse=True)[: self.max_size]

        if self.record_filtered:
            filtered_new_elements = new_elements
            if len(self.filtered_elements) > 0:
                min_filtered_element_score = min(self.filtered_elements).score
                filtered_new_elements = list(
                    filter(
                        lambda element: element.score > min_filtered_element_score,
                        filtered_new_elements,
                    )
                )

            if len(filtered_new_elements) > self.max_size:
                filtered_new_elements = sorted(filtered_new_elements, reverse=True)[
                    : self.max_size
                ]

            filtered_new_elements = list(
                filter(
                    lambda element: self.rd_filter(element.smile) > 0.5,
                    filtered_new_elements,
                )
            )

            self.filtered_elements.extend(filtered_new_elements)

            if len(self.filtered_elements) > self.max_size:
                self.filtered_elements = sorted(self.filtered_elements, reverse=True)[
                    : self.max_size
                ]

    def evaluate(self, rd_filtered: bool) -> float:
        if not rd_filtered:
            scores = [element.score for element in sorted(self.elements, reverse=True)]
        else:
            scores = [
                element.score
                for element in sorted(self.filtered_elements, reverse=True)
            ]

        evalution_score_elementwise = np.array(scores)

        evaluation_score = 0.0
        for scoring_num in self.scoring_num_list:
            evaluation_score += evalution_score_elementwise[:scoring_num].mean() / len(
                self.scoring_num_list
            )

        return evaluation_score

    def get_and_log_score(self) -> float:
        score = self.evaluate(rd_filtered=False)
        self.logger.log_metric("eval_optimized_score", score)

        if self.record_filtered:
            score_filtered = self.evaluate(rd_filtered=True)
            self.logger.log_metric("eval_filtered_score", score_filtered)
            return score_filtered

        return score

    def log_final(self) -> None:
        for element in self.elements:
            self.logger.log_text("optimized_smile", element.smile)
            self.logger.log_metric("optimized_score", element.score)

        if self.record_filtered:
            for element in self.filtered_elements:
                self.logger.log_text("filtered_smile", element.smile)
                self.logger.log_metric("filtered_score", element.score)

    def get_topk(self, top_k: int):
        self.elements = sorted(self.elements, reverse=True)[:top_k]
        return [element.smile for element in self.elements], [
            element.score for element in self.elements
        ]
