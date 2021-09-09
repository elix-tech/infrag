import sys
from pathlib import Path

import networkx as nx
from guacamol.common_scoring_functions import (
    RdkitScoringFunction,
    TanimotoScoringFunction,
)
from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.goal_directed_score_contributions import uniform_specification
from guacamol.scoring_function import MoleculewiseScoringFunction
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, RDConfig

sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
import sascorer

from egegl.scoring.constants import (
    ATOMRING_MEAN,
    ATOMRING_STD,
    CYCLEBASIS_MEAN,
    CYCLEBASIS_STD,
    LOGP_MEAN,
    LOGP_STD,
    SA_MEAN,
    SA_STD,
)


class ThresholdedImprovementScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, objective, constraint, threshold, offset):
        super().__init__()
        self.objective = objective
        self.constraint = constraint
        self.threshold = threshold
        self.offset = offset

    def raw_score(self, smiles):
        score = (
            self.corrupt_score
            if (self.constraint.score(smiles) < self.threshold)
            else (self.objective.score(smiles) + self.offset)
        )
        return score


def penalized_logp_atomrings():
    benchmark_name = "Penalized logP"
    objective = RdkitScoringFunction(
        descriptor=lambda mol: _penalized_logp_atomrings(mol)
    )
    objective.corrupt_score = -1000.0
    specification = uniform_specification(1)
    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=objective,
        contribution_specification=specification,
    )


def penalized_logp_cyclebasis():
    benchmark_name = "Penalized logP CycleBasis"
    objective = RdkitScoringFunction(
        descriptor=lambda mol: _penalized_logp_cyclebasis(mol)
    )
    objective.corrupt_score = -1000.0
    specification = uniform_specification(1)
    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=objective,
        contribution_specification=specification,
    )


def similarity_constrained_penalized_logp_atomrings(
    smiles, name: str, threshold: float, fp_type: str = "ECFP4"
) -> GoalDirectedBenchmark:
    benchmark_name = f"{name} {threshold:.1f} Similarity Constrained Penalized logP"

    objective = RdkitScoringFunction(
        descriptor=lambda mol: _penalized_logp_atomrings(mol)
    )
    offset = -objective.score(smiles)
    constraint = TanimotoScoringFunction(target=smiles, fp_type=fp_type)
    constrained_objective = ThresholdedImprovementScoringFunction(
        objective=objective, constraint=constraint, threshold=threshold, offset=offset
    )
    constrained_objective.corrupt_score = -1000.0
    specification = uniform_specification(1)

    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=constrained_objective,
        contribution_specification=specification,
    )


def similarity_constrained_penalized_logp_cyclebasis(
    smiles, name: str, threshold: float, fp_type: str = "ECFP4"
) -> GoalDirectedBenchmark:
    benchmark_name = f"{name} {threshold:.1f} Similarity Constrained Penalized logP"

    objective = RdkitScoringFunction(
        descriptor=lambda mol: _penalized_logp_cyclebasis(mol)
    )
    offset = -objective.score(smiles)
    constraint = TanimotoScoringFunction(target=smiles, fp_type=fp_type)
    constrained_objective = ThresholdedImprovementScoringFunction(
        objective=objective, constraint=constraint, threshold=threshold, offset=offset
    )
    constrained_objective.corrupt_score = -1000.0
    specification = uniform_specification(1)

    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=constrained_objective,
        contribution_specification=specification,
    )


def _penalized_logp_atomrings(mol: Mol) -> float:
    log_p = Descriptors.MolLogP(mol)
    sa_score = sascorer.calculateScore(mol)

    cycle_list = mol.GetRingInfo().AtomRings()
    largets_ring_size = max([len(size) for size in cycle_list]) if cycle_list else 0
    cycle_score = max(largets_ring_size - 6, 0)

    log_p_term = (log_p - LOGP_MEAN) / LOGP_STD
    sa_term = (sa_score - SA_MEAN) / SA_STD
    cycle_term = (cycle_score - ATOMRING_MEAN) / ATOMRING_STD

    return log_p_term - sa_term - cycle_term


def _penalized_logp_cyclebasis(mol: Mol) -> float:
    log_p = Descriptors.MolLogP(mol)
    sa_score = sascorer.calculateScore(mol)

    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    largets_ring_size = max([len(size) for size in cycle_list]) if cycle_list else 0
    cycle_score = max(largets_ring_size - 6, 0)

    log_p_term = (log_p - LOGP_MEAN) / LOGP_STD
    sa_term = (sa_score - SA_MEAN) / SA_STD
    cycle_term = (cycle_score - CYCLEBASIS_MEAN) / CYCLEBASIS_STD

    return log_p_term - sa_term - cycle_term
