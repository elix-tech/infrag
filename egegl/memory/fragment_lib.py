"""
Base class for the Fragment library

Copyright (c) 2021 Elix, Inc.
"""

import random
import re
from copy import deepcopy
from functools import total_ordering
from itertools import product
from typing import List, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from egegl.models.handlers import ExplainerHandler
from egegl.utils.featurizer import CanonicalFeaturizer


@total_ordering
class StorageElement:
    def __init__(self, smile: str, fragment_score: float):
        self.smile = smile
        self.fragment_score = fragment_score

    def __eq__(self, other):
        return np.isclose(self.fragment_score, other.fragment_score)

    def __lt__(self, other):
        return self.fragment_score < other.fragment_score

    def __hash__(self):
        return hash(self.smile)


class FragmentLibrary:
    def __init__(
        self,
        explainer_handler: ExplainerHandler,
        max_lib_size: float = 1000,
    ) -> None:
        self.fragments: List[StorageElement] = []

        self.max_lib_size = max_lib_size
        self.explainer_handler = explainer_handler

        self.featurizer = CanonicalFeaturizer()

    def __len__(self) -> int:
        return len(self.fragments)

    def add_list(self, fragment_list: List[str], fragment_scores: List[float]) -> None:
        new_fragments = [
            StorageElement(smile=smile, fragment_score=score)
            for smile, score in zip(fragment_list, fragment_scores)
        ]

        for fragment_new, fragment_original in product(new_fragments, self.fragments):
            if fragment_new.smile == fragment_original.smile:
                fragment_original.fragment_score = max(
                    fragment_new.fragment_score, fragment_original.fragment_score
                )

        self.fragments.extend(new_fragments)
        self.fragments = list(set(self.fragments))

    def squeeze_by_max_length(self):
        top_k = min(self.max_lib_size, len(self.fragments))
        self.fragments = sorted(self.fragments, reverse=True)[:top_k]

    def get_fragments(self) -> Tuple[List[str], List[float]]:
        smiles, scores = unravel_elements(sorted(self.fragments))
        return smiles, scores

    def expand_fragment_lib(
        self,
        candidates_pool: List[str],
        candidates_score: List[float],
        device: torch.device,
    ) -> None:
        frag_dict = {}

        for smiles, score in zip(candidates_pool, candidates_score):
            frags, scores = self.fragment_molecule(smiles, score, device)
            for fragment, score in zip(frags, scores):
                if fragment not in frag_dict:
                    frag_dict[fragment] = score
                else:
                    frag_dict[fragment] = max(frag_dict[fragment], score)

        self.add_list(
            fragment_list=list(frag_dict.keys()),
            fragment_scores=list(frag_dict.values()),
        )

    def fragment_molecule(
        self, smiles: str, score: float, device: torch.device
    ) -> Tuple[List[str], List[float]]:
        unique_fragments = []

        # Generate the graph data to predict attributions
        mol = Chem.MolFromSmiles(smiles)
        graph_data = self.featurizer.process(mol, score)
        graph_batch = Batch.from_data_list([graph_data])

        # Get attributions
        self.explainer_handler.model.eval()
        with torch.no_grad():
            attr_ten = self.explainer_handler.generate_attributions(
                graph_batch.to(device)
            )
            attr_arr = attr_ten.detach().cpu().numpy()

        dis_bonds = []
        for idx in range(len(mol.GetBonds())):
            bond = mol.GetBondWithIdx(idx)
            begin_atom, end_atom = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            if attr_arr[begin_atom] * attr_arr[end_atom] < 0:
                if not bond.GetIsAromatic():
                    if bond.GetBondType() == Chem.BondType.SINGLE:
                        dis_bonds.append(idx)

        if len(dis_bonds) == 0:
            return [smiles], [score]

        mol_frags = Chem.FragmentOnBonds(mol, tuple(dis_bonds))
        fragms = Chem.GetMolFrags(mol_frags, asMols=True)

        for fragment in fragms:
            frag_without_r = Chem.DeleteSubstructs(
                deepcopy(fragment), Chem.MolFromSmarts("[#0]")
            )
            frag_atom_indices = mol.GetSubstructMatches(frag_without_r)
            for index_tuple in frag_atom_indices:
                if np.sum(attr_arr[list(index_tuple)]) > 0.0:
                    unique_fragments.append(
                        re.sub("\[[1-9]*\*\]", "[*]", Chem.MolToSmiles(fragment))
                    )

        unique_fragments = list(set(unique_fragments))
        score_list = [score] * len(unique_fragments)

        return unique_fragments, score_list


def unravel_elements(elements: List[StorageElement]):
    return tuple(
        map(
            list,
            zip(*[(element.smile, element.fragment_score) for element in elements]),
        )
    )
