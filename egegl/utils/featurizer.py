"""
Featurizer class for Explainer model

Copyright (c) 2021 Elix, Inc.
"""

import itertools
from typing import Any, Callable, Dict, List, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Data

from .feature_functions import *


class CanonicalFeaturizer:
    def __init__(self):
        self.node_dim = 74
        self.edge_dim = 12

    def process(self, sample: Chem.Mol, score: float) -> Data:

        node_features = self._get_node_features(sample)
        edge_index, edge_attr = self._get_edge_features(sample)

        # Sort indices.
        if edge_index.numel() > 0:
            perm = (edge_index[0] * node_features.size(0) + edge_index[1]).argsort()
            edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

        data_sample = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.FloatTensor([score]),
        )
        return data_sample

    def _get_node_features(self, mol: Chem.Mol) -> torch.Tensor:
        node_features = [
            list(
                itertools.chain.from_iterable(
                    [
                        self._featurize_atom(atom_rep)
                        for atom_rep in self._get_atom_representations(atom)
                    ]
                )
            )
            for atom in self._get_mol_iterable(mol)
        ]
        node_features_tensor = torch.FloatTensor(node_features).view(  # type: ignore
            -1, self.node_dim
        )
        return node_features_tensor

    def _featurize_atom(self, atom: Chem.Atom) -> List[float]:
        return list(
            itertools.chain.from_iterable(
                [
                    featurizer(atom)
                    for featurizer in self._dict_atom_features()[type(atom)]
                ]
            )
        )

    def _get_mol_iterable(self, mol: Chem.Mol):
        return mol.GetAtoms()

    def _get_atom_representations(self, atom: Chem.Atom):
        return [atom]

    def _dict_atom_features(self) -> Dict[Any, List[Callable]]:
        return {
            Chem.Atom: [
                atom_type_one_hot,
                atom_degree_one_hot,
                atom_implicit_valence_one_hot,
                atom_formal_charge,
                atom_num_radical_electrons,
                atom_hybridization_one_hot,
                atom_is_aromatic,
                atom_total_num_H_one_hot,
            ]
        }

    def _list_bond_features(self) -> List[Callable]:
        return [
            bond_type_one_hot,
            bond_is_conjugated,
            bond_is_in_ring,
            bond_stereo_one_hot,
        ]

    def _get_edge_features(self, mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            e = self._featurize_bond(bond)

            edge_indices += [[i, j], [j, i]]
            edge_attrs += [e, e]

        edge_index = torch.tensor(edge_indices)
        edge_index = edge_index.t().to(torch.long).view(2, -1)
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float).view(-1, self.edge_dim)

        return edge_index, edge_attr

    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        return list(
            itertools.chain.from_iterable(
                [featurizer(bond) for featurizer in self._list_bond_features()]
            )
        )

