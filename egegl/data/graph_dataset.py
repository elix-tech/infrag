"""
Copyright (c) 2021 Elix, Inc.
"""

from typing import List

from rdkit import Chem
from torch_geometric.data import Data, Dataset

from egegl.utils.featurizer import CanonicalFeaturizer


class GraphDataset(Dataset):
    def __init__(
        self,
        canon_smiles: List[str],
        canon_scores: List[float],
        featurizer: CanonicalFeaturizer,
    ):
        super(GraphDataset, self).__init__(
            root=None, transform=None, pre_transform=None
        )

        self.featurizer = featurizer
        self.canon_smiles = canon_smiles
        self.canon_scores = canon_scores

    def len(self) -> int:
        return len(self.canon_smiles)

    def get(self, idx) -> Data:
        data = self.featurizer.process(
            Chem.MolFromSmiles(self.canon_smiles[idx]),
            self.canon_scores[idx],
        )
        return data
