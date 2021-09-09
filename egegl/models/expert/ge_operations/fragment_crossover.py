"""
Fragment-based crossover methods and functionalities

Copyright (c) 2021 Elix, Inc.
"""

from random import shuffle
from typing import List, Optional

import selfies as sf
from rdkit import Chem


def fragment_crossover(
    fragments: List[Chem.Mol], num_trials: int = 10
) -> Optional[Chem.Mol]:
    # Remove the dummy variables
    full_fragments = [
        Chem.ReplaceSubstructs(
            fragment,
            Chem.MolFromSmarts("[#0]"),
            Chem.MolFromSmiles("[H]"),
            replaceAll=True,
        )[0]
        for fragment in fragments
    ]

    for idx in range(len(full_fragments)):
        Chem.SanitizeMol(full_fragments[idx])
        Chem.RemoveHs(full_fragments[idx])
        Chem.Kekulize(full_fragments[idx])

    for _ in range(num_trials):
        random_frag_smiles = [
            Chem.MolToSmiles(
                frag,
                canonical=False,
                doRandom=True,
                isomericSmiles=False,
                kekuleSmiles=True,
            )
            for frag in full_fragments
        ]

        frag_selfies = [sf.encoder(frag_smiles) for frag_smiles in random_frag_smiles]
        shuffle(frag_selfies)
        new_smiles = sf.decoder("".join(frag_selfies))
        if new_smiles is not None:
            new_mol = Chem.MolFromSmiles(new_smiles)
            if new_mol is not None:
                return new_mol

    return None
