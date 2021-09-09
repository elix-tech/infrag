"""
SELFIES-based GE mutation methods

Copyright (c) 2021 Elix, Inc.
"""

import random
from typing import Optional

import numpy as np
import selfies as sf
from rdkit import Chem

from egegl.utils.molecules import mol_is_ok


def insert_char(selfie_chars, random_index, random_char):
    selfie_mutated_chars = (
        selfie_chars[:random_index] + [random_char] + selfie_chars[random_index:]
    )
    return selfie_mutated_chars


def replace_char(selfie_chars, random_index, random_char):
    if random_index == 0:
        selfie_mutated_chars = [random_char] + selfie_chars[1:]
    else:
        selfie_mutated_chars = (
            selfie_chars[:random_index]
            + [random_char]
            + selfie_chars[random_index + 1 :]
        )
    return selfie_mutated_chars


def delete_char(selfie_chars, random_index, random_char):
    if random_index == 0:
        selfie_mutated_chars = selfie_chars[1:]
    else:
        selfie_mutated_chars = (
            selfie_chars[:random_index] + selfie_chars[random_index + 1 :]
        )
    return selfie_mutated_chars


def selfies_mutate(
    mol: Chem.Mol, mutation_rate: float, num_trials: int = 10
) -> Optional[Chem.Mol]:
    if random.random() > mutation_rate:
        return mol

    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except ValueError:
        return mol

    alphabet = list(sf.get_semantic_robust_alphabet())
    p = [0.333, 0.333, 0.334]

    for _ in range(num_trials):
        mutations_list = [insert_char, replace_char, delete_char]
        mutation = np.random.choice(mutations_list, p=p)

        smiles = Chem.MolToSmiles(mol)
        selfies = sf.encoder(smiles)
        selfie_chars = list(sf.split_selfies(selfies))

        if mutation.__name__ == "insert_char":
            random_index = np.random.randint(len(selfie_chars) + 1)
        else:
            random_index = np.random.randint(len(selfie_chars))
        random_char = np.random.choice(alphabet, size=1)[0]

        new_selfie_list = mutation(selfie_chars, random_index, random_char)
        new_selfie = "".join(x for x in new_selfie_list)

        new_smiles = sf.decoder(new_selfie)
        if new_smiles is not None:
            new_mol = Chem.MolFromSmiles(new_smiles)
            if mol_is_ok(new_mol):
                return new_mol

    return None
