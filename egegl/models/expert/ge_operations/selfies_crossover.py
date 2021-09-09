"""
SELFIES-based GE crossover methods

Copyright (c) 2021 Elix, Inc.
"""

import warnings
from math import ceil
from typing import List, Optional, Tuple

import numpy as np
import selfies as sf
from rdkit import Chem, rdBase

warnings.simplefilter("ignore", np.RankWarning)

rdBase.DisableLog("rdApp.error")

from egegl.utils.smiles import get_fp_scores, partial_sanitized_selfie, randomize_smiles


def selfies_crossover(
    parent_a: str, parent_b: str, num_trials: int = 10
) -> Optional[Chem.Mol]:
    parent_smiles = [parent_a, parent_b]

    for _ in range(num_trials):
        child_mol = median_molecule_crossover(parent_a, parent_b)

        if child_mol is not None:
            child_smiles = Chem.MolToSmiles(child_mol)
            if child_smiles is not None and child_smiles not in parent_smiles:
                return child_mol

    return None


def median_molecule_crossover(
    starting_smile: str, target_smile: str
) -> Optional[Chem.Mol]:

    # Randomize parent smiles
    random_starting_smile, random_target_smile = randomize_smiles(
        starting_smile, target_smile
    )

    starting_selfie = sf.encoder(random_starting_smile)
    target_selfie = sf.encoder(random_target_smile)

    # Try partial sanitization for explicit valence errors
    if starting_selfie is None:
        starting_selfie = partial_sanitized_selfie(random_starting_smile)
    if target_selfie is None:
        target_selfie = partial_sanitized_selfie(random_target_smile)

    # Return None if partial sanitization failed too
    if starting_selfie is None or target_selfie is None:
        return None

    starting_selfie_tokens = list(sf.split_selfies(starting_selfie))
    target_selfie_tokens = list(sf.split_selfies(target_selfie))

    len_starting_tokens = len(starting_selfie_tokens)
    len_target_tokens = len(target_selfie_tokens)

    if len_starting_tokens < len_target_tokens:
        for _ in range(len_target_tokens - len_starting_tokens):
            starting_selfie_tokens.append(" ")
    else:
        for _ in range(len_starting_tokens - len_target_tokens):
            target_selfie_tokens.append(" ")
    new_token_length = len(target_selfie_tokens)

    indices_diff = [
        idx
        for idx in range(new_token_length)
        if starting_selfie_tokens[idx] != target_selfie_tokens[idx]
    ]
    path = {}
    path[0] = starting_selfie_tokens

    for iter_ in range(len(indices_diff)):
        idx = np.random.choice(indices_diff, 1)[0]  # Index to be operated on
        indices_diff.remove(idx)  # Remove that index

        # Select the last member of path:
        path_member = path[iter_].copy()

        # Mutate that character to the correct value:
        path_member[idx] = target_selfie_tokens[idx]
        path[iter_ + 1] = path_member.copy()

    # Collapse path to make them into SELFIE strings
    paths_selfies = []
    for i in range(len(path)):
        selfie_str = "".join(x for x in path[i])
        paths_selfies.append(selfie_str.replace(" ", ""))

    # Obtain similarity scores, and only choose the increasing members:
    path_smiles = [sf.decoder(x) for x in paths_selfies]
    path_smiles = [smile for smile in path_smiles if smile]

    # Return None if no intermediate smiles where found
    if len(path_smiles) == 0:
        return None

    path_fp_scores = []
    filtered_path_score: List[float] = []
    smiles_path: List[str] = []

    path_fp_scores = get_fp_scores(path_smiles, target_smile)

    for i in range(1, len(path_fp_scores) - 1):
        if i == 1:
            filtered_path_score.append(path_fp_scores[1])
            smiles_path.append(path_smiles[i])
            continue
        if filtered_path_score[-1] < path_fp_scores[i]:
            filtered_path_score.append(path_fp_scores[i])
            smiles_path.append(path_smiles[i])

    # Return None if no chemical path was found
    if len(smiles_path) == 0:
        return None

    # Get median molecule which is the one with the highest joint similarity
    similarity_starting_smile = get_fp_scores(smiles_path, starting_smile)
    similarity_target_smile = get_fp_scores(smiles_path, target_smile)
    similarities = np.array([similarity_starting_smile, similarity_target_smile])
    similarity_scores = np.average(similarities, axis=0) - (
        np.max(similarities, axis=0) - np.min(similarities, axis=0)
    )
    coefs = np.polyfit([-2 / 3, 0.0, 1.0], [-1.0, 0.0, 1.0], 3)
    final_score = (
        (coefs[0] * (similarity_scores ** 3))
        + (coefs[1] * (similarity_scores ** 2))
        + (coefs[2] * (similarity_scores))
    )

    return Chem.MolFromSmiles(smiles_path[np.argmax(final_score)])
