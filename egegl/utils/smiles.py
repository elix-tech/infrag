"""
Utility functions related to SMILES representation

Copyright (c) 2021 Elix, Inc.
"""

from typing import List, Tuple

import numpy as np
import selfies as sf
from guacamol.scoring_function import ScoringFunction
from guacamol.utils.chemistry import canonicalize
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoSimMat
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

from egegl.data.char_dict import SmilesCharDictionary


def smiles_to_actions(char_dict: SmilesCharDictionary, smis: List[str]):
    max_seq_length = char_dict.max_smi_len + 1
    enc_smis = list(map(lambda smi: char_dict.encode(smi) + char_dict.END, smis))
    actions = np.zeros((len(smis), max_seq_length), dtype=np.int32)
    seq_lengths = np.zeros((len(smis),), dtype=np.long)

    for i, enc_smi in list(enumerate(enc_smis)):
        for c in range(len(enc_smi)):
            try:
                actions[i, c] = char_dict.char_idx[enc_smi[c]]
            except:
                print(char_dict.char_idx)
                print(enc_smi)
                print(enc_smi[c])
                assert False

        seq_lengths[i] = len(enc_smi)

    return actions, seq_lengths


def canonicalize_and_score_smiles(
    smiles: List[str],
    scoring_function: ScoringFunction,
    char_dict: SmilesCharDictionary,
    pool: Parallel,
) -> Tuple[List[str], List[float]]:
    canon_smiles = pool(
        delayed(lambda smile: canonicalize(smile, include_stereocenters=False))(smile)
        for smile in smiles
    )

    canon_smiles = list(
        filter(
            lambda smile: (smile is not None) and char_dict.is_allowed(smile),
            canon_smiles,
        )
    )
    canon_scores = pool(
        delayed(scoring_function.score)(smile) for smile in canon_smiles
    )

    filted_smiles_and_scores = list(
        filter(
            lambda smile_and_score: smile_and_score[1]
            > scoring_function.scoring_function.corrupt_score,  # type: ignore
            zip(canon_smiles, canon_scores),
        )
    )

    canon_smiles, canon_scores = (
        map(list, zip(*filted_smiles_and_scores))  # type: ignore
        if len(filted_smiles_and_scores) > 0
        else ([], [])
    )

    return canon_smiles, canon_scores


def get_fp_scores(smiles_back: List[str], target_smi: str) -> List[float]:
    smiles_back_scores = []
    target = Chem.MolFromSmiles(target_smi)
    fp_target = Chem.AllChem.GetMorganFingerprint(target, 2)
    for item in smiles_back:
        mol = Chem.MolFromSmiles(item)
        if mol is not None:
            fp_mol = Chem.AllChem.GetMorganFingerprint(mol, 2)
            score = TanimotoSimilarity(fp_mol, fp_target)
            smiles_back_scores.append(score)
        else:
            smiles_back_scores.append(0.0)
    return smiles_back_scores


def partial_sanitized_selfie(smiles) -> str:
    ps_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    ps_mol.UpdatePropertyCache(strict=False)
    Chem.SanitizeMol(
        ps_mol,
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_KEKULIZE
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        catchErrors=True,
    )
    ps_smiles = Chem.MolToSmiles(ps_mol)
    return sf.encoder(ps_smiles)


def randomize_smiles(smiles_a: str, smiles_b: str) -> Tuple[str, str]:
    mol_a = Chem.MolFromSmiles(smiles_a)
    mol_b = Chem.MolFromSmiles(smiles_b)
    Chem.Kekulize(mol_a)
    Chem.Kekulize(mol_b)

    rand_smiles_a = Chem.MolToSmiles(
        mol_a,
        canonical=False,
        doRandom=True,
        isomericSmiles=False,
        kekuleSmiles=True,
    )
    rand_smiles_b = Chem.MolToSmiles(
        mol_b,
        canonical=False,
        doRandom=True,
        isomericSmiles=False,
        kekuleSmiles=True,
    )

    if rand_smiles_a is None:
        rand_smiles_a = smiles_a
    if rand_smiles_b is None:
        rand_smiles_b = smiles_b

    return rand_smiles_a, rand_smiles_b


def calculate_similarity(smiles: List[str]) -> float:
    all_fps = []

    for smiles_str in smiles:
        mol = Chem.MolFromSmiles(smiles_str)
        fp = Chem.AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        all_fps.append(fp)

    sim_mat = GetTanimotoSimMat(all_fps)
    return np.mean(sim_mat)
