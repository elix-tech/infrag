"""
Utility functions to manipulate molecules

Copyright (c) 2021 Elix, Inc.
"""

import random
from typing import List, Optional

import numpy as np
from rdkit import Chem, rdBase

rdBase.DisableLog("rdApp.error")

# For now, we restrict the size in the same manner as the original implementation!
# It seems that these values were taken from the GB_GA paper
SIZE_MEAN = 39.15
SIZE_STD = 3.50


def mol_is_ok(mol: Chem.Mol) -> bool:
    try:
        Chem.SanitizeMol(mol)
        target_size = SIZE_STD * np.random.randn() + SIZE_MEAN
        if mol.GetNumAtoms() > 5 and mol.GetNumAtoms() < target_size:
            return True
        else:
            return False
    except:
        return False


def ring_is_ok(mol: Chem.Mol) -> bool:
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]")):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts("[R]=[R]=[R]"))
    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6
    double_bond_in_small_ring = mol.HasSubstructMatch(
        Chem.MolFromSmarts("[r3,r4]=[r3,r4]")
    )

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def cut(mol: Chem.Mol) -> Optional[List[Chem.Mol]]:
    if not mol.HasSubstructMatch(Chem.MolFromSmarts("[*]-;!@[*]")):
        return None

    bis = random.choice(mol.GetSubstructMatches(Chem.MolFromSmarts("[*]-;!@[*]")))
    bs = [mol.GetBondBetweenAtoms(bis[0], bis[1]).GetIdx()]
    fragments_mol = Chem.FragmentOnBonds(mol, bs, addDummies=True, dummyLabels=[(1, 1)])

    try:
        return Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
    except:
        return None

    return None


def cut_ring(mol: Chem.Mol, num_trials: int = 10) -> Optional[List[Chem.Mol]]:
    for _ in range(num_trials):
        if random.random() <= 0.5:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R]@[R]@[R]")):
                return None
            bis = random.choice(
                mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R]@[R]@[R]"))
            )
            bis = (
                (bis[0], bis[1]),
                (bis[2], bis[3]),
            )
        else:
            if not mol.HasSubstructMatch(Chem.MolFromSmarts("[R]@[R;!D2]@[R]")):
                return None
            bis = random.choice(
                mol.GetSubstructMatches(Chem.MolFromSmarts("[R]@[R;!D2]@[R]"))
            )
            bis = (
                (bis[0], bis[1]),
                (bis[1], bis[2]),
            )

        bs = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bis]
        fragments_mol = Chem.FragmentOnBonds(
            mol, bs, addDummies=True, dummyLabels=[(1, 1), (1, 1)]
        )

        try:
            fragments = Chem.GetMolFrags(fragments_mol, asMols=True, sanitizeFrags=True)
            if len(fragments) == 2:
                return fragments
        except:
            return None

    return None
