"""
Crossover methods and functionalities

Copyright (c) 2021 Elix, Inc.
"""

import random
from typing import Any, List, Optional

from rdkit import Chem, rdBase

from egegl.utils.molecules import cut, cut_ring, mol_is_ok, ring_is_ok

rdBase.DisableLog("rdApp.error")


def crossover(parent_a: str, parent_b: str, num_trials: int = 10) -> Optional[Chem.Mol]:
    parent_smiles = [parent_a, parent_b]
    parent_mol_a, parent_mol_b = Chem.MolFromSmiles(parent_a), Chem.MolFromSmiles(
        parent_b
    )

    try:
        Chem.Kekulize(parent_mol_a, clearAromaticFlags=True)
        Chem.Kekulize(parent_mol_b, clearAromaticFlags=True)
    except:
        pass

    for _ in range(num_trials):
        if random.random() <= 0.5:
            # Non-ring crossover
            child_mol = non_ring_crossover(parent_mol_a, parent_mol_b)
        else:
            # Ring crossover
            child_mol = ring_crossover(parent_mol_a, parent_mol_b)

        if child_mol is not None:
            child_smiles = Chem.MolToSmiles(child_mol)
            if child_smiles is not None and child_smiles not in parent_smiles:
                return child_mol

    return None


def non_ring_crossover(
    parent_a: Chem.Mol, parent_b: Chem.Mol, num_trials: int = 10
) -> Optional[Chem.Mol]:
    for _ in range(num_trials):
        fragments_a = cut(parent_a)
        fragments_b = cut(parent_b)

        if fragments_a is None or fragments_b is None:
            return None

        reaction = Chem.AllChem.ReactionFromSmarts("[*:1]-[1*].[1*]-[*:2]>>[*:1]-[*:2]")

        new_mol_trial = []
        for frag_a in fragments_a:
            for frag_b in fragments_b:
                new_mol_trial.append(reaction.RunReactants((frag_a, frag_b))[0])

        child_molecules = []
        for mol in new_mol_trial:
            mol = mol[0]
            if mol_is_ok(mol):
                child_molecules.append(mol)

        if len(child_molecules) > 0:
            return random.choice(child_molecules)

    return None


def ring_crossover(
    parent_a: Chem.Mol, parent_b: Chem.Mol, num_trials: int = 10
) -> Optional[Chem.Mol]:
    ring_smarts = Chem.MolFromSmarts("[R]")
    if not parent_a.HasSubstructMatch(ring_smarts) and not parent_b.HasSubstructMatch(
        ring_smarts
    ):
        return None

    reaction_smarts1 = [
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]-[*:2]",
        "[*:1]~[1*].[1*]~[*:2]>>[*:1]=[*:2]",
    ]
    reaction_smarts2 = [
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]-[*:2]",
        "([*:1]~[1*].[1*]~[*:2])>>[*:1]=[*:2]",
    ]

    for _ in range(num_trials):
        fragments_a = cut_ring(parent_a)
        fragments_b = cut_ring(parent_b)

        if fragments_a is None or fragments_b is None:
            return None

        new_mol_trial: List[Any] = []
        for rs in reaction_smarts1:
            reaction1 = Chem.AllChem.ReactionFromSmarts(rs)
            new_mol_trial = []
            for fa in fragments_a:
                for fb in fragments_b:
                    new_mol_trial.append(reaction1.RunReactants((fa, fb))[0])

        new_mols = []
        for rs in reaction_smarts2:
            reaction2 = Chem.AllChem.ReactionFromSmarts(rs)
            for m in new_mol_trial:
                m = m[0]
                if mol_is_ok(m):
                    new_mols += list(reaction2.RunReactants((m,)))

        new_mols2 = []
        for m in new_mols:
            m = m[0]
            if mol_is_ok(m) and ring_is_ok(m):
                new_mols2.append(m)

        if len(new_mols2) > 0:
            return random.choice(new_mols2)

    return None
