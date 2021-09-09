"""
Graph features for explainer

Copyright (c) 2021 Elix, Inc.
"""

from typing import Any, List, Optional

from rdkit import Chem


def one_hot_encoding(
    x: Any, allowable_set: List[Any], encode_unknown: bool = False
) -> List[bool]:
    if encode_unknown and (allowable_set[-1] is not None):
        allowable_set.append(None)

    if encode_unknown and (x not in allowable_set):
        x = None

    return list(map(lambda s: x == s, allowable_set))


def atom_type_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[Any]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    if allowable_set is None:
        allowable_set = [
            "C",
            "N",
            "O",
            "S",
            "F",
            "Si",
            "P",
            "Cl",
            "Br",
            "Mg",
            "Na",
            "Ca",
            "Fe",
            "As",
            "Al",
            "I",
            "B",
            "V",
            "K",
            "Tl",
            "Yb",
            "Sb",
            "Sn",
            "Ag",
            "Pd",
            "Co",
            "Se",
            "Ti",
            "Zn",
            "H",
            "Li",
            "Ge",
            "Cu",
            "Au",
            "Ni",
            "Cd",
            "In",
            "Mn",
            "Zr",
            "Cr",
            "Pt",
            "Hg",
            "Pb",
        ]
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)


def atom_degree_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[Any]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)


def atom_implicit_valence_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[Any]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)


def atom_formal_charge(atom: Chem.Atom) -> List[float]:
    return [atom.GetFormalCharge()]


def atom_num_radical_electrons(atom: Chem.Atom) -> List[int]:
    return [atom.GetNumRadicalElectrons()]


def atom_hybridization_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[Any]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)


def atom_is_aromatic(atom):
    return [atom.GetIsAromatic()]


def atom_total_num_H_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[Any]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)


def bond_type_one_hot(
    bond: Chem.Bond,
    allowable_set: Optional[List[Any]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)


def bond_is_conjugated(bond: Chem.Bond) -> List[bool]:
    return [bond.GetIsConjugated()]


def bond_is_in_ring(bond: Chem.Bond) -> List[bool]:
    return [bond.IsInRing()]


def bond_stereo_one_hot(
    bond: Chem.Bond,
    allowable_set: Optional[List[Any]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
        ]
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)

