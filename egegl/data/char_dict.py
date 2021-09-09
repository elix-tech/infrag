"""
Smiles character dictionaries

Copyright (c) 2021 Elix, Inc.
"""

from typing import Dict, List, Set, Union

import torch

PAD = " "
BEGIN = "Q"
END = "\n"


class SmilesCharDictionary:
    def __init__(self, dataset: str, max_smi_len: int) -> None:
        self.max_smi_len = max_smi_len

        self.forbidden_symbols = get_forbidden_symbols(dataset)
        self.char_idx = get_char_idx(dataset)
        self.idx_char = {v: k for k, v in self.char_idx.items()}

        self.encode_dict = get_encode_dict(dataset)
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    def is_allowed(self, smiles: str) -> bool:
        if len(smiles) > self.max_smi_len:
            return False

        for symbol in self.forbidden_symbols:
            if symbol in smiles:
                return False

        return True

    def encode(self, smiles: str) -> str:
        temp_smiles = smiles
        for symbol, token in self.encode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def decode(self, smiles: str) -> str:
        temp_smiles = smiles
        for symbol, token in self.decode_dict.items():
            temp_smiles = temp_smiles.replace(symbol, token)
        return temp_smiles

    def get_char_num(self) -> int:
        return len(self.idx_char)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[BEGIN]

    @property
    def end_idx(self) -> int:
        return self.char_idx[END]

    @property
    def pad_idx(self) -> int:
        return self.char_idx[PAD]

    @property
    def BEGIN(self):
        return BEGIN

    @property
    def END(self):
        return END

    @property
    def PAD(self):
        return PAD

    def matrix_to_smiles(self, array: torch.Tensor, seq_lengths: torch.Tensor):
        array_list = array.tolist()
        smis = list(
            map(
                lambda item: self.vector_to_smiles(item[0], item[1]),
                zip(array_list, seq_lengths),  # type: ignore
            )
        )
        return smis

    def vector_to_smiles(self, vec, seq_length):
        chars = list(map(self.idx_char.get, vec[:seq_length]))
        smi = "".join(chars)
        smi = self.decode(smi)
        return smi


def get_forbidden_symbols(dataset: str) -> Union[Dict, Set]:
    """
    Get forbidden symbols of the dataset
    """
    if dataset == "guacamol":
        forbidden_symbols = {
            "Ag",
            "Al",
            "Am",
            "Ar",
            "At",
            "Au",
            "D",
            "E",
            "Fe",
            "G",
            "K",
            "L",
            "M",
            "Ra",
            "Re",
            "Rf",
            "Rg",
            "Rh",
            "Ru",
            "T",
            "U",
            "V",
            "W",
            "Xe",
            "Y",
            "Zr",
            "a",
            "d",
            "f",
            "g",
            "h",
            "k",
            "m",
            "si",
            "t",
            "te",
            "u",
            "v",
            "y",
        }
    else:
        forbidden_symbols = set()

    return forbidden_symbols


def get_char_idx(dataset: str) -> Dict:
    if dataset == "guacamol":
        char_idx = {
            PAD: 0,
            BEGIN: 1,
            END: 2,
            "#": 20,
            "%": 22,
            "(": 25,
            ")": 24,
            "+": 26,
            "-": 27,
            ".": 30,
            "0": 32,
            "1": 31,
            "2": 34,
            "3": 33,
            "4": 36,
            "5": 35,
            "6": 38,
            "7": 37,
            "8": 40,
            "9": 39,
            "=": 41,
            "A": 7,
            "B": 11,
            "C": 19,
            "F": 4,
            "H": 6,
            "I": 5,
            "N": 10,
            "O": 9,
            "P": 12,
            "S": 13,
            "X": 15,
            "Y": 14,
            "Z": 3,
            "[": 16,
            "]": 18,
            "b": 21,
            "c": 8,
            "n": 17,
            "o": 29,
            "p": 23,
            "s": 28,
            "@": 42,
            "R": 43,
            "/": 44,
            "\\": 45,
            "E": 46,
        }
    elif dataset == "zinc":
        char_idx = {
            PAD: 0,
            BEGIN: 1,
            END: 2,
            "#": 3,
            "%": 40,
            "(": 4,
            ")": 5,
            "+": 6,
            "-": 7,
            "0": 8,
            "1": 9,
            "2": 10,
            "3": 11,
            "4": 12,
            "5": 13,
            "6": 14,
            "7": 15,
            "8": 16,
            "9": 17,
            "=": 18,
            "C": 19,
            "F": 20,
            "H": 21,
            "I": 22,
            "N": 23,
            "O": 24,
            "P": 25,
            "S": 26,
            "X": 27,
            "Y": 28,
            "[": 29,
            "]": 30,
            "c": 31,
            "n": 32,
            "o": 33,
            "p": 34,
            "s": 35,
            "@": 36,
            "R": 37,
            "/": 38,
            "\\": 39,
        }

    return char_idx


def get_encode_dict(dataset: str) -> Dict:
    if dataset == "guacamol":
        encode_dict = {"Br": "Y", "Cl": "X", "Si": "A", "Se": "Z", "@@": "R", "se": "E"}
    elif dataset == "zinc":
        encode_dict = {"Br": "Y", "Cl": "X", "Si": "A", "@@": "R"}
    else:
        raise ValueError(
            f"The dataset {dataset} is not valid.\
                         Please choose either 'guacomol' or 'zinc'"
        )

    return encode_dict
