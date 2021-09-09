"""
Copyright (c) 2021 Elix, Inc.
"""

from pathlib import Path
from typing import List

from egegl.data.char_dict import SmilesCharDictionary


def load_dataset(char_dict: SmilesCharDictionary, smiles_path: str) -> List[str]:
    processed_dataset_path = (
        str(Path(smiles_path).with_suffix("")) + "_processed.smiles"
    )

    if Path(processed_dataset_path).exists():
        with open(processed_dataset_path, "r") as file:
            processed_dataset = file.read().splitlines()

    else:
        with open(smiles_path, "r") as file:
            dataset = file.read().splitlines()

        processed_dataset = list(filter(char_dict.is_allowed, dataset))
        with open(processed_dataset_path, "w") as file:
            file.writelines("\n".join(processed_dataset))

    return processed_dataset
