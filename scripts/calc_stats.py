"""
Calculate some statistics for the logp optimization task
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, RDConfig
from tqdm import tqdm

sys.path.append(str(Path(RDConfig.RDContribDir, "SA_Score")))
import sascorer

from egegl.data import SmilesCharDictionary, load_dataset

if __name__ == "__main__":
    char_dict = SmilesCharDictionary(dataset="zinc", max_smi_len=81)
    dataset = load_dataset(
        char_dict=char_dict, smiles_path="./data/datasets/zinc/all.txt"
    )

    logp_scores, sa_scores, atomring_cycle_scores, cyclebasis_cycle_scores = (
        [],
        [],
        [],
        [],
    )

    for smi in tqdm(dataset):
        mol = Chem.MolFromSmiles(smi)

        logp_scores.append(Descriptors.MolLogP(mol))
        sa_scores.append(sascorer.calculateScore(mol))

        cycle_list = mol.GetRingInfo().AtomRings()
        max_ring_size = max([len(cycle) for cycle in cycle_list]) if cycle_list else 0
        atomring_cycle_scores.append(max(max_ring_size - 6, 0))

        cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
        max_ring_size = max([len(cycle) for cycle in cycle_list]) if cycle_list else 0
        cyclebasis_cycle_scores.append(max(max_ring_size - 6, 0))

    # Log the results:
    print(f"LogP stats: {np.mean(logp_scores)}, {np.std(logp_scores)}")
    print(f"SA stats: {np.mean(sa_scores)}, {np.std(sa_scores)}")
    print(
        f"AtomRing stats: {np.mean(atomring_cycle_scores)}, {np.std(atomring_cycle_scores)}"
    )
    print(
        f"CycleBasis stats: {np.mean(cyclebasis_cycle_scores)}, {np.std(cyclebasis_cycle_scores)}"
    )
