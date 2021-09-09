"""
Script to get low scoring smiles strings
"""

import argparse
from pathlib import Path

from tqdm import tqdm

from egegl.data import SmilesCharDictionary, load_dataset
from egegl.scoring.benchmarks import load_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="low_logp", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument("--top_k", type=int, default=800)
    parser.add_argument("--benchmark_id", type=int, default=28)
    parser.add_argument(
        "--dataset_path", type=str, default="./data/datasets/zinc/test.txt"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./data/assets/low_logp_smiles.smi",
    )
    parser.add_argument("--max_smiles_length", type=int, default=80)
    args = parser.parse_args()

    # Create save directory if it doesn't exist
    save_dir = Path(args.output_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    char_dict = SmilesCharDictionary(
        dataset=args.dataset, max_smi_len=args.max_smiles_length
    )
    dataset = load_dataset(char_dict=char_dict, smiles_path=args.dataset_path)
    benchmark, scoring_num_list = load_benchmark(benchmark_id=args.benchmark_id)

    smi2score = {}
    for smile in tqdm(dataset):
        score = benchmark.wrapped_objective.score(smile)
        smi2score[smile] = score

    low_scoring_smiles = sorted(dataset, key=lambda smile: smi2score[smile])[
        : args.top_k
    ]

    with open(args.output_path, "w") as file:
        file.write("\n".join(low_scoring_smiles) + "\n")

