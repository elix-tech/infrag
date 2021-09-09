"""
Calculate the canonical smiles and scores for checking the
attribution method.

Saves the preprocessed data as a CSV file with both smiles representation
and score
"""

import argparse
from pathlib import Path

import pandas as pd
from joblib import Parallel

from egegl.data import SmilesCharDictionary, load_dataset
from egegl.scoring.benchmarks import load_benchmark
from egegl.utils.smiles import canonicalize_and_score_smiles

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument(
        "--dataset_path", type=str, default="./data/datasets/zinc/train.txt"
    )
    parser.add_argument("--save_root", default="./data/datasets/preprocessed/")
    parser.add_argument("--benchmark_id", type=int, required=True)
    parser.add_argument("--max_smiles_length", type=int, default=100)
    parser.add_argument("--num_jobs", type=int, default=1)
    args = parser.parse_args()

    save_dir = (
        args.save_root + "{}/".format(args.dataset) + "{}/".format(args.benchmark_id)
    )

    save_path = Path(save_dir)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    benchmark, _ = load_benchmark(args.benchmark_id)
    char_dict = SmilesCharDictionary(
        dataset=args.dataset, max_smi_len=args.max_smiles_length
    )
    dataset = load_dataset(char_dict=char_dict, smiles_path=args.dataset_path)

    pool = Parallel(n_jobs=args.num_jobs)
    canon_smiles, canon_scores = canonicalize_and_score_smiles(
        smiles=dataset,
        scoring_function=benchmark.wrapped_objective,
        char_dict=char_dict,
        pool=pool,
    )

    preprocessed_df = pd.DataFrame(
        list(zip(canon_smiles, canon_scores)), columns=["canonical_smiles", "score"]
    )
    preprocessed_df.to_csv(save_dir + "canonicalized_data.csv", index=False)
