"""
Script to pretrain the InFrag explainer model
"""

import argparse
import datetime
import random
from pathlib import Path

import pandas as pd
import torch
from torch.optim import Adam

from egegl.runners import ExplainerPreTrainer
from egegl.scoring.benchmarks import load_benchmark
from egegl.utils.load_funcs import load_explainer, load_explainer_handler, load_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain_explainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, required=True, choices["zinc", "guacamol"])
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument(
        "--explainer_type", type=str, choices=["GCN", "DMPNN"], default="GCN"
    )
    parser.add_argument(
        "--logger_type",
        type=str,
        choices=["Neptune", "CommandLine"],
        default="CommandLine",
    )
    parser.add_argument("--input_size", type=int, default=74)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--edge_size", type=int, default=12)
    parser.add_argument("--edge_hidden_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_root", default="./data/pretrained_models/")
    parser.add_argument("--benchmark_id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--project_qualified_name", type=str, default="")
    args = parser.parse_args()

    save_dir = (
        args.save_root
        + "explainer/"
        + "simplified/"
        + "{}/".format(args.benchmark_id)
        + "{}/".format(args.explainer_type)
    )

    save_path = Path(save_dir)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Prepare CUDA device if set
    if args.use_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")
    random.seed(args.seed)

    # Initialize logger
    tags = [args.dataset, args.benchmark_id]
    logger = load_logger(args, tags)

    # Load the benchmark and dataset
    benchmark, _ = load_benchmark(args.benchmark_id)

    explainer = load_explainer(args)
    explainer.to(device)

    optimizer = Adam(explainer.parameters(), lr=args.learning_rate)

    explainer_handler = load_explainer_handler(
        model=explainer,
        optimizer=optimizer,
    )

    dataset_df = pd.read_csv(args.dataset_path)
    canon_smiles = dataset_df["canonical_smiles"].to_list()
    canon_scores = dataset_df["score"].to_list()

    trainer = ExplainerPreTrainer(
        canon_smiles=canon_smiles,
        canon_scores=canon_scores,
        explainer_handler=explainer_handler,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_dir=save_dir,
        num_workers=args.num_workers,
        device=device,
        logger=logger,
    )
    trainer.pretrain()
