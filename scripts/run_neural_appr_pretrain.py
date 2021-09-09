"""
Neural apprentice pretraining script
"""

import argparse
import datetime
import random
from pathlib import Path

import torch
from torch.optim import Adam

from egegl.data.char_dict import SmilesCharDictionary
from egegl.data.dataset import load_dataset
from egegl.runners import PreTrainer
from egegl.utils.load_funcs import load_apprentice_handler, load_generator, load_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="zinc")
    parser.add_argument(
        "--dataset_path", type=str, default="./data/datasets/zinc/train.txt"
    )
    parser.add_argument(
        "--model_type", type=str, choices=["LSTM", "Transformer"], default="LSTM"
    )
    parser.add_argument(
        "--logger_type",
        type=str,
        choices=["Neptune", "CommandLine"],
        default="CommandLine",
    )
    parser.add_argument("--project_qualified_name", type=str, default="")
    parser.add_argument("--max_smiles_length", type=int, default=80)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_root", default="./data/pretrained_models/")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    # Create a unique experimental id based on the execution timestamp
    now = datetime.datetime.now()
    experiment_id = now.strftime("%y%m%d_%H%M")

    save_dir = (
        args.save_root
        + "{}/".format(args.dataset)
        + "{}/".format(args.model_type)
        + "{}/".format(experiment_id)
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
    tags = [args.dataset, experiment_id]
    logger = load_logger(args, tags)

    # Load the dataset and valide characters
    char_dict = SmilesCharDictionary(
        dataset=args.dataset, max_smi_len=args.max_smiles_length
    )
    dataset = load_dataset(char_dict=char_dict, smiles_path=args.dataset_path)

    input_size = max(char_dict.char_idx.values()) + 1
    generator = load_generator(input_size, args)
    generator.to(device)

    optimizer = Adam(params=generator.parameters(), lr=args.learning_rate)

    generator_handler = load_apprentice_handler(
        model=generator,
        optimizer=optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=0,
        args=args,
    )

    trainer = PreTrainer(
        char_dict=char_dict,
        dataset=dataset,
        generator_handler=generator_handler,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_dir=save_dir,
        device=device,
        num_workers=args.num_workers,
        logger=logger,
    )
    trainer.pretrain()
