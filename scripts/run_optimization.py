"""
Main optimization script
"""

import argparse
import datetime
import random
from pathlib import Path

import torch
from torch.optim import Adam

from egegl.data import SmilesCharDictionary, load_dataset
from egegl.memory import FragmentLibrary, MaxRewardPriorityMemory, Recorder
from egegl.runners.trainer import Trainer as GeglTrainer
from egegl.runners import Generator
from egegl.scoring.benchmarks import load_benchmark
from egegl.utils.load_funcs import (
    load_apprentice_handler,
    load_explainer,
    load_explainer_handler,
    load_genetic_experts,
    load_logger,
    load_neural_apprentice,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="normal_runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./results/",
        help="Path to save the final neural apprentice and other models",
    )
    parser.add_argument(
        "--benchmark_id",
        type=int,
        default=28,
        help="Determines which benchmark to run against.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["zinc", "guacamol"],
        default="zinc",
        help="Sets the starting dataset",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/datasets/zinc/all.txt",
        help="Path to the dataset. Should correspond to the chosen 'dataset'",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["LSTM", "Transformer"],
        default="LSTM",
        help="Chooses the type of model for the neural apprentice.",
    )
    parser.add_argument(
        "--explainer_type",
        type=str,
        choices=["GCN", "DMPNN"],
        default=None,
        help="The model type of the explainer model",
    )
    parser.add_argument(
        "--max_smiles_length",
        type=int,
        default=100,
        help="The maximum allowed smiles string lenght.",
    )
    parser.add_argument(
        "--apprentice_load_dir",
        type=str,
        default="./data/pretrained_models/original_benchmarks/zinc",
        help="Path to the pretrained neural apprentice.",
    )
    parser.add_argument(
        "--explainer_load_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--genetic_experts", type=str, nargs="+", default=["SELFIES", "SMILES", "ATTR"]
    )
    parser.add_argument(
        "--logger_type",
        type=str,
        choices=["Neptune", "CommandLine"],
        default="CommandLine",
    )
    parser.add_argument("--project_qualified_name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--mutation_initial_rate", type=float, default=1e-2)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--num_keep", type=int, default=1024)
    parser.add_argument("--max_sampling_batch_size", type=int, default=1024)
    parser.add_argument("--apprentice_sampling_batch_size", type=int, default=8192)
    parser.add_argument("--expert_sampling_batch_size", type=int, default=8192)
    parser.add_argument("--apprentice_training_batch_size", type=int, default=256)
    parser.add_argument("--apprentice_training_steps", type=int, default=8)
    parser.add_argument("--num_smiles_for_similarity", type=int, default=100)
    parser.add_argument("--num_jobs", type=int, default=8)
    parser.add_argument("--record_filtered", action="store_true", default=False)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--use_frozen_explainer", action="store_true", default=False)
    parser.add_argument("--sampling_strategy", type=str, default="softmax")
    parser.add_argument("--seed", type=int, default=404)
    args = parser.parse_args()

    # Create save dir for the neural apprentice and explainer
    now = datetime.datetime.now()
    experiment_id = now.strftime("%y%m%d_%H%M")
    save_dir = args.save_root + "{}/".format(experiment_id)

    save_path = Path(save_dir)
    if not save_path.exists():
        save_path.mkdir(parents=True)

    # Prepare CUDA device if is set
    if args.use_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    random.seed(args.seed)

    # Get a logger
    tags = [args.dataset, args.benchmark_id, args.model_type, *args.genetic_experts]
    logger = load_logger(args, tags)

    # Load benchmark, character dictionary
    benchmark, scoring_num_list = load_benchmark(args.benchmark_id)
    char_dict = SmilesCharDictionary(
        dataset=args.dataset, max_smi_len=args.max_smiles_length
    )

    # Prepare max-reward memory
    apprentice_memory = MaxRewardPriorityMemory()
    expert_memory = MaxRewardPriorityMemory()

    # Load neural apprentice
    neural_apprentice = load_neural_apprentice(args)
    neural_apprentice.to(device)
    neural_apprentice.train()

    optimizer = Adam(neural_apprentice.parameters(), lr=args.learning_rate)

    apprentice_handler = load_apprentice_handler(
        model=neural_apprentice,
        optimizer=optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=args.max_sampling_batch_size,
        args=args,
    )

    # Load the explainer and explainer handler
    if args.explainer_type is not None:
        explainer = load_explainer(args, args.explainer_load_dir)
        explainer.to(device)
        explainer_optimizer = Adam(explainer.parameters(), lr=args.learning_rate)
        explainer_handler = load_explainer_handler(explainer, explainer_optimizer)
        fragment_library = FragmentLibrary(explainer_handler)
    else:
        explainer_handler = None
        fragment_library = None  # type: ignore

    # Load the genetic experts
    expert_handlers = load_genetic_experts(
        args.genetic_experts,
        args=args,
    )

    initial_smiles = None
    if args.model_type == "Transformer":
        dataset = load_dataset(char_dict=char_dict, smiles_path=args.dataset_path)
        initial_smiles = random.choices(population=dataset, k=args.num_keep)

    # Load the gegl-trainer
    trainer = GeglTrainer(
        apprentice_memory=apprentice_memory,
        expert_memory=expert_memory,
        apprentice_handler=apprentice_handler,
        expert_handlers=expert_handlers,
        explainer_handler=explainer_handler,
        fragment_library=fragment_library,
        char_dict=char_dict,
        num_keep=args.num_keep,
        apprentice_sampling_batch_size=args.apprentice_sampling_batch_size,
        expert_sampling_batch_size=args.expert_sampling_batch_size,
        sampling_strategy=args.sampling_strategy,
        apprentice_training_batch_size=args.apprentice_training_batch_size,
        apprentice_training_steps=args.apprentice_training_steps,
        num_smiles_for_similarity=args.num_smiles_for_similarity,
        init_smiles=initial_smiles if initial_smiles is not None else [],
        logger=logger,
        use_frozen_explainer=args.use_frozen_explainer,
    )

    # Load recorder
    recorder = Recorder(
        scoring_num_list=scoring_num_list,
        logger=logger,
        record_filtered=args.record_filtered,
    )

    generator = Generator(
        trainer=trainer,
        recorder=recorder,
        num_steps=args.num_steps,
        device=device,
        dataset_type=args.dataset,
        scoring_num_list=scoring_num_list,
        num_jobs=args.num_jobs,
    )

    final_result = benchmark.assess_model(generator)
    logger.log_metric("benchmark_score", final_result.score)

    # Save the neural apprentice and explainer if used
    neural_apprentice.save(save_dir)

    if explainer_handler is not None:
        explainer_handler.model.save(save_dir)
