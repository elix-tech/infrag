"""
Loading methods to ease object instantiation

Copyright (c) 2021 Elix, Inc.
"""

import logging
from typing import List, Optional, Union

import torch

from egegl.logger import CommandLineLogger, NeptuneLogger
from egegl.memory.fragment_lib import FragmentLibrary
from egegl.models.apprentice import LSTMGenerator, TransformerGenerator
from egegl.models.attribution import DirectedMessagePassingNetwork, GraphConvNetwork
from egegl.models.handlers import (
    ExplainerHandler,
    GeneticOperatorHandler,
    LSTMGeneratorHandler,
    TransformerGeneratorHandler,
)


def load_logger(args, tags=None):
    if args.logger_type == "Neptune":
        logger = NeptuneLogger(args, tags)
    elif args.logger_type == "CommandLine":
        logger = CommandLineLogger(args)
    else:
        raise NotImplementedError

    return logger


def load_neural_apprentice(args):
    if args.model_type == "LSTM":
        neural_apprentice = LSTMGenerator.load(load_dir=args.apprentice_load_dir)
    elif args.model_type == "Transformer":
        neural_apprentice = TransformerGenerator.load(load_dir=args.apprentice_load_dir)
    else:
        raise ValueError(f"{args.model_type} is not a valid model-type")

    return neural_apprentice


def load_apprentice_handler(model, optimizer, char_dict, max_sampling_batch_size, args):
    if args.model_type == "LSTM":
        apprentice_handler = LSTMGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    else:
        apprentice_handler = TransformerGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )

    return apprentice_handler


def load_genetic_experts(
    expert_types: List[str],
    args,
) -> List[GeneticOperatorHandler]:
    experts = []
    for ge_type in expert_types:
        expert_handler = GeneticOperatorHandler(
            crossover_type=ge_type,
            mutation_type=ge_type,
            mutation_initial_rate=args.mutation_initial_rate,
        )
        experts.append(expert_handler)
    return experts


def load_generator(input_size: int, args):
    if args.model_type == "LSTM":
        generator = LSTMGenerator(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=input_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    else:
        generator = TransformerGenerator(  # type: ignore
            n_token=input_size,
            n_embed=args.embed_size,
            n_head=args.n_head,
            n_hidden=args.hidden_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )

    return generator


def load_explainer(args, save_dir: str = None):
    explainer: Union[GraphConvNetwork, DirectedMessagePassingNetwork]
    if save_dir is None:
        logging.info("Loading the explainer without weights")
        if args.explainer_type == "MPNN":
            raise NotImplementedError
        elif args.explainer_type == "DMPNN":
            explainer = DirectedMessagePassingNetwork(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                edge_size=args.edge_size,
                steps=args.steps,
                dropout=args.dropout,
            )
        elif args.explainer_type == "GCN":
            explainer = GraphConvNetwork(
                input_size=args.input_size,
                hidden_size=args.hidden_size,
                output_size=args.output_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
        else:
            raise ValueError(f"The explainer_type {args.explainer_type} is invalid")

    elif isinstance(save_dir, str):
        logging.info("Loading the explainer from pretrained weights!")
        if args.explainer_type == "DMPNN":
            explainer = DirectedMessagePassingNetwork.load(save_dir)  # type: ignore
        elif args.explainer_type == "GCN":
            explainer = GraphConvNetwork.load(save_dir)  # type: ignore

    return explainer


def load_explainer_handler(
    model,
    optimizer,
):
    return ExplainerHandler(
        model=model,
        optimizer=optimizer,
    )
