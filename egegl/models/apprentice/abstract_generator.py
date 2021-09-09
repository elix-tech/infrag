"""
Abstract class for Neural apprentices

Copyright (c) 2021 Elix, Inc.
"""

import json
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn


class AbstractGenerator(nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbstractGenerator, self).__init__()

    @classmethod
    def load(cls, load_dir: str) -> "AbstractGenerator":
        model_config_path = Path(load_dir) / "generator_config.json"
        with open(str(model_config_path), "r") as file:
            config = json.load(file)

        # Change the config keyword if original pretrained model is used
        if "lstm_dropout" in config.keys():
            config["dropout"] = config.pop("lstm_dropout")

        model = cls(**config)  # type: ignore
        model_weight_path = Path(load_dir) / "generator_weight.pt"

        try:
            model_state_dict = torch.load(model_weight_path, map_location="cpu")

            # Change the keyword if external state-dicts are used because of naming missmatch
            new_model_state_dict = OrderedDict()
            for name in model_state_dict.keys():
                if "rnn" in name:
                    new_model_state_dict[
                        name.replace("rnn", "lstm")
                    ] = model_state_dict[name]
                else:
                    new_model_state_dict[name] = model_state_dict[name]

            model_state_dict = new_model_state_dict
            model.load_state_dict(model_state_dict)
        except:
            print("No pretrained weight for SmilesGenerator")

        return model

    def save(self, save_dir: str) -> None:
        model_config = self.config()
        model_config_path = Path(save_dir) / "generator_config.json"
        with open(str(model_config_path), "w") as file:
            json.dump(model_config, file)

        model_state_dict = self.state_dict()
        model_weight_path = Path(save_dir) / "generator_weight.pt"
        torch.save(model_state_dict, str(model_weight_path))

    @abstractmethod
    def config(self) -> Dict:
        raise NotImplementedError
