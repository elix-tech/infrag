"""
Abstract Explainer class

Copyright (c) 2021 Elix, Inc.
"""

import json
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict

import torch


class AbstractExplainer(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self) -> None:
        super(AbstractExplainer, self).__init__()

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError

    @classmethod
    def load(cls, load_dir: str) -> "AbstractExplainer":
        model_config_path = Path(load_dir) / "explainer_config.json"
        with open(str(model_config_path), "r") as file:
            config = json.load(file)

        model = cls(**config)  # type: ignore
        model_weight_path = Path(load_dir) / "explainer_weight.pt"

        model_state_dict = torch.load(model_weight_path, map_location="cpu")

        try:
            model.load_state_dict(model_state_dict)
        except:
            print("Could not load pretrained weights for the explainer")

        return model

    def save(self, save_dir: str) -> None:
        model_config = self.config()
        model_config_path = Path(save_dir) / "explainer_config.json"
        with open(str(model_config_path), "w") as file:
            json.dump(model_config, file)

        model_state_dict = self.state_dict()
        model_weight_path = Path(save_dir) / "explainer_weight.pt"
        torch.save(model_state_dict, str(model_weight_path))

    @abstractmethod
    def config(self) -> Dict[str, Any]:
        raise NotImplementedError
