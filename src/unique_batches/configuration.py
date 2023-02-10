# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from __future__ import annotations

from typing import List

from omegaconf import DictConfig


class GlobalConfiguration:
    _instance = None

    def __init__(self, cfg: DictConfig) -> None:
        self.hydra_config = cfg

        # Unpack config object

        self.debugging: bool = (
            "debug" in self.hydra_config and self.hydra_config["debug"]
        )
        self.logging: dict = self.hydra_config["logging"]

        self.data_params = cfg["data"]
        self.experiment_params = cfg["experiment"]
        self.model_params = cfg["model"]

        self.experiments_dir: str = cfg["experiments_dir"]

        self.random_state: int = cfg["random_state"]

        self.profiling: bool = False
        if "profiling" in cfg:
            self.profiling = cfg["profiling"]

        self.mixed_precision: bool = False
        if "mixed_precision" in cfg:
            self.mixed_precision = cfg["mixed_precision"]

        self.model_checkpointing: bool = False
        if "model_checkpointing" in cfg:
            self.model_checkpointing = cfg["model_checkpointing"]

        self.global_name: str = self.experiment_params["experiment_name"]
        self.deduplicator_names: str = list(self.experiment_params["deduplicator"])
        self.num_runs: int = self.experiment_params["num_runs"]
        self.max_epochs: int = self.experiment_params["max_epochs"]

        self.batch_size: int = self.experiment_params["batch_size"]
        self.early_stopping = self.experiment_params["early_stopping"]

        self.gradient_clip_val = self.experiment_params["gradient_clip_val"]

        self.model_names: List[str] = list(self.model_params.keys())

        self.deterministic: bool = False
        if "deterministic" in self.experiment_params:
            self.deterministic = self.experiment_params["deterministic"]

        self.random_states: List[int] = None  # To be filled at runtime
        self.current_experiment_path = None
        self.current_experiment_name = None
        self.current_deduplicator_name = None

        GlobalConfiguration._instance = self

    @staticmethod
    def get() -> GlobalConfiguration:
        assert GlobalConfiguration._instance is not None

        return GlobalConfiguration._instance
