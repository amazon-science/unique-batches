# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (Callback, EarlyStopping,
                                         ModelCheckpoint, TQDMProgressBar)
from pytorch_lightning.loggers import Logger, TensorBoardLogger, WandbLogger
from pytorch_lightning.profiler import AdvancedProfiler

from unique_batches.configuration import GlobalConfiguration
from unique_batches.data.datamodule import NERDataModule
from unique_batches.data.encoders.encoder import Encoder
from unique_batches.data.encoders.factory import get_encoder
from unique_batches.models.factory import get_model, get_model_config
from unique_batches.models.model import Model
from unique_batches.utils.framework import get_num_gpus
from unique_batches.utils.logging import get_logger

logger = get_logger(__name__)


def seed_everything(random_state: int):
    cfg = GlobalConfiguration.get()

    pl.seed_everything(random_state)

    if cfg.deterministic:
        torch.use_deterministic_algorithms(True)

    # This random states sequence will be reproducible after setting initial seed above
    cfg.random_states = [random_state]

    if cfg.num_runs > 1:
        cfg.random_states = [
            random.randint(0, 2**32 - 1) for _ in range(cfg.num_runs)
        ]


def init_logger(*args, **kwargs) -> None:
    cfg = GlobalConfiguration.get()

    if cfg.logging["logger"] == "wandb":
        init_wandb(*args, **kwargs)
    elif cfg.logging["logger"] == "tensorboard":
        pass
    else:
        raise NotImplementedError


def finish_logger(*args, **kwargs) -> None:
    cfg = GlobalConfiguration.get()

    if cfg.logging["logger"] == "wandb":
        finish_wandb(*args, **kwargs)
    elif cfg.logging["logger"] == "tensorboard":
        pass
    else:
        raise NotImplementedError


def init_wandb(name: str, group: str = None) -> None:
    cfg = GlobalConfiguration.get()

    if cfg.debugging:
        return

    # Integrate wandb and hydra
    hydra_config = OmegaConf.to_container(
        cfg.hydra_config, resolve=True, throw_on_missing=True
    )

    wandb.init(
        project=cfg.logging["project_name"],
        config=hydra_config,
        name=name,
        group=group,
        reinit=True,
    )


def finish_wandb() -> None:
    cfg = GlobalConfiguration.get()

    if cfg.debugging:
        return

    assert wandb.run is not None

    wandb.run.finish()


def build_datamodule(deduplicator_name: str) -> NERDataModule:
    cfg = GlobalConfiguration.get()

    dataset_path = Path(cfg.data_params["dataset_path"]).resolve()
    unique_mode = cfg.data_params["unique_mode"]

    shuffle_train = cfg.hydra_config["data"]["shuffle"] and not cfg.deterministic

    datamodule = NERDataModule(
        dataset_path=dataset_path,
        deduplicator_name=deduplicator_name,
        batch_size=cfg.batch_size,
        random_state=cfg.random_state,
        shuffle_train=shuffle_train,
        unique_mode=unique_mode,
    )

    return datamodule


def build_model(model_name: str, datamodule: NERDataModule) -> Model:
    cfg = GlobalConfiguration.get()

    model_config = get_model_config(model_name, datamodule, cfg.hydra_config)
    model = get_model(model_name, model_config=model_config)

    return model


def get_encoder_for_model(model_name: str) -> Encoder:
    cfg = GlobalConfiguration.get()

    encoder_config = dict()
    if model_name == "lstm_based":
        encoder_name = "word_encoder"
    elif model_name == "cnn_based":
        encoder_name = "char_encoder"
    elif model_name == "bert_based":
        encoder_name = "wordpiece_encoder"

        bert_params = cfg.model_params[model_name]
        checkpoint = bert_params["encoder_checkpoint"]
        encoder_config.update({"checkpoint": checkpoint})
    else:
        raise NotImplementedError

    logger.info(f"Using encoder: {encoder_name}")

    encoder = get_encoder(encoder_name, **encoder_config)

    return encoder


def build_callbacks(cfg: GlobalConfiguration) -> List[Callback]:

    # callbacks = [TQDMProgressBar(refresh_rate=20)]
    callbacks = [TQDMProgressBar(refresh_rate=1)]

    # saves top-K checkpoints based on "val_loss" metric
    if cfg.model_checkpointing:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val/loss",
            mode="min",
            dirpath=cfg.current_experiment_path,
            filename="model",
        )
        callbacks.append(checkpoint_callback)

    # Early stopping on validation loss
    if cfg.early_stopping["enabled"]:
        logger.info("Using early stopping")

        monitor = cfg.early_stopping["monitor"]
        min_delta = cfg.early_stopping["min_delta"]
        patience = cfg.early_stopping["patience"]
        mode = cfg.early_stopping["mode"]

        early_stop_callback = EarlyStopping(
            monitor=monitor,
            min_delta=min_delta,
            patience=patience,
            verbose=False,
            mode=mode,
        )
        callbacks.append(early_stop_callback)

    return callbacks


def build_trainer_logger(cfg: GlobalConfiguration) -> Logger:

    # Will log metrics
    if cfg.logging["logger"] == "wandb":
        trainer_logger = WandbLogger(save_dir=cfg.current_experiment_path)
    elif cfg.logging["logger"] == "tensorboard":
        trainer_logger = TensorBoardLogger(save_dir=cfg.current_experiment_path)
    else:
        raise NotImplementedError

    return trainer_logger


def build_trainer(cfg: GlobalConfiguration) -> Trainer:

    # -- Global params
    max_epochs = cfg.max_epochs
    num_gpus = get_num_gpus()

    # -- Callbacks
    callbacks = build_callbacks(cfg)

    # -- Logger
    trainer_logger = build_trainer_logger(cfg)

    # -- Config-dependent trainer args
    trainer_args = dict()

    # Accelerator
    if num_gpus == 1:
        trainer_args.update(
            {
                "accelerator": "gpu",
                "devices": 1,
            }
        )
    elif num_gpus > 1:
        trainer_args.update(
            {
                "accelerator": "gpu",
                "devices": num_gpus,
                "strategy": "dp",
                # "strategy": "ddp",
            }
        )
    else:
        trainer_args.update(
            {
                "accelerator": None,
            }
        )

    # Profiling
    if cfg.profiling:
        profiler_dir_path = cfg.current_experiment_path
        profiler_filename = "trainer_profiling.txt"
        profiler = AdvancedProfiler(
            dirpath=profiler_dir_path, filename=profiler_filename
        )

        trainer_args.update({"profiler": profiler})

    # Determinsm
    if cfg.deterministic:
        trainer_args["deterministic"] = True

    # Precision
    if cfg.mixed_precision:
        trainer_args.update(dict(precision=16, amp_backend="native"))

    # Reload dataloaders
    if cfg.current_deduplicator_name in {
        "batchwise_weighted_unique",
        "batchwise_unique",
    }:
        trainer_args.update(dict(reload_dataloaders_every_n_epochs=1))

    # Checkpointing
    if not cfg.model_checkpointing:
        trainer_args.update(dict(enable_checkpointing=False))

    # Gradient clipping
    trainer_args.update(dict(gradient_clip_val=cfg.gradient_clip_val))

    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
        logger=trainer_logger,
        **trainer_args,
    )

    return trainer


def build_test_trainer(trainer_logger: Logger) -> Trainer:

    if get_num_gpus() > 0:
        tester_args = {
            "accelerator": "gpu",
            "devices": 1,
        }
    else:
        tester_args = {
            "accelerator": None,
        }

    test_trainer = Trainer(log_every_n_steps=1, logger=trainer_logger, **tester_args)

    return test_trainer


def aggregate_results(results_list: List[dict]) -> dict:

    # unpack metrics
    metric_results = defaultdict(list)

    for result in results_list:
        for metric, val in result.items():
            metric_results[metric].append(val)

    # compute mean and std
    aggr_results = defaultdict(dict)

    for metric, vals in metric_results.items():
        mean = np.mean(vals)
        std = np.std(vals)

        aggr_results[metric]["mean"] = mean
        aggr_results[metric]["std"] = std

    return aggr_results
