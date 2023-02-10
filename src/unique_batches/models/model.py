# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import math
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_lightning.core.module import LightningModule
from torch.optim import Adam
from torchmetrics import F1Score, Precision, Recall

from unique_batches.configuration import GlobalConfiguration
from unique_batches.data.vocab import Vocab
# from unique_batches.models.losses import cross_entropy, weighted_cross_entropy
from unique_batches.models.losses import (CrossEntropyLoss,
                                          FrequencyAwareCrossEntropyLoss)
from unique_batches.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelConfig:
    model_name: str = None
    num_tags: int = None
    tag_vocab: Vocab = None
    hidden_size: int = None
    deduplicator_name: str = None
    lr: int = None
    lr_increase_factor: int = None
    log_grads: bool = None
    log_params: bool = None
    params_logs_per_epoch: int = None

    def __init__(self, **kwargs):

        FIELD_NAMES = {field.name for field in fields(self)}

        for key, value in kwargs.items():
            if key in FIELD_NAMES:
                setattr(self, key, value)


class Model(LightningModule, ABC):
    def __init__(self, model_config: ModelConfig):

        super().__init__()

        self.model_name = model_config.model_name
        self.hidden_size = model_config.hidden_size

        self.deduplicator_name = model_config.deduplicator_name
        self.log_grads = model_config.log_grads
        self.log_params = model_config.log_params

        self.lr = model_config.lr
        self.lr_increase_factor = model_config.lr_increase_factor

        if self._is_frequencies_aware():
            self.criterion = FrequencyAwareCrossEntropyLoss()
        else:
            self.criterion = CrossEntropyLoss()

        tag_vocab = model_config.tag_vocab
        ignore_index = tag_vocab[tag_vocab.pad_symbol]
        self.metrics = nn.ModuleDict(
            {
                "F1": F1Score(
                    task="multiclass",
                    num_classes=model_config.num_tags,
                    average="micro",
                    ignore_index=ignore_index,
                ),
                "prec": Precision(
                    task="multiclass",
                    num_classes=model_config.num_tags,
                    average="micro",
                    ignore_index=ignore_index,
                ),
                "recall": Recall(
                    task="multiclass",
                    num_classes=model_config.num_tags,
                    average="micro",
                    ignore_index=ignore_index,
                ),
            }
        )

        self.logged_metrics = dict()

        self.num_training_steps_so_far = 0

        self.grads = []
        self.parameter_vectors = []

        if self.log_params:
            self.params_logs_per_epoch = model_config.params_logs_per_epoch

        self.save_hyperparameters(logger=False)

    @abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    # MAIN
    def configure_optimizers(self):
        if self.deduplicator_name == "batchwise_weighted_unique":
            self.lr *= self.lr_increase_factor
        return Adam(self.parameters(), self.lr)

    def training_step(self, batch, batch_idx):
        targets = batch["tags"]
        frequencies = batch["frequencies"] if self._is_frequencies_aware() else None

        logits = self._compute_logits(batch)
        loss = self._compute_loss(logits, targets, frequencies)

        self.log(
            "train/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(targets),
        )

        return loss

    def validation_step(self, batch, batch_idx):
        targets = batch["tags"]
        frequencies = batch["frequencies"] if self._is_frequencies_aware() else None

        logits = self._compute_logits(batch)

        loss = self._compute_loss(logits, targets, frequencies)

        self.log("val/loss", loss, on_step=True, on_epoch=True, batch_size=len(targets))

        # preds = torch.argmax(logits, -1)
        # for metric_name, metric_fn in self.metrics.items():
        #     metric_value = metric_fn(preds, targets)
        #     self.log(
        #         f"val/{metric_name}",
        #         metric_value,
        #         on_step=True,
        #         on_epoch=True,
        #         batch_size=len(targets),
        #     )

        return {"y": targets, "logits": logits, "frequencies": frequencies}

    def test_step(self, batch, batch_idx):
        y_preds = self._compute_preds(batch)
        y_true = batch["tags"]
        self._log_test_metrics(y_true.reshape(-1), y_preds.reshape(-1))

        return {"pred_tokens": y_preds, "tags": batch["tags"]}

    def reset(self):
        for layer in self.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        self.num_training_steps_so_far = 0
        self.grads = list()
        self.parameter_vectors = list()

    # HOOKS
    def on_train_start(self) -> None:
        logger.info("Training start")

        self.training_start_time = time.time()

        if self.log_params:
            self.log_params_interval = (
                self._num_training_steps_per_epoch() // self.params_logs_per_epoch
            )

    def on_train_epoch_start(self) -> None:
        self.epoch_type = "train"
        self.epoch_start_time = time.time()

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.train_batch_time = time.time()

        # Log current parameter vector
        if self.log_params:
            log_interval = max(
                1,
                math.floor(
                    self._num_training_steps_per_epoch() / self.params_logs_per_epoch
                ),
            )

            if self.num_training_steps_so_far % log_interval == 0:
                with torch.no_grad():
                    param_vector = (
                        torch.nn.utils.convert_parameters.parameters_to_vector(
                            self.parameters()
                        ).cpu()
                    )
                    self.parameter_vectors.append(param_vector)

    def on_train_epoch_end(self) -> None:
        f"""
        If {self.log_grads}, it stacks the batch gradients into a tensor
        of epoch gradients, computes and logs their stats.
        """
        if self.log_grads:
            assert len(self.grads) > 0

            epoch_grads = torch.stack(self.grads)

            self._compute_variance(epoch_grads)

            self.grads = list()

        if self.log_params:
            if len(self.parameter_vectors) == 0:
                return

            epoch_params = torch.stack(self.parameter_vectors)

            cfg = GlobalConfiguration.get()
            exp_path = Path(cfg.current_experiment_path)
            path = exp_path / f"params_vector_epoch_{self.current_epoch}.pt"
            torch.save(epoch_params, path)

            self.parameter_vectors = list()

    def _num_training_steps_per_epoch(self) -> int:
        """Total training steps (per epoch) inferred from datamodule and devices."""
        if self.trainer.max_steps > 0:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.train_dataloader)
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_devices, self.trainer.num_nodes)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return batches // effective_accum

    def _num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        return self._num_training_steps_per_epoch() * self.trainer.max_epochs

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        train_batch_time = time.time() - self.train_batch_time
        self.log("train/train_step_time", train_batch_time, on_step=True)

        self.num_training_steps_so_far += 1

    def on_validation_epoch_end(self) -> None:

        self.logged_metrics.update(self.trainer.logged_metrics)

    def on_train_end(self) -> None:
        training_time = time.time() - self.training_start_time

        logger.info("Training end")

        # Workaround since apparently ``can't log in on_train_end``
        steps_dict = {"train/total_train_val_steps": self.num_training_steps_so_far}
        self.trainer.logger.log_metrics(steps_dict)
        self.logged_metrics.update(steps_dict)

        time_dict = {"train/total_train_val_time": training_time}
        self.trainer.logger.log_metrics(time_dict)
        self.logged_metrics.update(time_dict)

        if self.log_params:

            with torch.no_grad():
                final_params = torch.nn.utils.convert_parameters.parameters_to_vector(
                    self.parameters()
                ).cpu()

            cfg = GlobalConfiguration.get()
            exp_path = Path(cfg.current_experiment_path)
            path = exp_path / f"params_vector_epoch_{self.current_epoch}.pt"
            torch.save(final_params, path)

            self.parameter_vectors = list()

    def on_after_backward(self) -> None:
        f"""
        If {self.log_grads}, stores the batch gradients and
        computes their statistics
        """

        if self.log_grads:

            batch_grads = self._collect_batch_grads()
            self.grads.append(batch_grads)

            batch_grads_stats = {
                "mean_grads": torch.mean(batch_grads, dim=0),
                "stddev_grads": torch.std(batch_grads, dim=0),
            }

            self.log_dict(batch_grads_stats)

    # PRIVATE
    def _compute_logits(self, batch):
        logits = self(**batch)

        return logits

    def _compute_preds(self, batch):
        logits = self._compute_logits(batch)

        preds = torch.argmax(logits, -1)

        return preds

    def _collect_batch_grads(self):
        """
        Collects the gradients of the parameters of each layer,
         flattens them and collates them together in a single tensor.
        """
        all_grads = []
        for name, param in self.named_parameters():
            # (num_params_layer, )
            flattened_grads = param.grad.reshape(-1)
            all_grads.append(flattened_grads)

        # (total_num_params, )
        batch_grads = torch.cat(all_grads, dim=0)
        return batch_grads

    def _log_test_metrics(self, tags, pred_tokens):

        for metric_name, metric in self.metrics.items():

            log_name = f"test/{metric_name}"
            metric_result = metric(pred_tokens, tags)

            self.log(
                log_name,
                metric_result,
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=True,
                batch_size=len(tags),
            )

    def _compute_loss(self, logits, targets, frequencies=None):
        logits = logits.transpose(-1, -2)
        assert logits.shape[-1] == targets.shape[-1]

        if self._is_frequencies_aware():
            loss = self.criterion(logits, targets, frequencies)
        else:
            loss = self.criterion(logits, targets)

        return loss

    def _is_frequencies_aware(self):
        return self.deduplicator_name in {
            "batchwise_weighted_unique",
            "datasetwise_weighted_unique",
        }

    def _compute_variance(self, epoch_grads):
        var = torch.var(epoch_grads, dim=0)

        var_stats = {
            "min_var": torch.min(var),
            "max_var": torch.max(var),
            "mean_var": torch.mean(var),
        }

        self.log_dict(var_stats)
