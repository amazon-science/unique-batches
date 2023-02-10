#   Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# This work is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License.
# To view a copy of this license, visit http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import gc
import warnings
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from unique_batches.constants import CONFIG_NAME, CONFIG_PATH
from unique_batches.framework import (GlobalConfiguration, aggregate_results,
                                      build_datamodule, build_model,
                                      build_test_trainer, build_trainer,
                                      build_trainer_logger, finish_logger,
                                      get_encoder_for_model, init_logger,
                                      seed_everything)
from unique_batches.utils.framework import get_num_gpus
from unique_batches.utils.io import write_json
from unique_batches.utils.logging import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(hydra_config: DictConfig):

    logger.info("Parsing Hydra config...")
    cfg = GlobalConfiguration(hydra_config)

    seed = cfg.random_state
    logger.info(f"Setting initial global seed with random state: {seed}")
    seed_everything(seed)

    for deduplicator_name in cfg.deduplicator_names:
        logger.info(f"Running deduplicator: {deduplicator_name}")
        cfg.current_deduplicator_name = deduplicator_name

        for model_name in cfg.model_names:
            logger.info(f"Running model: {model_name}")

            # Build datamodule
            logger.info("Building datamodule...")
            datamodule = build_datamodule(deduplicator_name)

            # Experiment name and path
            global_experiment_name = cfg.global_name
            dataset_name = Path(cfg.hydra_config["data"]["dataset_path"]).stem
            current_experiment_name = f"{dataset_name}/{global_experiment_name}/{deduplicator_name}/{model_name}"
            logger.info(f"Current experiment: {current_experiment_name}")

            current_experiment_path = (
                Path(cfg.experiments_dir) / current_experiment_name
            )
            current_experiment_path = current_experiment_path.resolve()
            current_experiment_path.mkdir(parents=True, exist_ok=True)

            # Preprocess datamodule with current encoder
            logger.info("Preparing datamodule for current experiment...")
            seed = cfg.random_state
            pl.seed_everything(seed)
            encoder = get_encoder_for_model(model_name)
            datamodule.set_encoder(encoder)
            datamodule.preprocess()

            # datamodule_path = current_experiment_path / "datamodule.pt"
            # torch.save(datamodule.state_dict(), datamodule_path)

            # Train and evaluate model for N runs
            results = list()
            for i in range(cfg.num_runs):
                # Init wandb
                init_logger(current_experiment_name, global_experiment_name)

                logger.info(f"Run {i+1}...")

                # Build model for this run
                seed = cfg.random_states[i]
                pl.seed_everything(seed)

                logger.info("Building model for this run...")
                model = build_model(model_name, datamodule)

                # # Avoid overwriting
                # if current_experiment_path.exists():
                #     current_experiment_path = get_new_path(current_experiment_path)

                cfg.current_experiment_path = current_experiment_path
                cfg.current_experiment_name = current_experiment_name

                logger.info(f"Saving artifacts at: {current_experiment_path}")

                # Train model for this run
                logger.info("Training model...")

                trainer = build_trainer(cfg)
                trainer.fit(model=model, datamodule=datamodule)

                time_steps_metrics = {
                    "train/total_train_val_steps": model.logged_metrics[
                        "train/total_train_val_steps"
                    ],
                    "train/total_train_val_time": model.logged_metrics[
                        "train/total_train_val_time"
                    ],
                }

                # Evaluate model for this run
                trainer_logger = trainer.logger
                test_trainer = build_test_trainer(trainer_logger)

                cur_results_list = test_trainer.test(model=model, datamodule=datamodule)
                cur_results = cur_results_list[0]

                cur_results.update(time_steps_metrics)

                results.append(cur_results)

                results_path = current_experiment_path / f"results_{i}.json"
                write_json(cur_results, results_path)

                finish_logger()

            aggr_results_path = current_experiment_path / "results.json"
            aggr_results = aggregate_results(results)
            write_json(aggr_results, aggr_results_path)

            # del datamodule
            # del model

            # torch.cuda.empty_cache()
            # gc.collect()


if __name__ == "__main__":
    main()
