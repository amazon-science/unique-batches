# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from pathlib import Path
from typing import Dict

import pandas as pd
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from unique_batches.constants import (DEFAULT_SPLIT_RATIO, DOMAIN,
                                      RARE_DOMAIN_SIZE, TAGS, UTTERANCE_TEXT)
from unique_batches.data.datasets.dataset import NERDataset
from unique_batches.data.deduplicators.deduplicator import Deduplicator
from unique_batches.data.deduplicators.factory import get_deduplicator
from unique_batches.data.encoders.encoder import Encoder
from unique_batches.data.sample import Sample
from unique_batches.data.vocab import Vocab
from unique_batches.utils.data import find_expected_virtual_batch_size
from unique_batches.utils.framework import Stage, create_index, get_num_workers
from unique_batches.utils.io import read_token_level_annotated_data
from unique_batches.utils.logging import get_logger

logger = get_logger(__name__)


class NERDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_path: Path,
        deduplicator_name: str,
        batch_size: int,
        random_state: int,
        shuffle_train: bool,
        unique_mode: str = "text",
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.deduplicator_name = deduplicator_name
        self.batch_size = batch_size
        self.random_state = random_state
        self.shuffle_train = shuffle_train
        self.unique_mode = unique_mode

        # Load data
        logger.info(f"Reading data from {self.dataset_path}...")
        self.data = read_token_level_annotated_data(self.dataset_path)

        # To be filled during preprocessing
        self.preprocessed = False
        self.split_data: Dict[Stage, pd.DataFrame] = dict()
        self.vocabularies: Dict[str, Vocab] = dict()
        self.max_word_len: int = -1
        self.datasets: Dict[Stage, NERDataset] = dict()
        self.encoder = None

        # To be filled during setup
        self.deduplicators: Dict[Stage, Deduplicator] = dict()

    @property
    def annotated(self):
        assert self.data is not None

        return TAGS in self.data.columns

    @property
    def expected_virtual_batch_size(self):
        if not hasattr(self, "_expected_virtual_batch_size"):
            train_data = self.split_data[Stage.train]
            self._expected_virtual_batch_size = find_expected_virtual_batch_size(
                train_data, self.batch_size
            )

        return self._expected_virtual_batch_size

    @property
    def virtual_over_batch_size_ratio(self):
        if self.deduplicator_name not in {
            "datasetwise_weighted_unique",
            "batchwise_weighted_unique",
        }:
            return 1

        # train_data = self.split_data[Stage.train]
        # original_num_batches = math.ceil(len(train_data) / self.batch_size)
        # new_num_batches = math.ceil(len(train_data) / self.expected_virtual_batch_size)

        virtual_over_batch_size_ratio = (
            self.expected_virtual_batch_size / self.batch_size
        )

        return virtual_over_batch_size_ratio

    def preprocess(self):

        # Remove domains with less than K utts
        logger.info(f"Removing rare domains...")
        self._delete_rare_domains()

        # Build vocabularies
        logger.info(f"Building vocabularies...")
        self._build_vocabularies()

        # Split data into (train, val, test)
        logger.info(f"Splitting data...")
        self._train_val_test_split()

        # Prepare (train, val, test) datasets (vocab -> encoding)
        logger.info(f"Preparing datasets...")
        self._prepare_datasets()

        self.preprocessed = True

    def _delete_rare_domains(self):
        cutoff_size = RARE_DOMAIN_SIZE
        for domain, size in self.data.groupby(DOMAIN).size().iteritems():
            if size < cutoff_size:
                self.data = self.data.drop(self.data[self.data[DOMAIN] == domain].index)

    def _train_val_test_split(self):

        train_split, val_split, test_split = (
            DEFAULT_SPLIT_RATIO[Stage.train],
            DEFAULT_SPLIT_RATIO[Stage.val],
            DEFAULT_SPLIT_RATIO[Stage.test],
        )

        # Split into (train, val_test)
        self.split_data[Stage.train], data_valtest = train_test_split(
            self.data,
            test_size=val_split + test_split,
            random_state=self.random_state,
        )

        # Split into (val, test)
        test_size = test_split / (val_split + test_split)
        self.split_data[Stage.val], self.split_data[Stage.test] = train_test_split(
            data_valtest,
            test_size=test_size,
            random_state=self.random_state,
        )

        for stage, data in self.split_data.items():
            self.split_data[stage] = data.reset_index(drop=True)

    def setup(self, stage: str):

        assert self.preprocessed

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            logger.info("Setting up deduplicators...")
            self._setup_deduplicators()

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

    def set_encoder(self, encoder: Encoder):
        self.encoder = encoder

    def _build_vocabularies(self):
        self._compute_max_len()

        self.vocabularies["word"] = self._build_word_vocab()
        self.vocabularies["char"] = self._build_char_vocab()

        if self.annotated:
            tag_vocab = self._build_tag_vocab()
            self.vocabularies["tag"] = tag_vocab

    def _build_tag_vocab(self) -> None:
        tags = set()

        for tags_utt in self.data[TAGS]:
            for tag in tags_utt:
                tags.add(tag)

        tag_to_idx = create_index(tags, offset=1)
        tag_to_idx["PAD"] = 0

        return Vocab(tag_to_idx)

    def _build_char_vocab(self) -> None:
        characters = set()

        for char in "".join(self.data[UTTERANCE_TEXT]):
            if char != " ":
                characters.update(char)

        char_to_idx = create_index(characters, offset=2)
        char_to_idx["PAD"] = 0
        char_to_idx["UNK"] = 1

        return Vocab(char_to_idx)

    def _build_word_vocab(self):

        word_to_idx = {"PAD": 0, "UNK": 1}

        for txt_utterance in self.data[UTTERANCE_TEXT]:
            words = txt_utterance.split(" ")
            for word in words:
                if word not in word_to_idx:
                    word_to_idx[word] = len(word_to_idx)

        return Vocab(word_to_idx)

    def _compute_max_len(self):
        max_utt_len = 0
        max_word_len = 0

        for utterance in self.data[UTTERANCE_TEXT]:

            tokens = utterance.strip().split(" ")

            if len(tokens) > max_utt_len:
                max_utt_len = len(tokens)

            for token in tokens:
                chars = list(token)
                if len(chars) > max_word_len:
                    max_word_len = len(chars)

        self.max_word_len = max_word_len
        self.max_utt_len = max_utt_len

        logger.info(f"Max utterance length: {max_utt_len}")
        logger.info(f"Max word length: {self.max_word_len}")

    def _prepare_datasets(self):
        # Build datasets
        for stage, data in self.split_data.items():
            logger.info(f"Creating dataset samples for stage: {stage}...")
            self.datasets[stage] = NERDataset(data, unique_mode=self.unique_mode)

        # Encode datasets
        kwargs = {
            "encoder": self.encoder,
            "word_vocab": self.vocabularies["word"],
            "char_vocab": self.vocabularies["char"],
            "tag_vocab": self.vocabularies["tag"],
            "max_utt_len": self.max_utt_len,
            "max_word_len": self.max_word_len,
        }

        for stage, dataset in self.datasets.items():
            logger.info(f"Preprocessing dataset for stage: {stage}...")
            dataset.preprocess(**kwargs)

    def _setup_deduplicators(self):
        stages = [Stage.train, Stage.val]

        # Get deduplicators for train and val
        for stage in stages:
            logger.info(
                f"Setting up deduplicator: {self.deduplicator_name} for stage: {stage}..."
            )
            dataset = self.datasets[stage]
            shuffle = self.shuffle_train and stage == Stage.train

            logger.info(f"Random shuffling: {shuffle}")

            deduplicator = get_deduplicator(
                self.deduplicator_name,
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
            )

            if (
                self.deduplicator_name
                in {
                    "batchwise_unique",
                    "batchwise_weighted_unique",
                }
                and stage == Stage.train
            ):
                logger.info(
                    f"Expected virtual batch size: {self.expected_virtual_batch_size}"
                )

            self.deduplicators[stage] = deduplicator

    def train_dataloader(self):
        deduplicator = self.deduplicators[Stage.train]
        dataloader = deduplicator.deduplicate()

        return dataloader

    def val_dataloader(self):
        deduplicator = self.deduplicators[Stage.val]
        dataloader = deduplicator.deduplicate()

        return dataloader

    def test_dataloader(self):
        return DataLoader(
            self.datasets[Stage.test],
            batch_size=self.batch_size,
            num_workers=get_num_workers(),
            # persistent_workers=True,
            shuffle=False,
        )

    def state_dict(self) -> dict:

        state_dict = {
            "dataset_path": self.dataset_path,
            "deduplicator_name": self.deduplicator_name,
            "batch_size": self.batch_size,
            "random_state": self.random_state,
            "shuffle_train": self.shuffle_train,
            "max_utt_len": self.max_utt_len,
        }

        if self.preprocessed:
            state_dict.update(
                {
                    "preprocessed": self.preprocessed,
                    "vocabularies": self.vocabularies,
                    "max_word_len": self.max_word_len,
                    "encoder": self.encoder,
                    "datasets": self.datasets,
                }
            )

        return state_dict

    def load_state_dict(self, state_dict: dict):
        for attr, val in state_dict.items():
            setattr(self, attr, val)
