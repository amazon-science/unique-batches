# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import math
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from unique_batches.data.dataloader import build_dataloader
from unique_batches.data.datasets.dataset import NERDataset
from unique_batches.data.deduplicators.deduplicator import Deduplicator
from unique_batches.data.deduplicators.factory import register
from unique_batches.utils.framework import get_num_workers


@register("datasetwise_logarithmic")
class DatasetwiseLogarithmicDeduplicator(Deduplicator):
    """
    Datasetwise deduplicator which only keeps log(N) repetitions of each sample,
    where N is its number of occurrences.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        super().__init__(dataset, batch_size)

        self.shuffle = shuffle

    def deduplicate(self) -> DataLoader:
        self.dataset = self._remove_log_duplicates(self.dataset)

        dataloader = build_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=get_num_workers(),
        )

        return dataloader

    def _remove_log_duplicates(self, dataset: NERDataset) -> NERDataset:
        # Get samples
        samples = dataset.get_encoded_samples()
        sample_indices_to_keep = list()

        # Compute occurrences
        sample_counter = Counter()
        for sample in samples:
            sample_counter[sample] += 1

        # Find indices to keep
        new_occurrences = Counter()
        sample_indices_to_keep = list()

        for idx, sample in enumerate(samples):
            max_num_occ_utt = (
                math.log2(sample_counter[sample]) if sample_counter[sample] > 1 else 1
            )

            # if we still have less than log(N) repetitions of the sample, we keep this instance
            if new_occurrences[sample] < max_num_occ_utt:
                new_occurrences[sample] += 1

                sample_indices_to_keep.append(idx)

        # Set non-tensor fields
        new_samples = []
        for idx in sample_indices_to_keep:
            new_samples.append(dataset.samples[idx])

        new_dataset = NERDataset(new_samples)

        # Set tensor fields
        for attr_name, attr in dataset.__dict__.items():
            if isinstance(attr, torch.Tensor):
                updated_attr = [attr[idx].detach() for idx in sample_indices_to_keep]
                updated_attr = torch.stack(updated_attr)

                setattr(new_dataset, attr_name, updated_attr)

        return new_dataset
