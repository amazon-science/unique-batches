# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

from collections import Counter

import torch
from torch.utils.data import DataLoader

from unique_batches.data.dataloader import build_dataloader
from unique_batches.data.datasets.dataset import NERDataset
from unique_batches.data.datasets.frequency_aware_dataset import \
    FrequencyAwareDataset
from unique_batches.data.deduplicators.datasetwise_unique import \
    DatasetwiseUniqueDeduplicator
from unique_batches.data.deduplicators.factory import register
from unique_batches.utils.framework import get_num_workers


@register("datasetwise_weighted_unique")
class DatasetwiseWeightedUniqueDeduplicator(DatasetwiseUniqueDeduplicator):
    """
    Datasetwise deduplicator which accounts for the frequency of the samples.
    """

    def __init__(self, dataset: NERDataset, batch_size: int, shuffle: bool = True):
        super().__init__(dataset, batch_size)

        self.shuffle = shuffle

    def deduplicate(self) -> DataLoader:

        frequencies = self._compute_frequencies(self.dataset)
        self.dataset = self.remove_duplicates(self.dataset)
        self.dataset = FrequencyAwareDataset(self.dataset, frequencies)

        data_loader = build_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=get_num_workers(),
        )

        return data_loader

    def _compute_frequencies(self, dataset: NERDataset) -> torch.FloatTensor:
        samples = dataset.get_encoded_samples()

        # Compute occurrences
        sample_counter = Counter()
        for sample in samples:
            sample_counter[sample] += 1

        occurrences = [sample_counter[sample] for sample in samples]

        # Turn to frequencies
        occurrences = torch.tensor(occurrences).float()
        frequencies = occurrences / occurrences.sum()

        return frequencies
