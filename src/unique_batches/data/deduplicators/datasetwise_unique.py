# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import torch
from torch.utils.data import DataLoader, Dataset

from unique_batches.data.dataloader import build_dataloader
from unique_batches.data.datasets.dataset import NERDataset
from unique_batches.data.deduplicators.deduplicator import Deduplicator
from unique_batches.data.deduplicators.factory import register
from unique_batches.utils.framework import get_num_workers


@register("datasetwise_unique")
class DatasetwiseUniqueDeduplicator(Deduplicator):
    """
    Datasetwise deduplicator which only keeps one instance per sample.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = True):
        super().__init__(dataset, batch_size)

        self.shuffle = shuffle

    def deduplicate(self) -> DataLoader:

        self.dataset = self.remove_duplicates(self.dataset)

        dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=get_num_workers(),
        )

        return dataloader

    def remove_duplicates(self, dataset: NERDataset) -> NERDataset:
        # Get samples filtering out duplicates (according to __hash__, __eq__)
        seen_samples = set()
        sample_indices_to_keep = list()

        for idx in range(len(dataset)):
            sample = dataset.get_encoded_sample(idx)

            if sample in seen_samples:
                continue

            seen_samples.add(sample)
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
