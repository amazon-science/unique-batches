# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import random

from torch.utils.data import DataLoader

from unique_batches.data.dataloader import build_dataloader
from unique_batches.data.datasets.frequency_aware_dataset import (
    FrequencyAwareDataset, NERDataset)
from unique_batches.data.deduplicators.batchwise import BatchwiseDeduplicator
from unique_batches.data.deduplicators.factory import register
from unique_batches.data.samplers import ScheduleRandomSampler, ScheduleSampler
from unique_batches.utils.framework import get_num_workers


@register("batchwise_unique")
class BatchwiseUniqueDeduplicator(BatchwiseDeduplicator):
    """
    Batchwise deduplicator which does not account for the frequencies of the samples.
    """

    def __init__(self, dataset: NERDataset, batch_size: int, **kwargs):
        super().__init__(dataset, batch_size, **kwargs)

    def deduplicate(self) -> DataLoader:

        if self.shuffle:
            self.dataset.shuffle()

        samples = self.dataset.get_encoded_samples()
        schedule, _ = BatchwiseDeduplicator.generate_schedule(samples, self.batch_size)

        sampler = ScheduleSampler(self.dataset, schedule)

        dataloader = build_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            # shuffle=self.shuffle, # cannot use with sampler
            num_workers=get_num_workers(),
        )

        return dataloader
