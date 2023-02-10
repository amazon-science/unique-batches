# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from torch.utils.data import DataLoader, Dataset

from unique_batches.data.dataloader import build_dataloader
from unique_batches.data.deduplicators.deduplicator import Deduplicator
from unique_batches.data.deduplicators.factory import register
from unique_batches.utils.framework import get_num_workers


@register("dummy")
class DummyDeduplicator(Deduplicator):
    """
    Dummy deduplicator which does not remove any duplicate.
    """

    def __init__(self, dataset: Dataset, batch_size: int, shuffle=True):
        super().__init__(dataset, batch_size)

        self.dataloader = build_dataloader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=get_num_workers(),
        )

    def deduplicate(self) -> DataLoader:
        """
        Does not remove any duplicate, so it only returns a standard DataLoader on the whole dataset.
        """
        return self.dataloader
