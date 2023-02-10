# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from abc import abstractmethod

from torch.utils.data import DataLoader

from unique_batches.data.datasets.frequency_aware_dataset import NERDataset
from unique_batches.utils.logging import get_logger

logger = get_logger(__name__)


class Deduplicator:
    """
    Abstract class representing a deduplicator.
    """

    def __init__(self, dataset: NERDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    @abstractmethod
    def deduplicate(self) -> DataLoader:
        """
        Takes a Dataset and returns a DataLoader which iterates over a subsample
         of the original dataset ignoring some or all the duplicates.
        """
        raise NotImplementedError
