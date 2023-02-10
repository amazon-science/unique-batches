# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from typing import List

from torch.utils.data import Dataset, Sampler


class ScheduleSampler(Sampler):
    """
    Yields the indices of a dataset by following a given schedule
    """

    def __init__(self, data_source: Dataset, schedule: List[int]):
        """
        :param data_source: dataset
        :param schedule: list of dataset indices which defines the iteration order over the dataset
        """
        super().__init__(data_source)
        self.data_source = data_source
        self.schedule = schedule

    def __iter__(self):
        return iter(self.schedule)

    def __len__(self) -> int:
        return len(self.schedule)
