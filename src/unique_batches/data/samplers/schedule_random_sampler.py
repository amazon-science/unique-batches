from typing import List

import torch
from torch.utils.data import Dataset, Sampler


class ScheduleRandomSampler(Sampler[int]):
    r"""Samples elements randomly from a given schedule. If without replacement, then sample from a shuffled dataset.


    Args:
        data_source (Dataset): dataset to sample from
        schedule (List[int]): schedule with dataset indices
    """

    def __init__(self, data_source: Dataset, schedule: List[int]) -> None:

        super().__init__(data_source)

        self.data_source = data_source
        self.schedule = schedule
        # self.generator = torch.Generator()

    def __iter__(self):
        n = len(self.schedule)
        # yield from torch.randperm(n, generator=self.generator).tolist()
        yield from torch.randperm(n).tolist()

    def __len__(self) -> int:
        return len(self.schedule)
