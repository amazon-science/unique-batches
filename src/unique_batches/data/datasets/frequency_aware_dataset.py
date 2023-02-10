from typing import List

from torch.utils.data import Dataset

from unique_batches.data.datasets.dataset import NERDataset
from unique_batches.data.datasets.factory import register
from unique_batches.data.datasets.wrapper import DatasetWrapper


@register("frequency_aware_dataset")
class FrequencyAwareDataset(DatasetWrapper):
    def __init__(self, dataset: NERDataset, frequencies: List[float]):

        super().__init__(dataset)

        self.frequencies = frequencies

    def __getitem__(self, idx):
        return {**self.dataset.__getitem__(idx), "frequencies": self.frequencies[idx]}
