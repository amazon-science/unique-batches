from torch.utils.data import Dataset

from unique_batches.data.datasets.dataset import NERDataset


class DatasetWrapper(Dataset):
    def __init__(self, dataset: NERDataset):

        self.dataset = dataset

    def __getattr__(self, name: str):
        attr = getattr(self.dataset, name)
        return attr

    def __len__(self) -> int:
        return len(self.dataset)
