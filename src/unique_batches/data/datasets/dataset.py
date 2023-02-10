from typing import Collection, List, Union

import pandas as pd
import torch
from torch.utils.data import Dataset

from unique_batches.data.datasets.factory import register
from unique_batches.data.encoders.encoder import Encoder
from unique_batches.data.sample import EncodedSample, Sample


@register("base_dataset")
class NERDataset(Dataset):
    def __init__(
        self, data: Union[pd.DataFrame, Collection[Sample]], unique_mode: str = "text"
    ):
        super().__init__()

        if isinstance(data, pd.DataFrame):
            samples = Sample.create_samples(data, unique_mode=unique_mode)
        elif isinstance(data, list) or isinstance(data, set):
            samples = data
        else:
            raise NotImplementedError

        self.samples = samples

        self.utterances = [sample.text for sample in samples]
        self.tags = [sample.tags for sample in samples]

        # To be filled during preprocessing
        self.encoded_utterances: torch.Tensor = None
        self.word_ids: torch.Tensor = None
        self.encoded_tags: torch.Tensor = None

        # To be filled at runtime
        self._encoded_samples = None
        # self._generator = None

    def shuffle(self):
        # if self._generator is None:
        #     self._generator = torch.Generator()

        n = len(self)
        # shuffled_indices = torch.randperm(n, generator=self._generator).tolist()
        shuffled_indices = torch.randperm(n).tolist()

        self.utterances = [self.utterances[idx] for idx in shuffled_indices]
        self.tags = [self.tags[idx] for idx in shuffled_indices]

        if self.encoded_utterances is not None:
            self.encoded_utterances = self.encoded_utterances[shuffled_indices, ...]

        if self.word_ids is not None:
            self.word_ids = self.word_ids[shuffled_indices, ...]

        if self.encoded_tags is not None:
            self.encoded_tags = self.encoded_tags[shuffled_indices, ...]

        if self._encoded_samples is not None:
            self._encoded_samples = [
                self._encoded_samples[idx] for idx in shuffled_indices
            ]

    def preprocess(self, encoder: Encoder, **encoder_kwargs):

        # Preprocess utts and tags
        (self.utterances, self.tags) = encoder.preprocess(
            utterances=self.utterances, tags=self.tags
        )

        # Tokenize and encode utts
        (encoded_token_tensor, word_ids_tensor,) = encoder.encode(
            utterances=self.utterances,
            **encoder_kwargs,
        )

        self.encoded_utterances = encoded_token_tensor
        self.word_ids = word_ids_tensor

        # Encode tags
        encoder_kwargs.update(dict(word_ids=self.word_ids))
        self.encoded_tags = encoder.encode_tags(tags=self.tags, **encoder_kwargs)

    def get_encoded_sample(self, idx: int) -> EncodedSample:
        utterance = self.utterances[idx]
        encoded_utterance = self.encoded_utterances[idx]
        encoded_tags = self.encoded_tags[idx] if self.encoded_tags is not None else None
        domain = self.samples[idx].domain
        intent = self.samples[idx].intent

        sample = EncodedSample(
            utterance_text=utterance,
            utterance=encoded_utterance,
            tags=encoded_tags,
            domain=domain,
            intent=intent,
        )

        return sample

    def get_encoded_samples(self) -> List[EncodedSample]:

        if self._encoded_samples is None:
            self._encoded_samples = [
                self.get_encoded_sample(idx) for idx in range(len(self))
            ]

        return self._encoded_samples

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        item = dict()

        # item = {
        #     "txt_utterances": self.txt_utterances[idx],
        # }

        # if self.tags:
        #     item.update({"tags": self.tags[idx]})

        if self.encoded_utterances is not None:
            item.update({"utterances": self.encoded_utterances[idx]})

        if self.word_ids is not None:
            item.update({"word_ids": self.word_ids[idx]})

        if self.encoded_tags is not None:
            item.update({"tags": self.encoded_tags[idx]})

        return item
