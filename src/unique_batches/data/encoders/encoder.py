# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from abc import ABC, abstractmethod
from typing import List

import torch

from unique_batches.data.vocab import Vocab
from unique_batches.utils.framework import pad_sequence

# from torch.nn.utils.rnn import pad_sequence


class Encoder(ABC):
    def preprocess(self, utterances, tags=None, **kwargs):
        return utterances, tags

    @abstractmethod
    def encode(self, utterances: List[str], **kwargs):
        pass

    @staticmethod
    def encode_tags(
        tags: List[List[str]], tag_vocab: Vocab, max_utt_len: int, **kwargs
    ) -> torch.Tensor:

        # no labels at inference time
        if tags is None:
            return tags

        encoded_tags = [
            torch.tensor([tag_vocab.item_to_idx[tag] for tag in utt_tags])
            for utt_tags in tags
        ]

        encoded_tags_tensor = pad_sequence(
            sequences=encoded_tags,
            max_len=max_utt_len,
            padding_value=tag_vocab.item_to_idx[tag_vocab.pad_symbol],
        ).long()

        return encoded_tags_tensor
