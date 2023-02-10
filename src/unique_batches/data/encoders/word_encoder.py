# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from typing import List

import torch

from unique_batches.data.encoders.encoder import Encoder
from unique_batches.data.encoders.factory import register
from unique_batches.data.vocab import Vocab
from unique_batches.utils.framework import pad_sequence


@register("word_encoder")
class WordEncoder(Encoder):
    def encode(
        self, utterances: List[str], word_vocab: Vocab, max_utt_len: int = -1, **kwargs
    ):

        encoded_utt_tensor = self.encode_utterances(
            utterances=utterances,
            word_vocab=word_vocab,
            max_utt_len=max_utt_len,
        )

        # Dummy subword mask (1-to-1 correspondence between words and tokens)
        subword_mask_tensor = torch.ones_like(encoded_utt_tensor)

        return (encoded_utt_tensor, subword_mask_tensor)

    def encode_utterances(
        self, utterances: List[str], word_vocab: Vocab, max_utt_len: int = -1
    ):

        encoded_utts = []

        for utt in utterances:

            words = utt.split(" ")
            encoded_utt = []

            for word in words:

                unk_idx = word_vocab.item_to_idx[word_vocab.unk_symbol]
                encoded_word = word_vocab.item_to_idx.get(word, unk_idx)
                encoded_utt.append(encoded_word)

            encoded_utt = torch.tensor(encoded_utt)
            encoded_utts.append(encoded_utt)

        if max_utt_len == -1:
            utt_lens = [len(encoded_utt) for encoded_utt in encoded_utts]
            max_utt_len = max(utt_lens)

        encoded_utts = pad_sequence(
            encoded_utts,
            max_len=max_utt_len,
            padding_value=word_vocab.item_to_idx[word_vocab.pad_symbol],
        )

        return encoded_utts
