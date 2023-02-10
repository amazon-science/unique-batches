# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from typing import List

import torch

from unique_batches.data.encoders.encoder import Encoder
from unique_batches.data.encoders.factory import register
from unique_batches.data.vocab import Vocab


@register("char_encoder")
class CharEncoder(Encoder):
    """
    Tokenize data to char list, e.g. ['metti musica in cucina', ..] --> [ ['m', 'e', 't', 't', 'i'], ['m', u', ...] .. ]
    and then encode chars according to the character vocabularity,
    e.g. [ ['m', 'e', 't', 't', 'i'], ['m', u', ...] .. ] --> [ [3, 4, 7, 7, 2], [3, 9, ...] .. ]
    """

    def encode(
        self,
        utterances: List[str],
        char_vocab: Vocab,
        max_utt_len: int = -1,
        max_word_len: int = -1,
        **kwargs
    ):
        """
        Params:
            utterances: tensor ~ (num_utterances) containing the text utterances
            token_lang_labels: list ~ (num_utterances, utt_len) containing
                               for each utterance the language tag for each word
            label_vocab: map language label to id e.g. 'it' -> 0, 'en' -> 1
            max_len: max sequence len found in all the utterances
        Returns:
        """

        tokenized_utterances = self.tokenize(utterances)

        encoded_utt_tensor = self.encode_utterances(
            tokenized_utterances=tokenized_utterances,
            char_vocab=char_vocab,
            max_utt_len=max_utt_len,
            max_word_len=max_word_len,
        )

        if max_utt_len == -1:
            utt_lens = [len(encoded_utt) for encoded_utt in encoded_utt_tensor]
            max_utt_len = max(utt_lens)

        if max_word_len == -1:
            word_lens = [
                len(encoded_word)
                for encoded_utt in encoded_utt_tensor
                for encoded_word in encoded_utt
            ]
            max_word_len = max(word_lens)

        # Dummy subword mask (1-to-1 correspondence between words and tokens)
        subword_mask_tensor = torch.ones_like(encoded_utt_tensor)

        return (encoded_utt_tensor, subword_mask_tensor)

    def tokenize(self, utterances: List[str]) -> List[List[str]]:
        """
        Turn utterances into array of characters
        Params:
            utterances: list of strings, e.g. ['play music in cocina', 'metti musica strong', .. ]
        Returns:
            data_char_form: [[['p', 'l', 'a', 'y'],
                              ['m', 'u', 's', 'i', 'c'],
                              ['i', 'n'],
                              ['c', 'o', 'c', 'i', 'n', 'a']],
                              [['m', 'e', 't', 't', 'i'],
                              ['m', 'u', 's', 'i', 'c', 'a'],
                              ['s', 't', 'r', 'o', 'n', 'g']], ...]
        """
        data_char_form = []
        characters = []
        for utterance in utterances:
            tokens = utterance.strip().split(" ")
            utt_char_form = []
            for whole_thing in tokens:
                token = whole_thing.strip().split("|")[0]
                char = list(token)
                characters.extend(char)
                utt_char_form.append(char)
            data_char_form.append(utt_char_form)

        return data_char_form

    def encode_utterances(
        self,
        tokenized_utterances: List[List[str]],
        char_vocab: Vocab,
        max_utt_len: int,
        max_word_len: int,
    ) -> torch.LongTensor:
        """
        Create array of ints from characters, adding padding.
        Params:
            tokenized_utterances: [[['p', 'l', 'a', 'y'],
                                  ['m', 'u', 's', 'i', 'c'],
                                  ['i', 'n'],
                                  ['c', 'o', 'c', 'i', 'n', 'a']],
                                  [['m', 'e', 't', 't', 'i'],
                                  ['m', 'u', 's', 'i', 'c', 'a'],
                                  ['s', 't', 'r', 'o', 'n', 'g']]]
            max_utt_len: maximum length of any utterance found in the data
            max_word_len: maximum length of any word found in the data
            char_vocab: mapping from characters to ids and vice versa

        Returns:
            x_char_index:  [[[11,  4,  6, 14,  0,  0],
                            [26, 18, 23, 13, 10,  0],
                            [13,  5,  0,  0,  0,  0],
                            [10,  9, 10, 13,  5,  6]],
                           [[26, 16, 24, 24, 13,  0],
                            [26, 18, 23, 13, 10,  6],
                            [23, 24,  3,  9,  5, 15],
                            [ 0,  0,  0,  0,  0,  0]]])
        """

        x_char_index = torch.zeros(
            (len(tokenized_utterances), max_utt_len, max_word_len)
        ).long()
        for i, sentence in enumerate(tokenized_utterances):
            for j in range(max_utt_len):
                for k in range(max_word_len):
                    try:
                        unk_index = char_vocab["UNK"]
                        char_index = char_vocab.item_to_idx.get(
                            sentence[j][k], unk_index
                        )
                        x_char_index[i, j, k] = char_index
                    except IndexError:  # padding, already = 0
                        pass
        return x_char_index
