# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import os
import re
from copy import copy
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import DistilBertTokenizerFast

from unique_batches.data.encoders.encoder import Encoder
from unique_batches.data.encoders.factory import register
from unique_batches.utils.framework import replace_null_values_with_constant
from unique_batches.utils.logging import get_logger

logger = get_logger(__name__)


@register("wordpiece_encoder")
class WordPieceEncoder(Encoder):
    """
    Preprocesses data in a Bert-compatible form, then uses
    a pretrained BertTokenizer to tokenize and encode the utterances
    """

    def __init__(self, checkpoint: str, *args, **kwargs):
        """
        Params:
            tokenizer_path: path to the pretrained Bert tokenizer
        """
        super().__init__(*args, **kwargs)

        self.tokenizer = self.setup_tokenizer(checkpoint)

    def setup_tokenizer(self, checkpoint: str):
        checkpoint = Path(checkpoint).resolve()
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            checkpoint, local_files_only=True
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        return tokenizer

    def preprocess(self, utterances: List[str], tags: List[str] = None):
        preproc_utterances, preproc_token_lang_labels = self._align_tokenization(
            list(utterances), tags
        )

        if tags is None:
            return preproc_utterances
        else:
            return preproc_utterances, preproc_token_lang_labels

    def encode(
        self,
        utterances: List[str] = None,
        **kwargs,
    ):
        """
        Params:
            utterances: tensor ~ (num_utterances) containing the text utterances
            token_lang_labels: list ~ (num_utterances, utt_len) containing
                               for each utterance the language tag for each word
            label_vocab: map language label to id e.g. 'it' -> 0, 'en' -> 1 and viceversa
        Returns:
            encoded_utts_tensor: tensor ~ (num_utterances, max_bert_len, emb_dim) where max_bert_len is the length
                                 of the longest utterance after the bert tokenization,
                                  i.e. with SEP, CLS and words split in wordpieces
            encoded_labels_tensor: tensor ~ (num_utterances, max_bert_len, emb_dim)
            word_ids: list of lists, each list contains the mapping from tokens to original words in the utterance,
                            e.g. the utterance 'presumo tu lo sappia' would result
                            in [ 'pres##', '##umo', 'tu', 'lo', 'sap##', '##ia']
                            and therefore in the following subword mask [0, 0, 1, 2, 3, 3]
        """

        encoded_utt_tensor, word_ids = self.encode_utterances(utterances)

        # to allow batching the subword masks we need to replace the Nones which are corresponding to the pad tokens
        # to a negative number so that it cannot be a index,
        # as each token is mapped to the position of the word it originated from
        word_ids = replace_null_values_with_constant(word_ids, -100)

        subword_mask_tensor = torch.tensor(word_ids).long()

        assert subword_mask_tensor.shape[0] == encoded_utt_tensor.shape[0]

        return (encoded_utt_tensor, subword_mask_tensor)

    def encode_utterances(
        self, utterances: List[str], **kwargs
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Params:
            utterances: tensor ~ (num_utterances) containing the text utterances
        Returns:
            encoded_utts_tensor: tensor ~ (num_utterances, max_bert_len, emb_dim) where max_bert_len is the length
                                 of the longest utterance after the bert tokenization,
                                  i.e. with SEP, CLS and words split in wordpieces
            word_ids: list of lists, each list contains the mapping from tokens to original words in the utterance,
                            e.g. the utterance 'presumo tu lo sappia'
                            would result in [ 'pres##', '##umo', 'tu', 'lo', 'sap##', '##ia']
                            and therefore in the following subword mask [0, 0, 1, 2, 3, 3]
        """

        encoded_utterances = self.tokenizer(
            utterances,
            padding=True,
            add_special_tokens=False,
            # truncation=True,
            # padding="max_length",
            # max_length=max_utt_len,
        )

        encoded_utts_tensor = encoded_utterances.convert_to_tensors("pt")["input_ids"]
        word_ids = [
            [word_id for word_id in encoded_utterances.word_ids(i)]
            for i in range(len(utterances))
        ]

        return encoded_utts_tensor, word_ids

    def _align_tokenization(self, utterances: list, token_lang_labels: list):
        """
        Handles the difference in tokenization between the input data and the one expected by Bert;
        dotted words e.g. 'turn off the t. v.' are tokenized like ['t.', 'v.'] in the input data
        and ['t','.','v','.'] in Bert, so we remove dots and merge them, e.g. 'turn off the tv'
        Params:
            utterances: list ~ (num_utterances) containing the text utterances
            token_lang_labels: list ~ (num_utterances, utt_len) containing for each utterance
             the language tag for each word
        Returns:
            utterances: list ~ (num_utterances) containing the fixed text utterances
            token_lang_labels: list ~ (num_utterances, utt_len) containing for each utterance
             the language tag for each word, aligned to the fixed utterances
        """

        if token_lang_labels is None:
            token_lang_labels = [[None for word in utt] for utt in utterances]

        logger.info("Removing non printable characters...")
        utterances = self._remove_non_printable_chars(utterances)

        logger.info("Preprocessing hyphen words...")
        utterances = self._preprocess_hyphen_words(utterances)

        logger.info("Preprocessing dotted words...")
        utterances, token_lang_labels = self._preprocess_dotted_words(
            utterances, token_lang_labels
        )

        logger.info("Preprocessing dashed words...")
        utterances, token_lang_labels = self._preprocess_dashed_words(
            utterances, token_lang_labels
        )
        return utterances, token_lang_labels

    def _preprocess_dotted_words(self, utterances: list, token_lang_labels: list):
        """
        Handles the difference in tokenization between the input data and the one expected by Bert;
        dotted words e.g. 'turn off the t. v.' are tokenized like ['t.', 'v.'] in the input data
        and ['t','.','v','.'] in Bert, so we remove dots and merge them, e.g. 'turn off the tv'

        Params:
            utterances: list ~ (num_utterances) containing the text utterances
            token_lang_labels: list ~ (num_utterances, utt_len) containing for each utterance
             the language tag for each word
        Returns:
            new_utterances: list ~ (num_utterances) containing the text utterances with the fixed dotted words
            new_token_lang_labels: list ~ (num_utterances) containing for each utterance
             the language tag for each word, aligned to the fixed utterances
        """
        new_token_lang_labels = []
        new_utterances = []

        for utt, utt_labels in zip(utterances, token_lang_labels):

            # remove all the internal (point, whitespace) pairs of characters inside dotted words
            # leaving only the last dot, e.g.    g. p. s. --> gps.
            utt = re.sub("([A-Za-z])\\. (?=([A-Za-z]\\.))", "\\g<1>", utt)

            words = utt.split(" ")
            new_words = copy(words)

            new_utt_labels = []
            label_index = 0

            for i, word in enumerate(words):

                try:
                    # assign the word the label at the right label index
                    new_utt_labels.append(utt_labels[label_index])
                except IndexError:
                    # print(f"Tokenization issue within the utterance: {utt}")
                    continue

                # if we find a dotted word
                if word.endswith(".") and word.count(".") == 1:

                    # remove the dot at the end
                    word = word[:-1]
                    # update the label index so to point to the next word
                    label_index += len(word)
                    new_words[i] = word

                # some words were added from enums, and these are not tokenized as the others,
                # so while most dotted words are spaced, eg "e. q.", these are not spaced, eg "e.q."
                elif word.endswith("."):

                    word = word.replace(".", "")
                    label_index += 1
                    new_words[i] = word

                # otherwise we just go to the next label
                else:
                    label_index += 1

            new_utterance = " ".join(new_words)

            new_token_lang_labels.append(new_utt_labels)
            new_utterances.append(new_utterance)

        return new_utterances, new_token_lang_labels

    def _preprocess_dashed_words(self, utterances: list, token_lang_labels: list):
        """
        dashed words e.g. 'schedule a follow-up' are tokenized like ['follow-up'] in input data and
        as ['follow', '-', 'up'] in Bert, so we remove the dash and split them, e.g. ['follow', 'up']

        Params:
            utterances: list ~ (num_utterances) containing the text utterances
            token_lang_labels: list ~ (num_utterances, utt_len) containing for each utterance
             the language tag for each word
        Returns:
            new_utterances: list ~ (num_utterances) containing the text utterances with the fixed dashed words
            new_token_lang_labels: list ~ (num_utterances) containing for each utterance
             the language tag for each word, aligned to the fixed utterances
        """

        new_utterances = []
        new_token_lang_labels = []

        for utt, utt_labels in zip(utterances, token_lang_labels):

            new_words = []

            new_utt_labels = []

            words = utt.split(" ")

            for word, word_label in zip(words, utt_labels):
                if "-" in word:
                    subwords = word.split("-")
                    for subword in subwords:
                        new_words.append(subword)
                        new_utt_labels.append(word_label)
                else:
                    new_words.append(word)
                    new_utt_labels.append(word_label)
            new_utterance = " ".join(new_words)
            new_utterances.append(new_utterance)
            new_token_lang_labels.append(new_utt_labels)

        return new_utterances, new_token_lang_labels

    def _preprocess_hyphen_words(self, utterances: list):
        """
        words with hyphens, e.g. "l' albero in giardino" are tokenized like ["l'", "albero"] in input data and
        as ["'", "l", "albero"] in Bert, so we just remove the hyphen, e.g. ["l", "albero"]

        Params:
            utterances: list ~ (num_utterances) containing the text utterances

        Returns:
            new_utterances: list ~ (num_utterances) containing the text utterances with removed hyphens
        """
        new_utterances = []

        for utt in utterances:
            utt = re.sub("'", "", utt)
            new_utterances.append(utt)

        return new_utterances

    def _remove_non_printable_chars(self, utterances):
        new_utterances = []
        for utt in utterances:
            utt = re.sub("\u200b", "", utt)
            if not utt or utt == " ":
                utt = None
            new_utterances.append(utt)
        return new_utterances
