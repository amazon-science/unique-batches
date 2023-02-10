# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from __future__ import annotations

from typing import List

import pandas as pd
import torch

from unique_batches.constants import DOMAIN, INTENT, TAGS, UTTERANCE_TEXT


class Sample:
    def __init__(
        self,
        text: str,
        tags: List[str] = None,
        domain: str = None,
        intent: str = None,
        unique_mode: str = "text",
    ) -> None:
        self.text = text
        self.tags = tags
        self.domain = domain
        self.intent = intent
        self.unique_mode = unique_mode

        self._len = self.text.split()

    @staticmethod
    def create_samples(data: pd.DataFrame, unique_mode: str = "text") -> List[Sample]:
        utterance_text_list = data[UTTERANCE_TEXT].tolist()
        tags_list = data[TAGS].tolist()
        domain_list = data[DOMAIN].tolist()
        intent_list = data[INTENT].tolist()

        sample_list = []
        for idx in range(len(data)):
            text = utterance_text_list[idx]
            tags = tags_list[idx]
            domain = domain_list[idx]
            intent = intent_list[idx]

            sample = Sample(
                text=text,
                tags=tags,
                domain=domain,
                intent=intent,
                unique_mode=unique_mode,
            )

            sample_list.append(sample)

        return sample_list

    def __len__(self):
        return self._len

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, Sample):
            return False

        if self.unique_mode == "text":
            this = self.text
            other = obj.text
        elif self.unique_mode == "annotation":
            this = (self.text, self.tags, self.domain, self.intent)
            other = (obj.text, obj.tags, obj.domain, obj.intent)
        else:
            raise NotImplementedError

        return this == other

    def __ne__(self, obj: object) -> bool:
        return not self.__eq__(obj)

    def __hash__(self) -> int:
        if self.unique_mode == "text":
            utt = self.text
        elif self.unique_mode == "annotation":
            utt = (self.text, self.tags, self.domain, self.intent)
        else:
            raise NotImplementedError

        return hash(utt)


class EncodedSample:
    def __init__(
        self,
        utterance: torch.LongTensor,
        utterance_text: str = None,
        tags: torch.LongTensor = None,
        domain: str = None,
        intent: str = None,
        unique_mode: str = "text",
    ) -> None:

        self.utterance = utterance

        self.utterance_text = utterance_text
        self.tags = tags
        self.domain = domain
        self.intent = intent

        self.unique_mode = unique_mode

        self._hash = None

    def _compute_hash(self):
        utt = tuple(self.utterance.tolist())

        if self.unique_mode == "text":
            hash_ = hash(utt)
        elif self.unique_mode == "annotation":
            tags = tuple(self.tags.tolist())
            hash_ = hash((utt, tags))
        else:
            raise NotImplementedError

        return hash_

    def __len__(self):
        return self.utterance.shape[0]

    def __eq__(self, obj: object) -> bool:
        if not isinstance(obj, EncodedSample):
            return False

        if self.unique_mode == "text":
            return torch.equal(self.utterance, obj.utterance)
        elif self.unique_mode == "annotation":
            return torch.equal(self.utterance, obj.utterance) and torch.equal(
                self.tags, obj.tags
            )
        else:
            raise NotImplementedError

    def __ne__(self, obj: object) -> bool:
        return not self.__eq__(obj)

    def __hash__(self) -> int:
        if self._hash is None:
            self._hash = self._compute_hash()

        return self._hash
