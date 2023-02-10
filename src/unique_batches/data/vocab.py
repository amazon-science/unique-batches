# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from __future__ import annotations

from abc import ABC, abstractstaticmethod
from typing import Collection, Dict, List, Set, TypeVar

T = TypeVar("T")


class Vocab(ABC):
    def __init__(
        self,
        item_to_idx: Dict[T, int],
        idx_to_item: Dict[int, T] = None,
        pad_symbol="PAD",
        unk_symbol="UNK",
    ):
        super().__init__()

        self.item_to_idx = item_to_idx
        self.idx_to_item = (
            idx_to_item if idx_to_item else {v: k for k, v in self.item_to_idx.items()}
        )
        self.pad_symbol = pad_symbol
        self.unk_symbol = unk_symbol

    def get_item_list(self) -> List[T]:
        """
        Returns the tags without the pad
        """
        items = set(self.item_to_idx.keys())
        items.remove(self.pad_symbol)
        return list(items)

    def get_index_list(self) -> List[int]:
        """
        Returns the indices except that for the pad index
        """
        indices = set(self.idx_to_item.keys())
        pad_index = self.item_to_idx[self.pad_symbol]
        indices.remove(pad_index)
        return list(indices)

    def __len__(self):
        return len(self.item_to_idx)

    def __getitem__(self, tag: str):
        return self.item_to_idx[tag]

    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError
