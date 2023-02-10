# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def _forward_deterministic(
        self, inputs: torch.FloatTensor, targets: torch.LongTensor
    ) -> torch.FloatTensor:

        device = inputs.device
        inputs = inputs.cpu()
        targets = targets.cpu()

        loss = self.criterion(inputs, targets)

        loss = loss.to(device)

        return loss

    def forward(
        self, inputs: torch.FloatTensor, targets: torch.LongTensor
    ) -> torch.FloatTensor:
        if torch.are_deterministic_algorithms_enabled():
            return self._forward_deterministic(inputs, targets)

        return self.criterion(inputs, targets)


class FrequencyAwareCrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=0)

    def _cross_entropy(
        self,
        inputs: torch.FloatTensor,
        targets: torch.LongTensor,
    ) -> torch.FloatTensor:

        if torch.are_deterministic_algorithms_enabled():
            device = inputs.device
            inputs = inputs.cpu()
            targets = targets.cpu()

        loss = self._cross_entropy_loss(inputs, targets)

        if torch.are_deterministic_algorithms_enabled():
            loss = loss.to(device)

        return loss

    def forward(
        self,
        inputs: torch.FloatTensor,
        targets: torch.LongTensor,
        frequencies: torch.LongTensor,
    ) -> torch.FloatTensor:

        # [batch_size, seq_len]
        loss = self._cross_entropy(inputs, targets)

        # [batch_size, ]
        loss = loss.mean(dim=1)
        loss = loss * frequencies

        # []
        loss = loss.sum(dim=0)

        return loss
