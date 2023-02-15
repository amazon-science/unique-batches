# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from torch.utils.data import DataLoader

from unique_batches.utils.framework import get_num_gpus
from unique_batches.utils.logging import get_logger

logger = get_logger(__name__)


def build_dataloader(*args, **kwargs) -> DataLoader:
    num_devices = max(1, get_num_gpus())

    # kwargs.update(dict(persistent_workers=True))

    if num_devices > 1:
        batch_size = kwargs["batch_size"]
        effective_batch_size = num_devices * batch_size

        logger.info(f"Using effective batch size: {effective_batch_size}")

        kwargs["batch_size"] = effective_batch_size

    dataloader = DataLoader(*args, **kwargs)

    return dataloader
