# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import functools
import multiprocessing
import operator
from enum import Enum
from typing import Dict, List, Optional, Set, TypeVar

import torch

from unique_batches.configuration import GlobalConfiguration

T = TypeVar("T")


class Stage(Enum):
    train = "training"
    val = "validation"
    test = "test"


def get_num_gpus(config: dict = None) -> int:
    """Returns the number of gpus being used

    Args:
        config (dict): configuration dict

    Returns:
        int: number of gpus
    """
    if not torch.cuda.is_available():
        return 0

    if not config:
        config = GlobalConfiguration.get().hydra_config

    desired_num_gpus = config["gpus"]
    actual_num_gpus = torch.cuda.device_count()

    if desired_num_gpus == -1:
        return actual_num_gpus

    return min(actual_num_gpus, desired_num_gpus)


def get_num_workers(config: dict = None) -> int:

    if not config:
        config = GlobalConfiguration.get().hydra_config

    desired_num_workers = config["workers"]
    actual_num_workers = multiprocessing.cpu_count() - 1

    if desired_num_workers == -1:
        return actual_num_workers

    return min(actual_num_workers, desired_num_workers)


def pad_sequence(
    sequences: List[torch.Tensor], max_len: int, padding_value: float = 0.0
) -> torch.Tensor:
    """
    adds padding_value to each sequence until they are all of length max_len
    """

    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    out_dims = (len(sequences), max_len) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor

        # length = min(tensor.size(0), max_len)
        # out_tensor[i, :length, ...] = tensor[:length, ...]

    return out_tensor


def create_index(symbols: Set[str], offset=0) -> Dict[str, int]:
    index = {symbol: i + offset for i, symbol in enumerate(symbols)}
    return index


def replace_null_values_with_constant(
    list_of_lists: List[List[T]], constant: T
) -> List[T]:
    return [
        [elem if elem is not None else constant for elem in inner_list]
        for inner_list in list_of_lists
    ]


def flatten(list_of_lists: List[List[T]]) -> List[T]:
    """
    flattens a list of lists into a list
        [ [1, 2, 3], [4, 5, 6] ] --> [1, 2, 3, 4, 5, 6]
    :param list_of_lists:
    :return: list_flattened
    """
    list_flattened = functools.reduce(operator.iconcat, list_of_lists, [])
    return list_flattened


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out
