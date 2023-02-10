# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import math

import numpy as np
import pandas as pd
from scipy.special import comb

from unique_batches.constants import UTTERANCE_TEXT


def find_expected_virtual_batch_size(dataset: pd.DataFrame, batch_size: int) -> int:
    """Computes the expected virtual batch size from removing duplicates batch-wise,
    with a given batch size, from a given dataset.

    We have a closed-form way of estimating the number of unique objects in a sample,
    GIVEN A SAMPLE SIZE. We don't know the sample size that makes a virtual batch have
    exactly `batch_size` duplicates, therefore we must find it.
    To do so, we exploit the fact that increasing the sample size can only increase (or leave as-is)
    the number of unique objects in the sample, therefore the sample sizes are ordered wrt the
    property we want, therefore we can resort to binary search to iterate over samples sizes.

    Args:
        dataset (pd.DataFrame): dataset
        batch_size (int): batch size

    Returns:
        int: expected virtual batch size
    """

    start, end = batch_size, 2**4 * batch_size

    while start <= end:

        sample_size = math.floor((end + start) / 2)
        num_unique_objs_sample = expected_num_unique_objects_in_sample(
            dataset, sample_size
        )

        if (
            math.ceil(num_unique_objs_sample) == batch_size
            or math.floor(num_unique_objs_sample) == batch_size
        ):
            # Found the sample size (virtual batch size) that makes a batch of size `batch_size`
            # have exactly `batch_size` unique objects
            return sample_size
        elif num_unique_objs_sample > batch_size:
            # Found more unique objects than can fit in the batch
            end = sample_size - 1
        else:
            # Found fewer unique objects than can fit in the batch
            start = sample_size + 1

    raise RuntimeError("Could not determine expected virtual batch size")


def expected_num_unique_objects_in_sample(
    dataset: pd.DataFrame, sample_size: int
) -> float:
    """Computes the expected number of unique objects in a sample of given size, given a certain dataset.
    We have a closed-form way of estimating the number of duplicates in a sample, therefore
        a) knowing the sample size
        b) knowing the number of duplicates
    the number of unique objects in the sample must be a - b

    Args:
        dataset (pd.DataFrame): dataset
        sample_size (int): sample size

    Returns:
        int: number of unique objects in sample
    """
    return sample_size - expected_num_duplicates_in_sample(dataset, sample_size)


def expected_num_duplicates_in_sample(dataset: pd.DataFrame, sample_size: int) -> float:
    """Computes the expected number of duplicates in a sample of given size given a certain dataset

    Args:
        dataset (pd.DataFrame): dataset
        sample_size (int): sample size

    Returns:
        int: Number of duplicates in sample
    """

    # size of dataset
    N = len(dataset)

    # num samples
    n = sample_size

    # occurrences per "object" (i.e. distinct utterance text)
    occurrences = num_occurrences_by_utterance(dataset)

    p0s = {}

    # expected num duplicates in sample
    d = 0

    for k_i in occurrences:
        # Considering object o_i
        if n >= (N - k_i):
            # Number of objects different from o_i is less than batch size
            # Probability of NOT sampling o_i is zero
            p0 = 0
        else:
            # Compute probability of NOT sampling o_i
            if k_i not in p0s:
                p0 = compute_p0(N, k_i, n)
                p0s[k_i] = p0

            p0 = p0s[k_i]

        # Expected num duplicates of object o_i in sample
        d_i = n * (k_i / N) - 1 + p0
        d = d + d_i

    return d


def num_occurrences_by_utterance(dataset: pd.DataFrame) -> np.ndarray:
    """Computes the number of occurrences per unique utterance in dataset

    Args:
        dataset (pd.DataFrame): dataset

    Returns:
        np.ndarray: Occurrences
    """
    grouped_by_utt = dataset.groupby(UTTERANCE_TEXT)

    grouped_by_utt_size = grouped_by_utt.size().to_frame("count").reset_index()
    grouped_by_utt_size = grouped_by_utt_size.sort_values("count", ascending=False)

    return np.array(list(grouped_by_utt_size["count"]))


def compute_p0(N: int, K: int, n: int) -> float:
    """Probability mass function of the counting variable following a multivariate hypergeometric, evaluated in 0,
    i.e. the probability of sampling 0 times object o_i in n draws (sample size = n), without replacement,
    from a finite population of size N that contains exactly K such objects

    Args:
        N (int): Population size
        K (int): Number of occurrences of object o_i in the population
        n (int): Sample size

    Returns:
        float: probability
    """
    return comb(N - K, n, exact=True) / comb(N, n, exact=True)
