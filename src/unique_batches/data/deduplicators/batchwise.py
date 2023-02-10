# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import random
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from torch.utils.data import DataLoader, Dataset

from unique_batches.data.datasets.frequency_aware_dataset import NERDataset
from unique_batches.data.deduplicators.deduplicator import Deduplicator
from unique_batches.data.sample import EncodedSample
from unique_batches.utils.framework import flatten


class BatchwiseDeduplicator(Deduplicator):
    """
    Abstract class representing a deduplicator which removes duplicates at the batch level.
    """

    def __init__(self, dataset: NERDataset, batch_size: int, shuffle: bool, **kwargs):
        super().__init__(dataset, batch_size)

        self.shuffle = shuffle

    @abstractmethod
    def deduplicate(self) -> DataLoader:
        pass

    @staticmethod
    def generate_schedule(samples: List[EncodedSample], batch_size: int):
        """
        Creates a schedule of utterances so that each batch has no duplicate utterances
        :param samples:
        :param batch_size:
        :return:
        """
        assert len(samples) > 0

        (
            schedule_per_batch,
            occurrences_per_batch,
        ) = BatchwiseDeduplicator.compute_per_batch_schedule_occurrences(
            samples, batch_size
        )

        frequencies_per_batch = BatchwiseDeduplicator.occurrences_to_frequencies(
            occurrences_per_batch
        )

        frequencies_flattened = flatten(frequencies_per_batch)
        schedule_flattened = flatten(schedule_per_batch)

        scheduled_frequencies = BatchwiseDeduplicator.schedule_frequencies(
            schedule_flattened, frequencies_flattened, len(samples)
        )

        return np.array(schedule_flattened), scheduled_frequencies

    @staticmethod
    def compute_per_batch_schedule_occurrences(
        samples: List, batch_size: int
    ) -> Tuple[List[int], List[List[float]]]:

        occurrences_per_batch = list()
        schedule_per_batch = list()

        sample_seen_in_batch = set()
        utt_pos_in_batch = dict()

        batch_occurrences = list()
        batch_schedule = list()

        batch_capacity = batch_size

        for sample_ind, sample in enumerate(samples):
            if sample not in sample_seen_in_batch:

                batch_schedule.append(sample_ind)
                sample_seen_in_batch.add(sample)
                utt_pos_in_batch[sample] = len(batch_occurrences)

                batch_occurrences.append(1)
                batch_capacity -= 1

            else:
                batch_idx = utt_pos_in_batch[sample]
                batch_occurrences[batch_idx] += 1

            if batch_capacity == 0 or sample_ind == len(samples) - 1:

                occurrences_per_batch.append(batch_occurrences)
                schedule_per_batch.append(batch_schedule)

                batch_capacity = batch_size
                utt_pos_in_batch = dict()
                sample_seen_in_batch = set()
                batch_schedule = list()
                batch_occurrences = list()

        return schedule_per_batch, occurrences_per_batch

    @staticmethod
    def occurrences_to_frequencies(occurrences):
        frequencies = []

        for batch_occurrences in occurrences:

            batch_occurrences = np.array(batch_occurrences, dtype=float)

            virtual_batch_size = sum(batch_occurrences)
            batch_frequencies = batch_occurrences / virtual_batch_size

            frequencies.append(batch_frequencies)

        return frequencies

    @staticmethod
    def schedule_frequencies(schedule, frequencies, num_utterances):

        assert len(schedule) == len(frequencies)
        assert num_utterances >= len(schedule)

        if not schedule:
            return []

        # utterances which are not indexed by schedule will have 0 freq
        # these are never accessed anyway
        scheduled_freqs = [0.0] * num_utterances

        # initially, frequencies[i] contains the frequency of the i-th utt in the schedule
        # at the end, scheduled_freqs[idx] contains the frequency of the utt pointed by index idx
        # in schedule
        for ind, sched in enumerate(schedule):
            scheduled_freqs[sched] = frequencies[ind]

        return scheduled_freqs
