# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import argparse
import re
from pathlib import Path

import pandas as pd

ANNOTATION = "annotation"
TAGS = "tags"
UTTERANCE_TEXT = "utterance_text"
DOMAIN = "domain"
INTENT = "intent"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute redundancy of a dataset in TSV format."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="path of the dataset, e.g. ./train_dataset.tsv",
        required=True,
    )

    args = parser.parse_args()

    return args


def get_NLU_tsv_columns(file):
    with open(file) as f:
        line = f.readline()

    num_columns = len(line.split("\t"))

    if num_columns == 1:
        return [ANNOTATION]
    elif num_columns == 3:
        return (DOMAIN, INTENT, ANNOTATION)
    elif num_columns == 5:
        return (DOMAIN, INTENT, ANNOTATION, "ecid", "uid")
    else:
        raise ValueError("Bad .tsv format")


def extract_utt_text(utt: str) -> str:
    """

    :param
    :return:
    """
    utt_tokens = re.findall(r"([^ ]+)\|[^ ]+", utt)
    utt_tokens = " ".join(utt_tokens)
    return utt_tokens


def extract_ner_labels(utt: str) -> list:
    """

    :param utt:
    :return:
    """
    labels = list(re.findall(r"[^ ]+\|([^ ]+)", utt))
    return labels


def add_dot(utt):
    return utt + " .|O"


def read_token_level_annotated_data(data_path: str, inference=False) -> pd.DataFrame:
    columns = get_NLU_tsv_columns(data_path)
    data = pd.read_table(data_path, names=columns)
    data["annotation"] = data["annotation"].apply(lambda utt: add_dot(utt)).values
    data.dropna(inplace=True)

    if not inference:
        data[TAGS] = data[ANNOTATION].apply(lambda utt: extract_ner_labels(utt)).values
        data[UTTERANCE_TEXT] = (
            data[ANNOTATION].apply(lambda utt: extract_utt_text(utt)).values
        )

    return data


def main():
    args = parse_args()
    path = Path(args.input)

    print(f"Loading data at {path}...")
    df = read_token_level_annotated_data(path)

    print("Computing redundancy...")
    utterances = df[UTTERANCE_TEXT]

    num_utterances = len(utterances)
    num_unique_utterances = len(utterances.value_counts())

    redundancy = 100.0 * (1 - num_unique_utterances / num_utterances)

    print(f"Num utterances: {num_utterances}")
    print(f"Num unique utterances: {num_unique_utterances}")
    print(f"Redundancy: {redundancy}")


if __name__ == "__main__":
    main()
