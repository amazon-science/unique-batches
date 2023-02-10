# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


import json
import re
from pathlib import Path

import pandas as pd
import yaml

from unique_batches.constants import (ANNOTATION, DOMAIN, INTENT, TAGS,
                                      UTTERANCE_TEXT)


def load_json(path: Path) -> dict:
    content = path.read_text()
    json_dict = json.loads(content)

    return json_dict


def load_yaml(path: Path) -> dict:
    content = path.read_text()
    yaml_dict = yaml.safe_load(content)

    return yaml_dict


def load_config(config_path: Path):
    ext = config_path.suffix

    if ext == ".json":
        return load_json(config_path)
    elif ext in {".yml", ".yaml"}:
        return load_yaml(config_path)
    else:
        raise NotImplementedError


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


def get_new_path(path: Path) -> Path:
    new_path = path
    count = 0
    while new_path.exists():
        count += 1
        new_path = Path(f"{str(path)}{count}")

    return new_path


def write_json(results: dict, path: Path):
    path.write_text(json.dumps(results, indent=4))
