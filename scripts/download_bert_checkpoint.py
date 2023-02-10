# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0

import argparse
from pathlib import Path

from transformers import AutoModel, DistilBertTokenizerFast


def parse_args():
    parser = argparse.ArgumentParser(description="Download BERT checkpoint.")

    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        help="name of the checkpoint, e.g. distilbert-base-multilingual-cased",
        required=True,
    )

    parser.add_argument(
        "--destination",
        "-d",
        type=str,
        help="path to the folder where to save the checkpoint, defaults to ./checkpoints",
        required=False,
        default="./checkpoints",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    checkpoint = args.checkpoint
    destination = Path(args.destination)

    assert destination.is_dir() or not destination.exists()
    destination.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(checkpoint)
    model.save_pretrained(destination / checkpoint / "model")

    tokenizer = DistilBertTokenizerFast.from_pretrained(checkpoint)
    tokenizer.save_pretrained(destination / checkpoint / "tokenizer")


if __name__ == "__main__":
    main()
