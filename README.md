# Unique Batches

This package contains the code to implement and test a new approach to model training. Its goal is to reduce the
training time while keeping the final accuracy on par. We do these by taking advantage (reducing) data redundancy. We
defined it Unique Batches because we deduplicate the data on a batch level and not on the entire dataset, keeping the
model learning trajectory very close to the full dataset one.

## Installation

A Linux environment is suggested. UniqueBatches dependencies are listed in the [requirements](requirements.txt) file.

You can easily install the package by creating a virtual environment (or conda environment), and then running
```
pip install .
```
in the root folder of the project. 

## Usage

Training and evaluation of models and deduplicators can be started by running
```
python3 scripts/main.py
```
from the root folder of the project. 

We use [Hydra](https://hydra.cc/docs/intro/) to maintain multiple configurations leading to reproducible experimental results.
These are kept in the `configs/` folder, and by default the one specified in the `defaults.yaml` config will be run, but everything can be overridden via the CLI.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License Summary

The documentation is made available under the Creative Commons Attribution-NonCommercial 4.0 International. See the LICENSE file.