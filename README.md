# Unique Batches

This package contains the code to implement and test a new approach to model training. Its goal is to reduce the
training time while keeping the final accuracy on par. We do these by taking advantage (reducing) data redundancy. We
defined it Unique Batches because we deduplicate the data on a batch level and not on the entire dataset, keeping the
model learning trajectory very close to the full dataset one.

See the paper for more details: [Mitigating the Burden of Redundant Datasets via Batch-Wise Unique Samples and Frequency-Aware Losses](https://aclanthology.org/2023.acl-industry.23/).

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


## Citing the paper
If this code or paper was useful, please consider using the following citation:

```
@inproceedings{crisostomi-etal-2023-mitigating,
    title = "Mitigating the Burden of Redundant Datasets via Batch-Wise Unique Samples and Frequency-Aware Losses",
    author = "Crisostomi, Donato  and
      Caciolai, Andrea  and
      Pedrani, Alessandro  and
      Rottmann, Kay  and
      Manzotti, Alessandro  and
      Palumbo, Enrico  and
      Bernardi, Davide",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 5: Industry Track)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-industry.23",
    doi = "10.18653/v1/2023.acl-industry.23",
    pages = "235--247",
}
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License Summary

The documentation is made available under the Creative Commons Attribution-NonCommercial 4.0 International. See the LICENSE file.
