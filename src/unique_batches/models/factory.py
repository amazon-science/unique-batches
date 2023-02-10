# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from unique_batches.data.datamodule import NERDataModule
from unique_batches.models.model import Model

__MODEL_REGISTRY__ = dict()


def get_model(name, **kwargs) -> Model:
    if name in __MODEL_REGISTRY__:
        cls = __MODEL_REGISTRY__[name]
        return cls(**kwargs)
    else:
        raise ValueError("Name %s not registered!" % name)


def register(name):
    def register_fn(cls):
        if name in __MODEL_REGISTRY__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Model):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Model))
        __MODEL_REGISTRY__[name] = cls
        setattr(cls, "name", name)
        return cls

    return register_fn


def get_model_config(model_name: str, datamodule: NERDataModule, global_config: dict):

    # Global params
    lr = global_config["experiment"]["lr"]
    log_grads = global_config["experiment"]["log_grads"]
    log_params = global_config["experiment"]["log_params"]

    # Datamodule params
    deduplicator_name = datamodule.deduplicator_name
    lr_increase_factor = datamodule.virtual_over_batch_size_ratio
    max_utt_len = datamodule.max_utt_len

    params = {
        "model_name": model_name,
        "max_utt_len": max_utt_len,
        "tag_vocab": datamodule.vocabularies["tag"],
        "num_tags": len(datamodule.vocabularies["tag"]),
        "deduplicator_name": deduplicator_name,
        "lr": lr,
        "lr_increase_factor": lr_increase_factor,
        "log_grads": log_grads,
        "log_params": log_params,
    }

    if log_params:
        params_logs_per_epoch = global_config["experiment"]["params_logs_per_epoch"]
        params.update({"params_logs_per_epoch": params_logs_per_epoch})

    # Model params
    model_params = global_config["model"][model_name]
    params.update(model_params)

    if model_name == "lstm_based":
        from unique_batches.models.lstm_based import LSTMbasedConfig

        params.update(
            {
                "max_utt_len": max_utt_len,
                "vocab_size": len(datamodule.vocabularies["word"]),
            }
        )
        config = LSTMbasedConfig

    elif model_name == "cnn_based":
        from unique_batches.models.cnn_based import CNNBasedConfig

        params.update(
            {
                "vocab_size": len(datamodule.vocabularies["char"]),
            }
        )
        config = CNNBasedConfig

    elif model_name == "bert_based":
        from unique_batches.models.bert_based import BertBasedConfig

        params.update(
            {
                "max_utt_len": max_utt_len,
                "vocab_size": len(datamodule.vocabularies["word"]),
            }
        )
        config = BertBasedConfig

    else:
        raise NotImplementedError

    return config(**params)
