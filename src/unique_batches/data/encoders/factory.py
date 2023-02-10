# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from unique_batches.data.encoders.encoder import Encoder

__ENCODER_REGISTRY__ = dict()


def get_encoder(name, **kwargs) -> Encoder:
    if name in __ENCODER_REGISTRY__:
        cls = __ENCODER_REGISTRY__[name]
        return cls(**kwargs)
    else:
        raise ValueError("Name %s not registered!" % name)


def register(name):
    def register_fn(cls):
        if name in __ENCODER_REGISTRY__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Encoder):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Encoder))
        __ENCODER_REGISTRY__[name] = cls
        setattr(cls, "name", name)
        return cls

    return register_fn
