# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  

# SPDX-License-Identifier: CC-BY-NC-4.0


from unique_batches.data.deduplicators.deduplicator import Deduplicator

__DEDUPLICATOR_REGISTRY__ = dict()


def get_deduplicator(name, **kwargs) -> Deduplicator:
    if name in __DEDUPLICATOR_REGISTRY__:
        cls = __DEDUPLICATOR_REGISTRY__[name]
        return cls(**kwargs)
    else:
        raise ValueError("Name %s not registered!" % name)


def register(name):
    def register_fn(cls):
        if name in __DEDUPLICATOR_REGISTRY__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Deduplicator):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Deduplicator))
        __DEDUPLICATOR_REGISTRY__[name] = cls
        return cls

    return register_fn
