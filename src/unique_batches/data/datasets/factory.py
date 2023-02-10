from torch.utils.data import Dataset

__DATASET_REGISTRY__ = dict()


def get_dataset(name, **kwargs) -> Dataset:
    if name in __DATASET_REGISTRY__:
        cls = __DATASET_REGISTRY__[name]
        return cls(**kwargs)
    else:
        raise ValueError("Name %s not registered!" % name)


def register(name):
    def register_fn(cls):
        if name in __DATASET_REGISTRY__:
            raise ValueError("Name %s already registered!" % name)
        if not issubclass(cls, Dataset):
            raise ValueError("Class %s is not a subclass of %s" % (cls, Dataset))
        __DATASET_REGISTRY__[name] = cls
        return cls

    return register_fn
