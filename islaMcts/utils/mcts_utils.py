import pickle
from copy import deepcopy
from pickle import PicklingError

from collections.abc import Mapping


class LazyDict(Mapping):
    def __init__(self, *args, **kw):
        self._raw_dict = dict(*args, **kw)

    def __getitem__(self, key):
        func, arg = self._raw_dict.__getitem__(key)
        return func(arg)

    def __iter__(self):
        return iter(self._raw_dict)

    def __len__(self):
        return len(self._raw_dict)


def my_deepcopy(obj):
    try:
        return pickle.loads(pickle.dumps(obj))
    except PicklingError:
        return deepcopy(obj)
