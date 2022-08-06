import pickle
from copy import deepcopy
from pickle import PicklingError


def my_deepcopy(obj):
    try:
        return pickle.loads(pickle.dumps(obj))
    except PicklingError:
        return deepcopy(obj)
