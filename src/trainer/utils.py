from dataclasses import dataclass
from typing import TypeVar

import numpy as np
import tensorflow as tf


def shape(exp):
    if type(exp) is np.ndarray:
        return list(exp.shape)
    else:
        return []


def type_of(exp):
    if type(exp) is np.ndarray:
        return exp.dtype
    else:
        return type(exp)


def dict_slice(d, idx):
    return {key: value[None, idx] for key, value in d.items()}


@tf.function
def dict_slices_tf(d, idx):
    return {key: value[idx] for key, value in d.items()}


def dict_slice_set(d, idx, assign):
    for key, value in d.items():
        value[idx] = assign[key]
    return d


def dict_to_tensor(d):
    return {key: tf.convert_to_tensor(value) for key, value in d.items()}


def toggle(value, a=True, b=False):
    if value == a:
        return b
    return a


@dataclass
class DecayParams:
    base: float = 1e-3
    decay_rate: float = 0.1
    decay_steps: float = 1_000_000


ParamType = TypeVar("ParamType")
