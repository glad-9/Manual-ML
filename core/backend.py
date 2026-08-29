import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()
    HAS_GPU = True
except Exception as e:
    cp = None
    HAS_GPU = False
    logger.warning(f"No usable CUDA GPU found ({e.__class__.__name__}). Using NumPy.")

xp = cp if HAS_GPU else np


def to_cpu(array):
    """Safely extracts data from GPU back to a standard NumPy array."""
    if hasattr(array, "get"):
        return array.get()
    return array


def scatter_add(target, index, values):
    """
    Backend-agnostic scatter-add: target[index] += values, accumulating
    over repeated indices instead of overwriting (see im2col/pool2d backward).

    cupyx.scatter_add has no NumPy namesake - np.add.at is the CPU equivalent.
    """
    if HAS_GPU:
        from cupyx import scatter_add as _scatter_add

        _scatter_add(target, index, values)
    else:
        np.add.at(target, index, values)


def as_strided(array, shape, strides):
    """
    Backend-agnostic as_strided. Library cannot be accessed via xp so
    the paths are hard-coded.
    """
    if HAS_GPU:
        from cupy.lib.stride_tricks import as_strided as _as_strided
    else:
        from numpy.lib.stride_tricks import as_strided as _as_strided
    return _as_strided(array, shape=shape, strides=strides)
