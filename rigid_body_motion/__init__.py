"""Top-level package for rigid-body-motion."""
__author__ = """Peter Hausamann"""
__email__ = 'peter@hausamann.de'
__version__ = '0.1.0'

from rigid_body_motion.coordinate_systems import *

__all__ = [
    'transform',
    'cartesian_to_polar_2d',
    'polar_to_cartesian_2d',
]

_cs_funcs = {
    'cartesian': {'polar': {2: cartesian_to_polar_2d}}}


def transform(arr, outof=None, into=None, axis=-1, **kwargs):
    """"""
    dim = arr.shape[axis]
    try:
        transform_func = _cs_funcs[outof][into][dim]
    except KeyError:
        raise ValueError(
            'Unsupported transformation: {} to {} in {} dimensions.'.format(
                outof, into, dim))

    return transform_func(arr, axis=axis, **kwargs)
