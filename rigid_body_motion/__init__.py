"""Top-level package for rigid-body-motion."""
__author__ = """Peter Hausamann"""
__email__ = 'peter@hausamann.de'
__version__ = '0.1.0'

from rigid_body_motion.coordinate_systems import *

__all__ = [
    'transform',
    'cartesian_to_polar',
    'polar_to_cartesian',
]

_cs_funcs = {
    'cartesian': {'polar': cartesian_to_polar},
    'polar': {'cartesian': polar_to_cartesian},
}


def transform(arr, outof=None, into=None, axis=-1, **kwargs):
    """"""
    try:
        transform_func = _cs_funcs[outof][into]
    except KeyError:
        raise ValueError(
            'Unsupported transformation: {} to {}.'.format(outof, into))

    return transform_func(arr, axis=axis, **kwargs)
