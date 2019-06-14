"""Top-level package for rigid-body-motion."""
__author__ = """Peter Hausamann"""
__email__ = 'peter@hausamann.de'
__version__ = '0.1.0'

from rigid_body_motion.coordinate_systems import *
from rigid_body_motion.reference_frames import \
    register_frame, deregister_frame, clear_registry, ReferenceFrame
from rigid_body_motion.reference_frames import _registry as _rf_registry

__all__ = [
    'transform',
    # coordinate system transforms
    'cartesian_to_polar',
    'polar_to_cartesian',
    'cartesian_to_spherical',
    'spherical_to_cartesian',
    # reference frames
    'register_frame',
    'deregister_frame',
    'clear_registry',
    'ReferenceFrame',
]

_cs_funcs = {
    'cartesian': {'polar': cartesian_to_polar,
                  'spherical': cartesian_to_spherical},
    'polar': {'cartesian': polar_to_cartesian},
    'spherical': {'cartesian': spherical_to_cartesian}
}


def transform(arr, outof=None, into=None, axis=-1, **kwargs):
    """"""
    if outof in _rf_registry:
        transform_func = _rf_registry[outof].get_transform_func(into)
    else:
        try:
            transform_func = _cs_funcs[outof][into]
        except KeyError:
            raise ValueError(
                'Unsupported transformation: {} to {}.'.format(outof, into))

    return transform_func(arr, axis=axis, **kwargs)
