"""Top-level package for rigid-body-motion."""
__author__ = """Peter Hausamann"""
__email__ = 'peter@hausamann.de'
__version__ = '0.1.0'

from rigid_body_motion.coordinate_systems import *
from rigid_body_motion.reference_frames import \
    register_frame, deregister_frame, clear_registry, ReferenceFrame
from rigid_body_motion.reference_frames import _registry as _rf_registry
from rigid_body_motion.utils import qmean

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
    # utils
    'qmean'
]

_cs_funcs = {
    'cartesian': {'polar': cartesian_to_polar,
                  'spherical': cartesian_to_spherical},
    'polar': {'cartesian': polar_to_cartesian},
    'spherical': {'cartesian': spherical_to_cartesian}
}


def transform(arr, outof=None, into=None, axis=-1, **kwargs):
    """ Transform motion between coordinate systems and reference frames.

    Parameters
    ----------
    arr: array-like
        The array to transform.

    outof: str
        The name of a coordinate system or registered reference frame in
        which the array is currently represented.

    into: str
        The name of a coordinate system or registered reference frame in
        which the array will be represented after the transformation.

    axis: int, default -1
        The axis of the array representing the coordinates of the angular or
        linear motion.

    Returns
    -------
    arr_transformed: array-like
        The transformed array.
    """
    # TODO support ReferenceFrame objects
    if outof in _rf_registry:
        transformation_func = _rf_registry[outof].get_transformation_func(into)
    else:
        try:
            transformation_func = _cs_funcs[outof][into]
        except KeyError:
            raise ValueError(
                'Unsupported transformation: {} to {}.'.format(outof, into))

    return transformation_func(arr, axis=axis, **kwargs)
