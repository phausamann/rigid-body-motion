"""Top-level package for rigid-body-motion."""
__author__ = """Peter Hausamann"""
__email__ = 'peter@hausamann.de'
__version__ = '0.1.0'

from warnings import warn

from rigid_body_motion.coordinate_systems import \
    cartesian_to_polar, polar_to_cartesian, cartesian_to_spherical, \
    spherical_to_cartesian
from rigid_body_motion.reference_frames import \
    register_frame, deregister_frame, clear_registry, ReferenceFrame
from rigid_body_motion.reference_frames import _registry as _rf_registry
from rigid_body_motion.utils import \
    qmean, rotate_vectors, _maybe_unpack_dataarray, _make_dataarray, _resolve

try:
    import rigid_body_motion.ros as ros
except ImportError:
    pass

__all__ = [
    'transform',
    'transform_points',
    'transform_quaternions',
    'transform_vectors',
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
    'qmean',
    'rotate_vectors',
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
    arr: array_like
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
    arr_transformed: array_like
        The transformed array.
    """
    # TODO support ReferenceFrame objects
    if outof in _rf_registry:
        warn('transform for reference frame transformations is deprecated, '
             'use transform_points, transform_vectors or '
             'transform_quaternions instead.', DeprecationWarning)
        transformation_func = _rf_registry[outof].get_transformation_func(into)
    else:
        try:
            transformation_func = _cs_funcs[outof][into]
        except KeyError:
            raise ValueError(
                'Unsupported transformation: {} to {}.'.format(outof, into))

    return transformation_func(arr, axis=axis, **kwargs)


def transform_vectors(
        arr, outof=None, into=None, dim=None, axis=None, timestamps=None):
    """ Transform an array of vectors between reference frames.

    Parameters
    ----------
    arr: array_like
        The array to transform.

    outof: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array is currently represented.

    into: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array will be represented after the transformation.

    dim: str, optional
        If the array is a DataArray, the name of the dimension
        representing the coordinates of the vectors.

    axis: int, optional
        The axis of the array representing the coordinates of the vectors.
        Defaults to the last axis of the array.

    timestamps: array_like or str, optional
        The timestamps of the vectors, corresponding to the first axis
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The first axis of the array will be
        re-sampled to the timestamps for which the transformation is defined.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    ts: array_like
        The timestamps after the transformation.
    """
    arr, axis, ts_in, coords, dims = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, timestamps=timestamps)

    arr, ts_out = _resolve(outof).transform_vectors(
        arr, into, axis=axis, timestamps=ts_in, return_timestamps=True)

    if coords is not None:
        return _make_dataarray(arr, coords, dims, timestamps, ts_out)
    elif ts_out is not None:
        # TODO not so pretty. Maybe also introduce return_timestamps
        #  parameter and do this when return_timestamps=None
        return arr, ts_out
    else:
        return arr


def transform_points(
        arr, outof=None, into=None, dim=None, axis=None, timestamps=None):
    """ Transform an array of points between reference frames.

    Parameters
    ----------
    arr: array_like
        The array to transform.

    outof: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array is currently represented.

    into: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array will be represented after the transformation.

    dim: str, optional
        If the array is a DataArray, the name of the dimension
        representing the coordinates of the points.

    axis: int, optional
        The axis of the array representing the coordinates of the points.
        Defaults to the last axis of the array.

    timestamps: array_like or str, optional
        The timestamps of the points, corresponding to the first axis
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The first axis of the array will be
        re-sampled to the timestamps for which the transformation is defined.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    ts: array_like
        The timestamps after the transformation.
    """
    arr, axis, ts_in, coords, dims = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, timestamps=timestamps)

    arr, ts_out = _resolve(outof).transform_points(
        arr, into, axis=axis, timestamps=ts_in, return_timestamps=True)

    if coords is not None:
        return _make_dataarray(arr, coords, dims, timestamps, ts_out)
    elif ts_out is not None:
        # TODO not so pretty. Maybe also introduce return_timestamps
        #  parameter and do this when return_timestamps=None
        return arr, ts_out
    else:
        return arr


def transform_quaternions(
        arr, outof=None, into=None, dim=None, axis=None, timestamps=None):
    """ Transform an array of quaternions between reference frames.

    Parameters
    ----------
    arr: array_like
        The array to transform.

    outof: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array is currently represented.

    into: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array will be represented after the transformation.

    dim: str, optional
        If the array is a DataArray, the name of the dimension
        representing the coordinates of the quaternions.

    axis: int, optional
        The axis of the array representing the coordinates of the quaternions.
        Defaults to the last axis of the array.

    timestamps: array_like or str, optional
        The timestamps of the quaternions, corresponding to the first axis
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The first axis of the array will be
        re-sampled to the timestamps for which the transformation is defined.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    ts: array_like
        The timestamps after the transformation.
    """
    arr, axis, ts_in, coords, dims = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, timestamps=timestamps)

    arr, ts_out = _resolve(outof).transform_quaternions(
        arr, into, axis=axis, timestamps=ts_in, return_timestamps=True)

    if coords is not None:
        return _make_dataarray(arr, coords, dims, timestamps, ts_out)
    elif ts_out is not None:
        # TODO not so pretty. Maybe also introduce return_timestamps
        #  parameter and do this when return_timestamps=None
        return arr, ts_out
    else:
        return arr
