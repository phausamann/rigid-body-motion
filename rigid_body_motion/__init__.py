"""Top-level package for rigid-body-motion."""
__author__ = """Peter Hausamann"""
__email__ = "peter@hausamann.de"
__version__ = "0.1.0"
from rigid_body_motion import ros as ros  # noqa
from rigid_body_motion.coordinate_systems import (
    _replace_dim,
    cartesian_to_polar,
    cartesian_to_spherical,
    polar_to_cartesian,
    spherical_to_cartesian,
)
from rigid_body_motion.core import (
    _make_dataarray,
    _maybe_unpack_dataarray,
    _resolve_rf,
)
from rigid_body_motion.estimators import shortest_arc_rotation
from rigid_body_motion.reference_frames import ReferenceFrame
from rigid_body_motion.reference_frames import _registry as registry
from rigid_body_motion.reference_frames import (
    clear_registry,
    deregister_frame,
    register_frame,
)
from rigid_body_motion.utils import qmean, rotate_vectors

__all__ = [
    "transform_points",
    "transform_quaternions",
    "transform_vectors",
    # coordinate system transforms
    "cartesian_to_polar",
    "polar_to_cartesian",
    "cartesian_to_spherical",
    "spherical_to_cartesian",
    # reference frames
    "registry",
    "register_frame",
    "deregister_frame",
    "clear_registry",
    "ReferenceFrame",
    # estimators
    "shortest_arc_rotation",
    # utils
    "qmean",
    "rotate_vectors",
]

_cs_funcs = {
    "cartesian": {
        "polar": cartesian_to_polar,
        "spherical": cartesian_to_spherical,
    },
    "polar": {"cartesian": polar_to_cartesian},
    "spherical": {"cartesian": spherical_to_cartesian},
}


def transform_vectors(
    arr,
    outof=None,
    into=None,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
):
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
        The timestamps of the points, corresponding to the `time_axis`
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The axis defined by `time_axis` will
        be re-sampled to the timestamps for which the transformation is
        defined.

    time_axis: int, optional
        The axis of the array representing the timestamps of the points.
        Defaults to the first axis of the array.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    ts: array_like
        The timestamps after the transformation.

    See Also
    --------
    transform_quaternions, transform_points, ReferenceFrame
    """
    (
        arr,
        axis,
        time_axis,
        ts_in,
        coords,
        dims,
        name,
        attrs,
    ) = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, time_axis=time_axis, timestamps=timestamps
    )

    arr, ts_out = _resolve_rf(outof).transform_vectors(
        arr,
        into,
        axis=axis,
        time_axis=time_axis,
        timestamps=ts_in,
        return_timestamps=True,
    )

    if coords is not None:
        return _make_dataarray(
            arr, coords, dims, name, attrs, timestamps, ts_out
        )
    elif ts_out is not None:
        # TODO not so pretty. Maybe also introduce return_timestamps
        #  parameter and do this when return_timestamps=None
        return arr, ts_out
    else:
        return arr


def transform_points(
    arr,
    outof=None,
    into=None,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
):
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
        The timestamps of the points, corresponding to the `time_axis`
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The axis defined by `time_axis` will
        be re-sampled to the timestamps for which the transformation is
        defined.

    time_axis: int, optional
        The axis of the array representing the timestamps of the points.
        Defaults to the first axis of the array.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    ts: array_like
        The timestamps after the transformation.

    See Also
    --------
    transform_vectors, transform_quaternions, ReferenceFrame
    """
    (
        arr,
        axis,
        time_axis,
        ts_in,
        coords,
        dims,
        name,
        attrs,
    ) = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, time_axis=time_axis, timestamps=timestamps
    )

    arr, ts_out = _resolve_rf(outof).transform_points(
        arr,
        into,
        axis=axis,
        time_axis=time_axis,
        timestamps=ts_in,
        return_timestamps=True,
    )

    if coords is not None:
        return _make_dataarray(
            arr, coords, dims, name, attrs, timestamps, ts_out
        )
    elif ts_out is not None:
        # TODO not so pretty. Maybe also introduce return_timestamps
        #  parameter and do this when return_timestamps=None
        return arr, ts_out
    else:
        return arr


def transform_quaternions(
    arr,
    outof=None,
    into=None,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
):
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
        The timestamps of the points, corresponding to the `time_axis`
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The axis defined by `time_axis` will
        be re-sampled to the timestamps for which the transformation is
        defined.

    time_axis: int, optional
        The axis of the array representing the timestamps of the points.
        Defaults to the first axis of the array.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    ts: array_like
        The timestamps after the transformation.

    See Also
    --------
    transform_vectors, transform_points, ReferenceFrame
    """
    (
        arr,
        axis,
        time_axis,
        ts_in,
        coords,
        dims,
        name,
        attrs,
    ) = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, time_axis=time_axis, timestamps=timestamps
    )

    arr, ts_out = _resolve_rf(outof).transform_quaternions(
        arr,
        into,
        axis=axis,
        time_axis=time_axis,
        timestamps=ts_in,
        return_timestamps=True,
    )

    if coords is not None:
        return _make_dataarray(
            arr, coords, dims, name, attrs, timestamps, ts_out
        )
    elif ts_out is not None:
        # TODO not so pretty. Maybe also introduce return_timestamps
        #  parameter and do this when return_timestamps=None
        return arr, ts_out
    else:
        return arr


def transform_coordinates(
    arr, outof=None, into=None, dim=None, axis=None, replace_dim=True
):
    """ Transform motion between coordinate systems.

    Parameters
    ----------
    arr: array_like
        The array to transform.

    outof: str
        The name of a coordinate system in which the array is currently
        represented.

    into: str
        The name of a coordinate system in which the array will be represented
        after the transformation.

    dim: str, optional
        If the array is a DataArray, the name of the dimension representing
        the coordinates of the motion.

    axis: int, optional
        The axis of the array representing the coordinates of the motion.
        Defaults to the last axis of the array.

    replace_dim: bool, default True
        If True and the array is a DataArray, replace the dimension
        representing the coordinates by a new dimension that describes the
        new coordinate system and its axes (e.g.
        ``cartesian_axis: [x, y, z]``). All coordinates that contained the
        original dimension will be dropped.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    See Also
    --------
    cartesian_to_polar, polar_to_cartesian, cartesian_to_spherical,
    spherical_to_cartesian
    """
    try:
        transform_func = _cs_funcs[outof][into]
    except KeyError:
        raise ValueError(f"Unsupported transformation: {outof} to {into}.")

    arr, axis, _, _, coords, dims, name, attrs = _maybe_unpack_dataarray(
        arr, dim, axis
    )

    arr = transform_func(arr, axis=axis)

    if coords is not None:
        if replace_dim:
            # TODO accept (name, coord) tuple
            coords, dims = _replace_dim(
                coords, dims, axis, into, arr.shape[axis]
            )
        return _make_dataarray(arr, coords, dims, name, attrs, None, None)
    else:
        return arr
