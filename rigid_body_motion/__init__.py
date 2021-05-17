"""Top-level package for rigid-body-motion."""
__author__ = """Peter Hausamann"""
__email__ = "peter.hausamann@tum.de"
__version__ = "0.6.0"

from . import io, plot, ros  # noqa
from .coordinate_systems import (
    cartesian_to_polar,
    cartesian_to_spherical,
    polar_to_cartesian,
    spherical_to_cartesian,
)
from .core import (
    _make_dataarray,
    _make_transform_or_pose_dataset,
    _make_twist_dataset,
    _make_velocity_dataarray,
    _maybe_unpack_dataarray,
    _replace_dim,
    _resolve_rf,
    _transform,
)
from .estimators import (
    best_fit_rotation,
    best_fit_transform,
    estimate_angular_velocity,
    estimate_linear_velocity,
    iterative_closest_point,
    shortest_arc_rotation,
)
from .reference_frames import ReferenceFrame
from .reference_frames import _registry as registry
from .reference_frames import (
    clear_registry,
    deregister_frame,
    register_frame,
    render_tree,
)
from .utils import ExampleDataStore, qinterp, qinv, qmean, qmul, rotate_vectors

try:
    import rigid_body_motion.accessors  # noqa
except ImportError:
    pass

__all__ = [
    "transform_points",
    "transform_quaternions",
    "transform_vectors",
    "transform_angular_velocity",
    "transform_linear_velocity",
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
    "render_tree",
    # estimators
    "estimate_linear_velocity",
    "estimate_angular_velocity",
    "shortest_arc_rotation",
    "best_fit_rotation",
    "best_fit_transform",
    "iterative_closest_point",
    "lookup_transform",
    "lookup_pose",
    "lookup_twist",
    "lookup_linear_velocity",
    "lookup_angular_velocity",
    # utils
    "example_data",
    "qinterp",
    "qinv",
    "qmean",
    "qmul",
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


example_data = ExampleDataStore()


def transform_vectors(
    arr,
    into,
    outof=None,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
    return_timestamps=False,
):
    """ Transform an array of vectors between reference frames.

    Parameters
    ----------
    arr: array_like
        The array to transform.

    into: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array will be represented after the transformation.

    outof: str or ReferenceFrame, optional
        ReferenceFrame instance or name of a registered reference frame in
        which the array is currently represented. Can be omitted if the array
        is a DataArray whose ``attrs`` contain a "representation_frame" entry
        with the name of a registered frame.

    dim: str, optional
        If the array is a DataArray, the name of the dimension
        representing the spatial coordinates of the vectors.

    axis: int, optional
        The axis of the array representing the spatial coordinates of the
        vectors. Defaults to the last axis of the array.

    timestamps: array_like or str, optional
        The timestamps of the vectors, corresponding to the `time_axis`
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The axis defined by `time_axis` will
        be re-sampled to the timestamps for which the transformation is
        defined.

    time_axis: int, optional
        The axis of the array representing the timestamps of the vectors.
        Defaults to the first axis of the array.

    return_timestamps: bool, default False
        If True, also return the timestamps after the transformation.

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
    return _transform(
        "transform_vectors",
        arr,
        into,
        outof,
        dim,
        axis,
        timestamps,
        time_axis,
        return_timestamps=return_timestamps,
    )


def transform_points(
    arr,
    into,
    outof=None,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
    return_timestamps=False,
):
    """ Transform an array of points between reference frames.

    Parameters
    ----------
    arr: array_like
        The array to transform.

    into: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array will be represented after the transformation.

    outof: str or ReferenceFrame, optional
        ReferenceFrame instance or name of a registered reference frame which
        is the current reference frame of the array. Can be omitted if the
        array is a DataArray whose ``attrs`` contain a "reference_frame" entry
        with the name of a registered frame.

    dim: str, optional
        If the array is a DataArray, the name of the dimension
        representing the spatial coordinates of the points.

    axis: int, optional
        The axis of the array representing the spatial coordinates of the
        points. Defaults to the last axis of the array.

    timestamps: array_like or str, optional
        The timestamps of the points, corresponding to the `time_axis`
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The axis defined by `time_axis` will
        be re-sampled to the timestamps for which the transformation is
        defined.

    time_axis: int, optional
        The axis of the array representing the timestamps of the points.
        Defaults to the first axis of the array.

    return_timestamps: bool, default False
        If True, also return the timestamps after the transformation.

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
    return _transform(
        "transform_points",
        arr,
        into,
        outof,
        dim,
        axis,
        timestamps,
        time_axis,
        return_timestamps=return_timestamps,
    )


def transform_quaternions(
    arr,
    into,
    outof=None,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
    return_timestamps=False,
):
    """ Transform an array of quaternions between reference frames.

    Parameters
    ----------
    arr: array_like
        The array to transform.

    into: str or ReferenceFrame
        ReferenceFrame instance or name of a registered reference frame in
        which the array will be represented after the transformation.

    outof: str or ReferenceFrame, optional
        ReferenceFrame instance or name of a registered reference frame which
        is the current reference frame of the array. Can be omitted if the
        array is a DataArray whose ``attrs`` contain a "reference_frame" entry
        with the name of a registered frame.

    dim: str, optional
        If the array is a DataArray, the name of the dimension
        representing the spatial coordinates of the quaternions.

    axis: int, optional
        The axis of the array representing the spatial coordinates of the
        quaternions. Defaults to the last axis of the array.

    timestamps: array_like or str, optional
        The timestamps of the quaternions, corresponding to the `time_axis`
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The axis defined by `time_axis` will
        be re-sampled to the timestamps for which the transformation is
        defined.

    time_axis: int, optional
        The axis of the array representing the timestamps of the quaternions.
        Defaults to the first axis of the array.

    return_timestamps: bool, default False
        If True, also return the timestamps after the transformation.

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
    return _transform(
        "transform_quaternions",
        arr,
        into,
        outof,
        dim,
        axis,
        timestamps,
        time_axis,
        return_timestamps=return_timestamps,
    )


def transform_angular_velocity(
    arr,
    into,
    outof=None,
    what="reference_frame",
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
    cutoff=None,
    return_timestamps=False,
):
    """ Transform an array of angular velocities between frames.

    The array represents the velocity of a moving body or frame wrt a
    reference frame, expressed in a representation frame.

    The transformation changes either the reference frame, the moving
    frame or the representation frame of the velocity from this frame to
    another. In either case, it is assumed that the array is represented in
    the frame that is being changed and will be represented in the new
    frame after the transformation.

    When transforming the reference frame R to a new frame R' while keeping
    the moving frame M fixed, the transformed velocity is calculated
    according to the formula:

    .. math:: \omega_{M/R'} = \omega_{M/R} + \omega_{R/R'}

    When transforming the moving frame M to a new frame M' while keeping
    the reference frame R fixed, the transformed velocity is calculated
    according to the formula:

    .. math:: \omega_{M'/R} = \omega_{M/R} + \omega_{M'/M}

    Parameters
    ----------
    arr: array_like
        The array to transform.

    into: str or ReferenceFrame
        The target reference frame.

    outof: str or ReferenceFrame, optional
        The source reference frame. Can be omitted if the array
        is a DataArray whose ``attrs`` contain a "representation_frame",
        "reference_frame" or "moving_frame" entry with the name of a
        registered frame (depending on what you want to transform, see `what`).

    what: str
        What frame of the velocity to transform. Can be "reference_frame",
        "moving_frame" or "representation_frame".

    dim: str, optional
        If the array is a DataArray, the name of the dimension
        representing the spatial coordinates of the velocities.

    axis: int, optional
        The axis of the array representing the spatial coordinates of the
        velocities. Defaults to the last axis of the array.

    timestamps: array_like or str, optional
        The timestamps of the velocities, corresponding to the `time_axis`
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The axis defined by `time_axis` will
        be re-sampled to the timestamps for which the transformation is
        defined.

    time_axis: int, optional
        The axis of the array representing the timestamps of the velocities.
        Defaults to the first axis of the array.

    cutoff: float, optional
        Frequency of a low-pass filter applied to linear and angular
        velocity after the twist estimation as a fraction of the Nyquist
        frequency.

    return_timestamps: bool, default False
        If True, also return the timestamps after the transformation.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    ts: array_like
        The timestamps after the transformation.

    See Also
    --------
    transform_linear_velocity, transform_vectors, transform_quaternions,
    transform_points, ReferenceFrame
    """  # noqa
    return _transform(
        "transform_angular_velocity",
        arr,
        into,
        outof,
        dim,
        axis,
        timestamps,
        time_axis,
        what=what,
        cutoff=cutoff,
        return_timestamps=return_timestamps,
    )


def transform_linear_velocity(
    arr,
    into,
    outof=None,
    what="reference_frame",
    moving_frame=None,
    reference_frame=None,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
    cutoff=None,
    outlier_thresh=None,
    return_timestamps=False,
):
    """ Transform an array of linear velocities between frames.

    The array represents the velocity of a moving body or frame wrt a
    reference frame, expressed in a representation frame.

    The transformation changes either the reference frame, the moving
    frame or the representation frame of the velocity from this frame to
    another. In either case, it is assumed that the array is represented in
    the frame that is being changed and will be represented in the new
    frame after the transformation.

    When transforming the reference frame R to a new frame R' while keeping
    the moving frame M fixed, the transformed velocity is calculated
    according to the formula:

    .. math:: v_{M/R'} = v_{M/R} + v_{R/R'} + \omega_{R/R'} \\times t_{M/R}

    When transforming the moving frame M to a new frame M' while keeping
    the reference frame R fixed, the transformed velocity is calculated
    according to the formula:

    .. math:: v_{M'/R} = v_{M/R} + v_{M'/M} + \omega_{M/R} \\times t_{M'/M}

    Parameters
    ----------
    arr: array_like
        The array to transform.

    into: str or ReferenceFrame
        The target reference frame.

    outof: str or ReferenceFrame, optional
        The source reference frame. Can be omitted if the array
        is a DataArray whose ``attrs`` contain a "representation_frame",
        "reference_frame" or "moving_frame" entry with the name of a
        registered frame (depending on what you want to transform, see `what`).

    what: str
        What frame of the velocity to transform. Can be "reference_frame",
        "moving_frame" or "representation_frame".

    moving_frame: str or ReferenceFrame, optional
        The moving frame when transforming the reference frame of the
        velocity.

    reference_frame: str or ReferenceFrame, optional
        The reference frame when transforming the moving frame of the
        velocity.

    dim: str, optional
        If the array is a DataArray, the name of the dimension
        representing the spatial coordinates of the velocities.

    axis: int, optional
        The axis of the array representing the spatial coordinates of the
        velocities. Defaults to the last axis of the array.

    timestamps: array_like or str, optional
        The timestamps of the velocities, corresponding to the `time_axis`
        of the array. If str and the array is a DataArray, the name of the
        coordinate with the timestamps. The axis defined by `time_axis` will
        be re-sampled to the timestamps for which the transformation is
        defined.

    time_axis: int, optional
        The axis of the array representing the timestamps of the velocities.
        Defaults to the first axis of the array.

    cutoff: float, optional
        Frequency of a low-pass filter applied to linear and angular
        velocity after the twist estimation as a fraction of the Nyquist
        frequency.

    outlier_thresh: float, optional
        Some SLAM-based trackers introduce position corrections when a new
        camera frame becomes available. This introduces outliers in the
        linear velocity estimate. The estimation algorithm used here
        can suppress these outliers by throwing out samples where the
        norm of the second-order differences of the position is above
        `outlier_thresh` and interpolating the missing values. For
        measurements from the Intel RealSense T265 tracker, set this value
        to 1e-3.

    return_timestamps: bool, default False
        If True, also return the timestamps after the transformation.

    Returns
    -------
    arr_transformed: array_like
        The transformed array.

    ts: array_like
        The timestamps after the transformation.

    See Also
    --------
    transform_angular_velocity, transform_vectors, transform_quaternions,
    transform_points, ReferenceFrame
    """  # noqa
    return _transform(
        "transform_linear_velocity",
        arr,
        into,
        outof,
        dim,
        axis,
        timestamps,
        time_axis,
        what=what,
        moving_frame=moving_frame,
        reference_frame=reference_frame,
        cutoff=cutoff,
        outlier_thresh=outlier_thresh,
        return_timestamps=return_timestamps,
    )


def transform_coordinates(
    arr, into, outof=None, dim=None, axis=None, replace_dim=True
):
    """ Transform motion between coordinate systems.

    Parameters
    ----------
    arr: array_like
        The array to transform.

    into: str
        The name of a coordinate system in which the array will be represented
        after the transformation.

    outof: str, optional
        The name of a coordinate system in which the array is currently
        represented. Can be omitted if the array is a DataArray whose ``attrs``
        contain a "coordinate_system" entry with the name of a valid coordinate
        system.

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
    arr, axis, _, _, _, _, coords, dims, name, attrs = _maybe_unpack_dataarray(
        arr, dim, axis, timestamps=False
    )

    if outof is None:
        if attrs is not None and "coordinate_system" in attrs:
            # TODO warn if outof(.name) != attrs["coordinate_system"]
            outof = attrs["coordinate_system"]
        else:
            raise ValueError(
                "'outof' must be specified unless you provide a DataArray "
                "whose ``attrs`` contain a 'coordinate_system' entry with the "
                "name of a valid coordinate system"
            )

    try:
        transform_func = _cs_funcs[outof][into]
    except KeyError:
        raise ValueError(f"Unsupported transformation: {outof} to {into}.")

    if attrs is not None and "coordinate_system" in attrs:
        attrs.update({"coordinate_system": into})

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


def lookup_transform(outof, into, as_dataset=False, return_timestamps=False):
    """ Look up transformation from one frame to another.

    The transformation is a rotation `r` followed by a translation `t` which,
    when applied to a point expressed wrt the base frame `B`, yields that
    point wrt the target frame `T`:

    .. math:: p_T = rot(r, p_B) + t

    Parameters
    ----------
    outof: str or ReferenceFrame
        Base frame of the transformation.

    into: str or ReferenceFrame
        Target frame of the transformation.

    as_dataset: bool, default False
        If True, return an xarray.Dataset. Otherwise, return a tuple of
        translation and rotation.

    return_timestamps: bool, default False
        If True, and `as_dataset` is False, also return the timestamps of the
        lookup.

    Returns
    -------
    translation, rotation: each numpy.ndarray
        Translation and rotation of transformation between the frames,
        if `as_dataset` is False.

    timestamps: numpy.ndarray
        Corresponding timestamps of the lookup if `return_timestamps` is True.

    ds: xarray.Dataset
        The above arrays as an xarray.Dataset, if `as_dataset` is True.
    """
    into = _resolve_rf(into)
    outof = _resolve_rf(outof)
    translation, rotation, timestamps = outof.lookup_transform(into)

    if as_dataset:
        return _make_transform_or_pose_dataset(
            translation, rotation, outof, timestamps
        )
    elif return_timestamps:
        return translation, rotation, timestamps
    else:
        return translation, rotation


def lookup_pose(frame, reference, as_dataset=False, return_timestamps=False):
    """ Look up pose of one frame wrt a reference.

    Parameters
    ----------
    frame: str or ReferenceFrame
        Frame for which to look up the pose.

    reference: str or ReferenceFrame
        Reference frame of the pose.

    as_dataset: bool, default False
        If True, return an xarray.Dataset. Otherwise, return a tuple of
        position and orientation.

    return_timestamps: bool, default False
        If True, and `as_dataset` is False, also return the timestamps of the
        lookup.

    Returns
    -------
    position, orientation: each numpy.ndarray
        Position and orientation of the pose between the frames,
        if `as_dataset` is False.

    timestamps: numpy.ndarray
        Corresponding timestamps of the lookup if `return_timestamps` is True.

    ds: xarray.Dataset
        The above arrays as an xarray.Dataset, if `as_dataset` is True.
    """
    reference = _resolve_rf(reference)
    frame = _resolve_rf(frame)
    position, orientation, timestamps = frame.lookup_transform(reference)

    if as_dataset:
        return _make_transform_or_pose_dataset(
            position, orientation, reference, timestamps, pose=True
        )
    elif return_timestamps:
        return position, orientation, timestamps
    else:
        return position, orientation


def lookup_twist(
    frame,
    reference=None,
    represent_in=None,
    outlier_thresh=None,
    cutoff=None,
    mode="quaternion",
    as_dataset=False,
    return_timestamps=False,
):
    """ Estimate linear and angular velocity of a frame wrt a reference.

    Parameters
    ----------
    frame: str or ReferenceFrame
        The reference frame whose twist is estimated.

    reference: str or ReferenceFrame, optional
        The reference frame wrt which the twist is estimated. Defaults to
        the parent frame of the moving frame.

    represent_in: str or ReferenceFrame, optional
        The reference frame in which the twist is represented. Defaults
        to the reference frame.

    outlier_thresh: float, optional
        Some SLAM-based trackers introduce position corrections when a new
        camera frame becomes available. This introduces outliers in the
        linear velocity estimate. The estimation algorithm used here
        can suppress these outliers by throwing out samples where the
        norm of the second-order differences of the position is above
        `outlier_thresh` and interpolating the missing values. For
        measurements from the Intel RealSense T265 tracker, set this value
        to 1e-3.

    cutoff: float, optional
        Frequency of a low-pass filter applied to linear and angular
        velocity after the estimation as a fraction of the Nyquist
        frequency.

    mode: str, default "quaternion"
        If "quaternion", compute the angular velocity from the quaternion
        derivative. If "rotation_vector", compute the angular velocity from
        the gradient of the axis-angle representation of the rotations.

    as_dataset: bool, default False
        If True, return an xarray.Dataset. Otherwise, return a tuple of linear
        and angular velocity.

    return_timestamps: bool, default False
        If True, and `as_dataset` is False, also return the timestamps of the
        lookup.

    Returns
    -------
    linear, angular: each numpy.ndarray
        Linear and angular velocity of moving frame wrt reference frame,
        represented in representation frame, if `as_dataset` is False.

    timestamps: numpy.ndarray
        Corresponding timestamps of the lookup if `return_timestamps` is True.

    ds: xarray.Dataset
        The above arrays as an xarray.Dataset, if `as_dataset` is True.
    """
    frame = _resolve_rf(frame)
    reference = _resolve_rf(reference or frame.parent)
    represent_in = _resolve_rf(represent_in or reference)

    linear, angular, timestamps = frame.lookup_twist(
        reference,
        represent_in,
        outlier_thresh=outlier_thresh,
        cutoff=cutoff,
        mode=mode,
        return_timestamps=True,
    )

    if as_dataset:
        return _make_twist_dataset(
            angular, linear, frame, reference, represent_in, timestamps
        )
    elif return_timestamps:
        return linear, angular, timestamps
    else:
        return linear, angular


def lookup_linear_velocity(
    frame,
    reference=None,
    represent_in=None,
    outlier_thresh=None,
    cutoff=None,
    as_dataarray=False,
    return_timestamps=False,
):
    """ Estimate linear velocity of a frame wrt a reference.

    Parameters
    ----------
    frame: str or ReferenceFrame
        The reference frame whose velocity is estimated.

    reference: str or ReferenceFrame, optional
        The reference frame wrt which the velocity is estimated. Defaults to
        the parent frame of the moving frame.

    represent_in: str or ReferenceFrame, optional
        The reference frame in which the twist is represented. Defaults
        to the reference frame.

    outlier_thresh: float, optional
        Some SLAM-based trackers introduce position corrections when a new
        camera frame becomes available. This introduces outliers in the
        linear velocity estimate. The estimation algorithm used here
        can suppress these outliers by throwing out samples where the
        norm of the second-order differences of the position is above
        `outlier_thresh` and interpolating the missing values. For
        measurements from the Intel RealSense T265 tracker, set this value
        to 1e-3.

    cutoff: float, optional
        Frequency of a low-pass filter applied to linear and angular
        velocity after the estimation as a fraction of the Nyquist
        frequency.

    as_dataarray: bool, default False
        If True, return an xarray.DataArray.

    return_timestamps: bool, default False
        If True and `as_dataarray` is False, also return the timestamps of the
        lookup.

    Returns
    -------
    linear: numpy.ndarray or xarray.DataArray
        Linear velocity of moving frame wrt reference frame, represented in
        representation frame.

    timestamps: numpy.ndarray
        Corresponding timestamps of the lookup if `return_timestamps` is True.
    """
    frame = _resolve_rf(frame)
    reference = _resolve_rf(reference or frame.parent)
    represent_in = _resolve_rf(represent_in or reference)

    linear, timestamps = frame.lookup_linear_velocity(
        reference,
        represent_in,
        outlier_thresh=outlier_thresh,
        cutoff=cutoff,
        return_timestamps=True,
    )

    if as_dataarray:
        return _make_velocity_dataarray(
            linear, "linear", frame, reference, represent_in, timestamps
        )
    elif return_timestamps:
        return linear, timestamps
    else:
        return linear


def lookup_angular_velocity(
    frame,
    reference=None,
    represent_in=None,
    outlier_thresh=None,
    cutoff=None,
    mode="quaternion",
    as_dataarray=False,
    return_timestamps=False,
):
    """ Estimate angular velocity of a frame wrt a reference.

    Parameters
    ----------
    frame: str or ReferenceFrame
        The reference frame whose velocity is estimated.

    reference: str or ReferenceFrame, optional
        The reference frame wrt which the velocity is estimated. Defaults to
        the parent frame of the moving frame.

    represent_in: str or ReferenceFrame, optional
        The reference frame in which the twist is represented. Defaults
        to the reference frame.

    outlier_thresh: float, optional
        Suppress samples where the norm of the second-order differences of the
        rotation is above `outlier_thresh` and interpolate the missing values.

    cutoff: float, optional
        Frequency of a low-pass filter applied to angular and angular
        velocity after the estimation as a fraction of the Nyquist
        frequency.

    mode: str, default "quaternion"
        If "quaternion", compute the angular velocity from the quaternion
        derivative. If "rotation_vector", compute the angular velocity from
        the gradient of the axis-angle representation of the rotations.

    as_dataarray: bool, default False
        If True, return an xarray.DataArray.

    return_timestamps: bool, default False
        If True and `as_dataarray` is False, also return the timestamps of the
        lookup.

    Returns
    -------
    angular: numpy.ndarray or xarray.DataArray
        Angular velocity of moving frame wrt reference frame, represented in
        representation frame.

    timestamps: numpy.ndarray
        Corresponding timestamps of the lookup if `return_timestamps` is True.
    """
    frame = _resolve_rf(frame)
    reference = _resolve_rf(reference or frame.parent)
    represent_in = _resolve_rf(represent_in or reference)

    angular, timestamps = frame.lookup_angular_velocity(
        reference,
        represent_in,
        outlier_thresh=outlier_thresh,
        cutoff=cutoff,
        mode=mode,
        return_timestamps=True,
    )

    if as_dataarray:
        return _make_velocity_dataarray(
            angular, "angular", frame, reference, represent_in, timestamps,
        )
    elif return_timestamps:
        return angular, timestamps
    else:
        return angular
