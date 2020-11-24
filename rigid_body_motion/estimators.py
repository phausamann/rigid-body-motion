""""""
import numpy as np
from quaternion import as_float_array, from_rotation_matrix
from scipy.spatial import cKDTree

from rigid_body_motion.core import (
    _estimate_angular_velocity,
    _estimate_linear_velocity,
    _make_dataarray,
    _maybe_unpack_dataarray,
    _replace_dim,
)
from rigid_body_motion.utils import rotate_vectors


def _reshape_vectors(v1, v2, axis, dim, same_shape=True):
    """ Reshape input vectors to two dimensions. """
    # TODO v2 as DataArray with possibly different dimension order
    v1, axis, _, _, _, _, coords, *_ = _maybe_unpack_dataarray(
        v1, dim, axis, None, False
    )
    v2, *_ = _maybe_unpack_dataarray(v2, None, axis, None)

    if v1.shape[axis] != 3 or v2.shape[axis] != 3:
        raise ValueError(
            f"Shape of v1 and v2 along axis {axis} must be 3, got "
            f"{v1.shape[axis]} for v1 and {v2.shape[axis]} for v2"
        )
    if v1.ndim < 2:
        raise ValueError("v1 must have at least two dimensions")

    # flatten everything except spatial dimension
    v1 = np.swapaxes(v1, axis, -1).reshape(-1, 3)
    v2 = np.swapaxes(v2, axis, -1).reshape(-1, 3)

    if same_shape and v1.shape != v2.shape:
        raise ValueError("v1 and v2 must have the same shape")

    return v1, v2, coords is not None


def _make_transform_dataarrays(translation, rotation):
    """ Make translation and rotation DataArrays. """
    import xarray as xr

    translation = xr.DataArray(
        translation,
        {"cartesian_axis": ["x", "y", "z"]},
        "cartesian_axis",
        name="translation",
    )
    rotation = xr.DataArray(
        rotation,
        {"quaternion_axis": ["w", "x", "y", "z"]},
        "quaternion_axis",
        name="rotation",
    )

    return translation, rotation


def estimate_linear_velocity(
    arr,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
    outlier_thresh=None,
    cutoff=None,
):
    """ Estimate linear velocity from a time series of translation.

    Parameters
    ----------
    arr: array_like
        Array of translations.

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

    cutoff: float, optional
        Frequency of a low-pass filter applied to the linear velocity after
        the estimation as a fraction of the Nyquist frequency.

    outlier_thresh: float, optional
        Some SLAM-based trackers introduce position corrections when a new
        camera frame becomes available. This introduces outliers in the
        linear velocity estimate. The estimation algorithm used here
        can suppress these outliers by throwing out samples where the
        norm of the second-order differences of the position is above
        `outlier_thresh` and interpolating the missing values. For
        measurements from the Intel RealSense T265 tracker, set this value
        to 1e-3.

    Returns
    -------
    linear: array_like
        Array of linear velocities.
    """
    (
        arr,
        axis,
        dim,
        time_axis,
        time_dim,
        timestamps,
        coords,
        dims,
        name,
        attrs,
    ) = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, time_axis=time_axis, timestamps=timestamps
    )

    linear = _estimate_linear_velocity(
        arr,
        timestamps,
        time_axis=time_axis,
        outlier_thresh=outlier_thresh,
        cutoff=cutoff,
    )

    if coords is not None:
        return _make_dataarray(
            linear, coords, dims, name, attrs, time_dim, timestamps
        )
    else:
        return linear


def estimate_angular_velocity(
    arr,
    dim=None,
    axis=None,
    timestamps=None,
    time_axis=None,
    mode="quaternion",
    outlier_thresh=None,
    cutoff=None,
):
    """ Estimate angular velocity from a time series of rotations.

    Parameters
    ----------
    arr: array_like
        Array of rotations, expressed in quaternions.

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

    mode: str, default "quaternion"
        If "quaternion", compute the angular velocity from the quaternion
        derivative. If "rotation_vector", compute the angular velocity from
        the gradient of the axis-angle representation of the rotations.

    outlier_thresh: float, optional
        Suppress samples where the norm of the second-order differences of the
        rotation is above `outlier_thresh` and interpolate the missing values.

    cutoff: float, optional
        Frequency of a low-pass filter applied to the angular velocity after
        the estimation as a fraction of the Nyquist frequency.

    Returns
    -------
    angular: array_like
        Array of angular velocities.
    """
    (
        arr,
        axis,
        dim,
        time_axis,
        time_dim,
        timestamps,
        coords,
        dims,
        name,
        attrs,
    ) = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, time_axis=time_axis, timestamps=timestamps
    )

    angular = _estimate_angular_velocity(
        arr,
        timestamps,
        axis=axis,
        time_axis=time_axis,
        mode=mode,
        outlier_thresh=outlier_thresh,
        cutoff=cutoff,
    )

    if coords is not None:
        coords, dims = _replace_dim(coords, dims, axis, "cartesian", 3)
        return _make_dataarray(
            angular, coords, dims, name, attrs, time_dim, timestamps
        )
    else:
        return angular


def shortest_arc_rotation(v1, v2, dim=None, axis=None):
    """ Estimate the shortest-arc rotation between two arrays of vectors.

    Parameters
    ----------
    v1: array_like, shape (..., 3, ...)
        The first array of vectors.

    v2: array_like, shape (..., 3, ...)
        The second array of vectors.

    dim: str, optional
        If the first array is a DataArray, the name of the dimension
        representing the spatial coordinates of the vectors.

    axis: int, optional
        The axis of the arrays representing the spatial coordinates of the
        vectors. Defaults to the last axis of the arrays.

    Returns
    -------
    rotation: array_like, shape (..., 4, ...)
        The quaternion representation of the shortest-arc rotation.
    """
    # TODO accept tuple for v2
    v1, axis, _, _, _, _, coords, dims, name, attrs = _maybe_unpack_dataarray(
        v1, dim, axis, None
    )

    sn1 = np.sum(v1 ** 2, axis=axis, keepdims=True)
    sn2 = np.sum(v2 ** 2, axis=axis, keepdims=True)
    d12 = np.sum(v1 * v2, axis=axis, keepdims=True)
    c12 = np.cross(v1, v2, axis=axis)
    rotation = np.concatenate((np.sqrt(sn1 * sn2) + d12, c12), axis=axis)
    rotation /= np.linalg.norm(rotation, axis=axis, keepdims=True)

    if coords is not None:
        coords, dims = _replace_dim(coords, dims, axis, "quaternion", 3)
        return _make_dataarray(rotation, coords, dims, name, attrs, None, None)
    else:
        return rotation


def best_fit_rotation(v1, v2, dim=None, axis=None):
    """ Least-squares best-fit rotation between two arrays of vectors.

    Finds the rotation `r` that minimizes:

    .. math:: || v_2 - rot(r, v_1) ||

    Parameters
    ----------
    v1: array_like, shape (..., 3, ...)
        The first array of vectors.

    v2: array_like, shape (..., 3, ...)
        The second array of vectors.

    dim: str, optional
        If the first array is a DataArray, the name of the dimension
        representing the spatial coordinates of the vectors.

    axis: int, optional
        The axis of the arrays representing the spatial coordinates of the
        vectors. Defaults to the last axis of the arrays.

    Returns
    -------
    rotation: array_like, shape (4,)
        Rotation of transform.

    References
    ----------
    Adapted from https://github.com/ClayFlannigan/icp

    See Also
    --------
    iterative_closest_point, best_fit_transform
    """
    v1, v2, was_dataarray = _reshape_vectors(v1, v2, axis, dim)

    # rotation matrix
    H = np.dot(v1.T, v2)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # rotation as quaternion
    rotation = as_float_array(from_rotation_matrix(R))

    if was_dataarray:
        import xarray as xr

        rotation = xr.DataArray(
            rotation,
            {"quaternion_axis": ["w", "x", "y", "z"]},
            "quaternion_axis",
            name="rotation",
        )

    return rotation


def best_fit_transform(v1, v2, dim=None, axis=None):
    """ Least-squares best-fit transform between two arrays of vectors.

    Finds the rotation `r` and the translation `t` that minimize:

    .. math:: || v_2 - (rot(r, v_1) + t) ||

    Parameters
    ----------
    v1: array_like, shape (..., 3, ...)
        The first array of vectors.

    v2: array_like, shape (..., 3, ...)
        The second array of vectors.

    dim: str, optional
        If the first array is a DataArray, the name of the dimension
        representing the spatial coordinates of the vectors.

    axis: int, optional
        The axis of the arrays representing the spatial coordinates of the
        vectors. Defaults to the last axis of the arrays.

    Returns
    -------
    translation: array_like, shape (3,)
        Translation of transform.

    rotation: array_like, shape (4,)
        Rotation of transform.

    References
    ----------
    Adapted from https://github.com/ClayFlannigan/icp

    See Also
    --------
    iterative_closest_point, best_fit_rotation
    """
    v1, v2, was_dataarray = _reshape_vectors(v1, v2, axis, dim)

    # translate points to their centroids
    mean_v1 = np.mean(v1, axis=0)
    mean_v2 = np.mean(v2, axis=0)
    v1_centered = v1 - mean_v1
    v2_centered = v2 - mean_v2

    # rotation matrix
    H = np.dot(v1_centered.T, v2_centered)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # rotation as quaternion
    rotation = as_float_array(from_rotation_matrix(R))

    # translation
    translation = mean_v2.T - np.dot(R, mean_v1.T)

    if was_dataarray:
        translation, rotation = _make_transform_dataarrays(
            translation, rotation
        )

    return translation, rotation


def _nearest_neighbor(v1, v2):
    """ Find the nearest neighbor in v2 for each point in v1. """
    kd_tree = cKDTree(v2)
    distances, idx = kd_tree.query(v1, 1)

    return idx.ravel(), distances.ravel()


def iterative_closest_point(
    v1,
    v2,
    dim=None,
    axis=None,
    init_transform=None,
    max_iterations=20,
    tolerance=1e-3,
):
    """ Iterative closest point algorithm matching two arrays of vectors.

    Finds the rotation `r` and the translation `t` such that:

    .. math:: v_2 \simeq rot(r, v_1) + t

    Parameters
    ----------
    v1: array_like, shape (..., 3, ...)
        The first array of vectors.

    v2: array_like, shape (..., 3, ...)
        The second array of vectors.

    dim: str, optional
        If the first array is a DataArray, the name of the dimension
        representing the spatial coordinates of the vectors.

    axis: int, optional
        The axis of the arrays representing the spatial coordinates of the
        vectors. Defaults to the last axis of the arrays.

    init_transform: tuple, optional
        Initial guess as (translation, rotation) tuple.

    max_iterations: int, default 20
        Maximum number of iterations.

    tolerance: float, default 1e-3
        Abort if the mean distance error between the transformed arrays does
        not improve by more than this threshold between iterations.

    Returns
    -------
    translation: array_like, shape (3,)
        Translation of transform.

    rotation: array_like, shape (4,)
        Rotation of transform.

    References
    ----------
    Adapted from https://github.com/ClayFlannigan/icp

    Notes
    -----
    For points with known correspondences (e.g. timeseries of positions), it is
    recommended to interpolate the points to a common sampling base and use the
    `best_fit_transform` method.

    See Also
    --------
    best_fit_transform, best_fit_rotation
    """  # noqa
    v1, v2, was_dataarray = _reshape_vectors(
        v1, v2, axis, dim, same_shape=False
    )

    v1_new = np.copy(v1)

    # apply the initial pose estimation
    if init_transform is not None:
        t, r = init_transform
        v1_new = rotate_vectors(np.asarray(r), v1_new) + np.asarray(t)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination
        # points
        idx, distances = _nearest_neighbor(v1_new, v2)

        # compute the transformation between the current source and nearest
        # destination points
        t, r = best_fit_transform(v1_new, v2[idx])

        # update the current source
        v1_new = rotate_vectors(r, v1_new) + t

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    translation, rotation = best_fit_transform(v1, v1_new)

    if was_dataarray:
        translation, rotation = _make_transform_dataarrays(
            translation, rotation
        )

    return translation, rotation
