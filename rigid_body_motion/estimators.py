""""""
import numpy as np
from quaternion import as_float_array, from_rotation_matrix

from rigid_body_motion.coordinate_systems import _replace_dim
from rigid_body_motion.core import _make_dataarray, _maybe_unpack_dataarray
from rigid_body_motion.utils import rotate_vectors


def _reshape_vectors(v1, v2, axis, dim):
    """ Reshape input vectors to two dimensions. """
    # TODO v2 as DataArray with possibly different dimension order
    v1, axis, _, _, _, _, coords, *_ = _maybe_unpack_dataarray(
        v1, dim, axis, None
    )
    v2, *_ = _maybe_unpack_dataarray(v2, None, axis, None)

    if v1.shape[axis] != 3:
        raise ValueError(
            f"Shape of v1 along axis {axis} must be 3, got {v1.shape[axis]}"
        )
    if v1.ndim < 2:
        raise ValueError("v1 must have at least two dimensions")
    if v1.shape != v2.shape:
        raise ValueError("v1 and v2 must have the same shape")

    # flatten everything except spatial dimension
    v1 = np.swapaxes(v1, axis, -1).reshape(-1, 3)
    v2 = np.swapaxes(v2, axis, -1).reshape(-1, 3)

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
    m = 3
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
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
    """  # noqa
    v1, v2, was_dataarray = _reshape_vectors(v1, v2, axis, dim)

    v1_new = np.copy(v1)

    # apply the initial pose estimation
    if init_transform is not None:
        t, r = init_transform
        v1_new = rotate_vectors(np.asarray(r), v1_new) + np.asarray(t)

    prev_error = 0

    for i in range(max_iterations):
        distances = np.linalg.norm(v1_new - v2, axis=1)

        # compute the transformation between the current source and nearest
        # destination points
        t, r = best_fit_transform(v1_new, v2)

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
