""""""
import numpy as np


def cartesian_to_polar(arr, axis=-1):
    """ Transform cartesian to polar coordinates in two dimensions.

    Parameters
    ----------
    arr : array_like
        Input array.
    axis : int, default -1
        Axis of input array representing x and y in cartesian coordinates.
        Must be of length 2.

    Returns
    -------
    arr_polar : array_like
        Output array.
    """
    if arr.shape[axis] != 2:
        raise ValueError(
            f"Expected length of axis {axis} to be 2, got {arr.shape[axis]} "
            f"instead."
        )

    r = np.linalg.norm(arr, axis=axis)
    phi = np.arctan2(np.take(arr, 1, axis=axis), np.take(arr, 0, axis=axis))
    return np.stack((r, phi), axis=axis)


def polar_to_cartesian(arr, axis=-1):
    """ Transform polar to cartesian coordinates in two dimensions.

    Parameters
    ----------
    arr : array_like
        Input array.
    axis : int, default -1
        Axis of input array representing r and phi in polar coordinates.
        Must be of length 2.

    Returns
    -------
    arr_cartesian : array_like
        Output array.
    """
    if arr.shape[axis] != 2:
        raise ValueError(
            f"Expected length of axis {axis} to be 2, got {arr.shape[axis]} "
            f"instead."
        )

    x = np.take(arr, 0, axis=axis) * np.cos(np.take(arr, 1, axis=axis))
    y = np.take(arr, 0, axis=axis) * np.sin(np.take(arr, 1, axis=axis))
    return np.stack((x, y), axis=axis)


def cartesian_to_spherical(arr, axis=-1):
    """ Transform cartesian to spherical coordinates in three dimensions.

    The spherical coordinate system is defined according to ISO 80000-2.

    Parameters
    ----------
    arr : array_like
        Input array.
    axis : int, default -1
        Axis of input array representing x, y and z in cartesian coordinates.
        Must be of length 3.

    Returns
    -------
    arr_spherical : array_like
        Output array.
    """
    if arr.shape[axis] != 3:
        raise ValueError(
            f"Expected length of axis {axis} to be 3, got {arr.shape[axis]} "
            f"instead."
        )

    r = np.linalg.norm(arr, axis=axis)
    theta = np.arccos(np.take(arr, 2, axis=axis) / r)
    phi = np.arctan2(np.take(arr, 1, axis=axis), np.take(arr, 0, axis=axis))
    return np.stack((r, theta, phi), axis=axis)


def spherical_to_cartesian(arr, axis=-1):
    """ Transform spherical to cartesian coordinates in three dimensions.

    The spherical coordinate system is defined according to ISO 80000-2.

    Parameters
    ----------
    arr : array_like
        Input array.
    axis : int, default -1
        Axis of input array representing r, theta and phi in spherical
        coordinates. Must be of length 3.

    Returns
    -------
    arr_cartesian : array_like
        Output array.
    """
    if arr.shape[axis] != 3:
        raise ValueError(
            f"Expected length of axis {axis} to be 3, got {arr.shape[axis]} "
            f"instead."
        )

    x = (
        np.take(arr, 0, axis=axis)
        * np.sin(np.take(arr, 1, axis=axis))
        * np.cos(np.take(arr, 2, axis=axis))
    )
    y = (
        np.take(arr, 0, axis=axis)
        * np.sin(np.take(arr, 1, axis=axis))
        * np.sin(np.take(arr, 2, axis=axis))
    )
    z = np.take(arr, 0, axis=axis) * np.cos(np.take(arr, 1, axis=axis))
    return np.stack((x, y, z), axis=axis)
