""""""
import numpy as np


def cartesian_to_polar(arr, axis=-1):
    """ Transform cartesian to polar coordinates in two dimensions.

    Parameters
    ----------
    arr : array-like
        Input array.
    axis : int, default -1
        Axis of input array representing x and y in cartesian coordinates.
        Must be of length 2.

    Returns
    -------
    arr_polar : array-like
        Output array.
    """
    if arr.shape[axis] != 2:
        raise ValueError(
            'Expected length of axis {} to be 2, got {} instead.'.format(
                axis, arr.shape[axis]))

    r = np.linalg.norm(arr, axis=axis)
    phi = np.arctan2(np.take(arr, 1, axis=axis), np.take(arr, 0, axis=axis))
    return np.stack((r, phi), axis=axis)


def polar_to_cartesian(arr, axis=-1):
    """ Transform polar to cartesian coordinates in two dimensions.

    Parameters
    ----------
    arr : array-like
        Input array.
    axis : int, default -1
        Axis of input array representing r and phi in polar coordinates.
        Must be of length 2.

    Returns
    -------
    arr_cartesian : array-like
        Output array.
    """
    if arr.shape[axis] != 2:
        raise ValueError(
            'Expected length of axis {} to be 2, got {} instead.'.format(
                axis, arr.shape[axis]))

    x = np.take(arr, 0, axis=axis) * np.cos(np.take(arr, 1, axis=axis))
    y = np.take(arr, 0, axis=axis) * np.sin(np.take(arr, 1, axis=axis))
    return np.stack((x, y), axis=axis)
