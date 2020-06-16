""""""
import numpy as np


def _replace_dim(coords, dims, axis, into, dimensionality):
    """ Replace the dimension after coordinate transformation """
    # TODO can we improve this with assign_coords / swap_dims?
    old_dim = dims[axis]

    if dimensionality == 2:
        if into == "cartesian":
            new_dim = "cartesian_axis"
            new_coord = ["x", "y"]
        elif into == "polar":
            new_dim = "polar_axis"
            new_coord = ["r", "phi"]
    elif dimensionality == 3:
        if into == "cartesian":
            new_dim = "cartesian_axis"
            new_coord = ["x", "y", "z"]
        elif into == "spherical":
            new_dim = "spherical_axis"
            new_coord = ["r", "theta", "phi"]
        elif into == "quaternion":
            new_dim = "quaternion_axis"
            new_coord = ["w", "x", "y", "z"]

    dims = tuple((d if d != old_dim else new_dim) for d in dims)

    coords = {c: coords[c] for c in coords if old_dim not in coords[c].dims}
    coords[new_dim] = new_coord

    return coords, dims


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
