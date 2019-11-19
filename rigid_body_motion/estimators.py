""""""
import numpy as np


def shortest_arc_rotation(v1, v2, axis=-1):
    """ Estimate the shortest-arc rotation between two vectors.

    Parameters
    ----------
    v1: array_like, shape (..., 3, ...)
        The first vector.

    v2: array_like, shape (..., 3, ...)
        The second vector.

    axis: int, default -1
        The axis of the arrays representing the vector coordinates.

    Returns
    -------
    q: array_like, shape (..., 3, ...)
        The quaternion representation of the shortest-arc rotation.
    """
    sn1 = np.sum(v1 ** 2, axis=axis, keepdims=True)
    sn2 = np.sum(v2 ** 2, axis=axis, keepdims=True)
    d12 = np.sum(v1 * v2, axis=axis, keepdims=True)
    c12 = np.cross(v1, v2, axis=axis)
    q = np.concatenate((np.sqrt(sn1 * sn2) + d12, c12), axis=axis)
    q /= np.linalg.norm(q, axis=axis, keepdims=True)

    return q
