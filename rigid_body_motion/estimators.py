""""""
import numpy as np

from rigid_body_motion.core import _maybe_unpack_dataarray, _make_dataarray
from rigid_body_motion.coordinate_systems import _replace_dim


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
        representing the coordinates of the vectors.

    axis: int, optional
        The axis of the arrays representing the coordinates of the vectors.
        Defaults to the last axis of the arrays.

    Returns
    -------
    q: array_like, shape (..., 4, ...)
        The quaternion representation of the shortest-arc rotation.
    """
    # TODO accept tuple for v2
    v1, axis, _, coords, dims, name, attrs = \
        _maybe_unpack_dataarray(v1, dim, axis, None)

    sn1 = np.sum(v1 ** 2, axis=axis, keepdims=True)
    sn2 = np.sum(v2 ** 2, axis=axis, keepdims=True)
    d12 = np.sum(v1 * v2, axis=axis, keepdims=True)
    c12 = np.cross(v1, v2, axis=axis)
    q = np.concatenate((np.sqrt(sn1 * sn2) + d12, c12), axis=axis)
    q /= np.linalg.norm(q, axis=axis, keepdims=True)

    if coords is not None:
        coords, dims = _replace_dim(coords, dims, axis, 'quaternion', 3)
        return _make_dataarray(q, coords, dims, name, attrs, None, None)
    else:
        return q
