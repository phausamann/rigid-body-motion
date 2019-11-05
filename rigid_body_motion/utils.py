""""""
import numpy as np

from quaternion import quaternion, as_float_array, as_quat_array


def _resolve_axis(axis, ndim):
    """ Convert axis argument into actual array axes. """
    if isinstance(axis, int) and axis < 0:
        axis = ndim + axis
    elif isinstance(axis, tuple):
        axis = tuple(ndim + a if a < 0 else a for a in axis)
    elif axis is None:
        axis = tuple(np.arange(ndim))

    if isinstance(axis, tuple):
        if any(a < 0 or a >= ndim for a in axis):
            raise IndexError('Axis index out of range')
    elif axis < 0 or axis >= ndim:
        raise IndexError('Axis index out of range')

    return axis


def qmean(q, axis=None):
    """ Quaternion mean.

    Adapted from https://github.com/christophhagen/averaging-quaternions.

    Parameters
    ----------
    q : array_like, quaternion dtype
        Array containing quaternions whose mean is to be computed.

    axis: None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

    Returns
    -------
    qm : ndarray, quaternion dtype
        A new array containing the mean values.
    """
    if q.dtype != quaternion:
        raise ValueError('Array dtype must be quaternion.')

    axis = _resolve_axis(axis, q.ndim)

    q = as_float_array(q)

    # compute outer product of quaternion elements
    q = q[..., np.newaxis]
    qt = np.swapaxes(q, -2, -1)
    A = np.mean(q * qt, axis=axis)

    # compute largest eigenvector of A
    l, v = np.linalg.eig(A)
    idx = np.unravel_index(l.argsort()[..., ::-1], l.shape) + (0,)
    v = v[idx]

    return as_quat_array(np.real(v))
