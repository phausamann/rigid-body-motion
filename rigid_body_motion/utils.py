""""""
import numpy as np

from quaternion import quaternion, as_float_array, as_quat_array
from quaternion import rotate_vectors as quat_rv

from rigid_body_motion.core import _resolve_axis


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
    # TODO 4-arrays instead of quaternions
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


def rotate_vectors(q, v, axis=-1, one_to_one=True):
    """ Rotate an array of vectors by an array of quaternions.

    Parameters
    ----------
    q : array_like, quaternion dtype
        Array of quaternions.

    v : array_like
        The array of vectors to be rotated.

    axis : int, default -1
        The axis of the ``v`` array representing the coordinates of the
        vectors. Must have length 3.

    one_to_one : bool, default True
        If True, rotate each vector by a single quaternion. In this case,
        non-singleton dimensions of ``q`` and ``v`` must match. Otherwise,
        perform rotations for all combinations of ``q`` and ``v``.

    Returns
    -------
    vr : array_like
        The array of rotated vectors. If ``one_to_one=True`` this array has
        the shape of all non-singleton dimensions in ``q`` and ``v``.
        Otherwise, this array has shape ``q.shape`` + ``v.shape``.
    """
    # TODO 4-arrays instead of quaternions
    if not one_to_one or q.ndim == 0:
        return quat_rv(q, v, axis=axis)

    if v.shape[axis] != 3:
        raise ValueError(
            'Expected axis {} of v to have length 3, got {}'.format(
                axis, v.shape[axis]))

    # make sure that non-singleton axes match
    v_shape = list(v.shape)
    v_shape.pop(axis)
    nonmatching_axes = (
        qs != vs for qs, vs in zip(q.shape, v_shape) if qs != 1 and vs != 1)
    if q.ndim != v.ndim - 1 or any(nonmatching_axes):
        raise ValueError(
            'Incompatible shapes for q and v: {} and {}.'.format(
                q.shape, v.shape))

    # compute rotation
    q = as_float_array(q)
    r = q[..., 1:]
    s = np.swapaxes(q[..., :1], -1, axis)
    m = np.swapaxes(np.linalg.norm(q, axis=-1, keepdims=True), -1, axis)
    rxv = np.cross(r, v, axisb=axis, axisc=axis)
    vr = v + 2 * np.cross(r, s * v + rxv, axisb=axis, axisc=axis) / m

    return vr


def is_dataarray(obj, require_attrs=None):
    """ Check whether an object is a DataArray.

    Parameters
    ----------
    obj : anything
        The object to be checked.

    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a DataArray.

    Returns
    -------
    bool
        Whether the object is a DataArray or not.
    """
    require_attrs = require_attrs or [
        'values', 'coords', 'dims', 'to_dataset']

    return all([hasattr(obj, name) for name in require_attrs])


def is_dataset(obj, require_attrs=None):
    """ Check whether an object is a Dataset.

    Parameters
    ----------
    obj : anything
        The object to be checked.

    require_attrs : list of str, optional
        The attributes the object has to have in order to pass as a Dataset.

    Returns
    -------
    bool
        Whether the object is a Dataset or not.
    """
    require_attrs = require_attrs or [
        'data_vars', 'coords', 'dims', 'to_array']

    return all([hasattr(obj, name) for name in require_attrs])
