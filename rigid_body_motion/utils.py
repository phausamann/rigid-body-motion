""""""
import numpy as np

from quaternion import quaternion, as_float_array, as_quat_array
from quaternion import rotate_vectors as quat_rv


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


def rotate_vectors(q, v, axis=-1, one_to_one=True):
    """

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


def _maybe_unpack_dataarray(arr, dim=None, axis=None, timestamps=None):
    """ If input is DataArray, unpack into data, coords and dims. """
    if not is_dataarray(arr):
        if dim is not None:
            raise ValueError(
                'dim argument specified without DataArray input.')
        axis = axis or -1
        coords = None
        dims = None
    else:
        if dim is not None and axis is not None:
            raise ValueError('You can either specify the dim or the axis '
                             'argument, not both.')
        elif dim is not None:
            axis = arr.dims.index(dim)
        else:
            axis = axis or -1
        if isinstance(timestamps, str):
            # TODO transpose if time dim is not first?
            # TODO convert datetimeindex?
            timestamps = arr[timestamps].data
        elif timestamps is not None:
            # TODO time_dim argument
            raise NotImplementedError(
                'timestamps argument must be dimension name or None.')
        coords = dict(arr.coords)
        dims = arr.dims
        arr = arr.data

    return arr, axis, timestamps, coords, dims


def _make_dataarray(arr, coords, dims, ts_arg, ts_out):
    """ Make DataArray out of transformation results. """
    import xarray as xr

    if ts_arg is None:
        # no timestamps specified
        if ts_out is not None:
            coords['time'] = ts_out
            dims = ('time',) + dims
    elif isinstance(ts_arg, str):
        # timestamps specified as coord
        # TODO transpose if time dim is not first?
        assert ts_out is not None
        if len(coords[ts_arg]) != len(ts_out) \
                or np.any(coords[ts_arg] != ts_out):
            # interpolate if timestamps after transform have changed
            for c in coords:
                if ts_arg in coords[c].dims and c != ts_arg:
                    coords[c] = coords[c].interp({ts_arg: ts_out})
            coords[ts_arg] = ts_out
    else:
        # timestamps specified as array
        # TODO time_dim argument
        raise NotImplementedError(
            'timestamps argument must be dimension name or None')

    return xr.DataArray(arr, coords, dims)
