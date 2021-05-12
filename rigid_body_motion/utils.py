""""""
import operator
from functools import reduce
from pathlib import Path

import numpy as np
from quaternion import as_float_array, as_quat_array, quaternion
from quaternion import rotate_vectors as quat_rv
from quaternion import squad

from rigid_body_motion.core import _resolve_axis


def qinv(q, qaxis=-1):
    """ Quaternion inverse.

    Parameters
    ----------
    q: array_like
        Array containing quaternions whose inverse is to be computed. Its dtype
        can be quaternion, otherwise `qaxis` specifies the axis representing
        the quaternions.

    qaxis: int, default -1
        If `q` is not quaternion dtype, axis of the quaternion array
        representing the coordinates of the quaternions.

    Returns
    -------
    qi: ndarray
        A new array containing the inverse values.
    """
    # TODO xarray support
    if q.dtype != quaternion:
        q = np.swapaxes(q, qaxis, -1)
        qi = as_float_array(1 / as_quat_array(q))
        return np.swapaxes(qi, -1, qaxis)
    else:
        return 1 / q


def qmul(*q, qaxis=-1):
    """ Quaternion multiplication.

    Parameters
    ----------
    q: iterable of array_like
        Arrays containing quaternions to multiply. Their dtype can be
        quaternion, otherwise `qaxis` specifies the axis representing
        the quaternions.

    qaxis: int, default -1
        If `q` are not quaternion dtype, axis of the quaternion arrays
        representing the coordinates of the quaternions.

    Returns
    -------
    qm: ndarray
        A new array containing the multiplied quaternions.
    """
    # TODO xarray support
    if len(q) < 2:
        raise ValueError("Please provide at least 2 quaternions to multiply")

    if all(qq.dtype != quaternion for qq in q):
        q = (as_quat_array(np.swapaxes(qq, qaxis, -1)) for qq in q)
        qm = reduce(operator.mul, q, 1)
        return np.swapaxes(as_float_array(qm), -1, qaxis)
    elif all(qq.dtype == quaternion for qq in q):
        return reduce(operator.mul, q, 1)
    else:
        raise ValueError(
            "Either all or none of the provided quaternions must be "
            "quaternion dtype"
        )


def qmean(q, axis=None, qaxis=-1):
    """ Quaternion mean.

    Adapted from https://github.com/christophhagen/averaging-quaternions.

    Parameters
    ----------
    q: array_like
        Array containing quaternions whose mean is to be computed. Its dtype
        can be quaternion, otherwise `qaxis` specifies the axis representing
        the quaternions.

    axis: None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

    qaxis: int, default -1
        If `q` is not quaternion dtype, axis of the quaternion array
        representing the coordinates of the quaternions.

    Returns
    -------
    qm: ndarray
        A new array containing the mean values.
    """
    # TODO xarray support
    if q.dtype != quaternion:
        q = np.swapaxes(q, qaxis, -1)
        was_quaternion = False
    else:
        q = as_float_array(q)
        was_quaternion = True

    axis = _resolve_axis(axis, q.ndim - 1)

    # compute outer product of quaternion elements
    q = q[..., np.newaxis]
    qt = np.swapaxes(q, -2, -1)
    A = np.mean(q * qt, axis=axis)

    # compute largest eigenvector of A
    l, v = np.linalg.eig(A)
    idx = np.unravel_index(l.argsort()[..., ::-1], l.shape) + (0,)
    v = v[idx]

    qm = np.real(v)

    if was_quaternion:
        return as_quat_array(qm)
    else:
        return np.swapaxes(qm, -1, qaxis)


def qinterp(q, t_in, t_out, axis=0, qaxis=-1):
    """ Quaternion interpolation.

    Parameters
    ----------
    q: array_like
        Array containing quaternions to interpolate. Its dtype
        can be quaternion, otherwise `qaxis` specifies the axis representing
        the quaternions.

    t_in: array_like
        Array of current sampling points of `q`.

    t_out: array_like
        Array of desired sampling points of `q`.

    axis: int, default 0
        Axis along which the quaternions are interpolated.

    qaxis: int, default -1
        If `q` is not quaternion dtype, axis of the quaternion array
        representing the coordinates of the quaternions.

    Returns
    -------
    qi: ndarray
        A new array containing the interpolated values.
    """
    # TODO xarray support
    axis = axis % q.ndim
    t_in = np.array(t_in).astype(float)
    t_out = np.array(t_out).astype(float)

    if q.dtype != quaternion:
        qaxis = qaxis % q.ndim
        # fix axis if it's the last axis of the array and will be swapped with
        # axis when converting to quaternion dtype
        if axis == q.ndim - 1:
            axis = qaxis
        q = as_quat_array(np.swapaxes(q, qaxis, -1))
        was_quaternion = False
    else:
        was_quaternion = True

    q = np.swapaxes(q, axis, 0)
    try:
        qi = squad(q, t_in, t_out)
    except ValueError:
        raise RuntimeError(
            "Error using SQUAD with multi-dimensional array, please upgrade "
            "the quaternion package to the latest version"
        )
    qi = np.swapaxes(qi, 0, axis)

    if was_quaternion:
        return qi
    else:
        return np.swapaxes(as_float_array(qi), -1, qaxis)


def rotate_vectors(q, v, axis=-1, qaxis=-1, one_to_one=True):
    """ Rotate an array of vectors by an array of quaternions.

    Parameters
    ----------
    q: array_like
        Array of quaternions. Its dtype can be quaternion, otherwise `q_axis`
        specifies the axis representing the quaternions.

    v: array_like
        The array of vectors to be rotated.

    axis: int, default -1
        The axis of the `v` array representing the coordinates of the
        vectors. Must have length 3.

    qaxis: int, default -1
        If `q` is not quaternion dtype, axis of the quaternion array
        representing the coordinates of the quaternions.

    one_to_one: bool, default True
        If True, rotate each vector by a single quaternion. In this case,
        non-singleton dimensions of `q` and `v` must match. Otherwise,
        perform rotations for all combinations of `q` and `v`.

    Returns
    -------
    vr: array_like
        The array of rotated vectors. If `one_to_one=True` this array has
        the shape of all non-singleton dimensions in `q` and `v`.
        Otherwise, this array has shape `q.shape` + `v.shape`.
    """
    # TODO proper broadcasting if v is DataArray
    if q.dtype != quaternion:
        q = as_quat_array(np.swapaxes(q, qaxis, -1))

    if not one_to_one or q.ndim == 0:
        return quat_rv(q, v, axis=axis)

    if v.shape[axis] != 3:
        raise ValueError(
            f"Expected axis {axis} of v to have length 3, got {v.shape[axis]}"
        )

    # make sure that non-singleton axes match
    v_shape = list(v.shape)
    v_shape.pop(axis)
    nonmatching_axes = (
        qs != vs for qs, vs in zip(q.shape, v_shape) if qs != 1 and vs != 1
    )
    if q.ndim != v.ndim - 1 or any(nonmatching_axes):
        raise ValueError(
            f"Incompatible shapes for q and v: {q.shape} and {v.shape}."
        )

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
    obj: anything
        The object to be checked.

    require_attrs: list of str, optional
        The attributes the object has to have in order to pass as a DataArray.

    Returns
    -------
    bool
        Whether the object is a DataArray or not.
    """
    require_attrs = require_attrs or [
        "values",
        "coords",
        "dims",
        "name",
        "attrs",
    ]

    return all([hasattr(obj, name) for name in require_attrs])


def is_dataset(obj, require_attrs=None):
    """ Check whether an object is a Dataset.

    Parameters
    ----------
    obj: anything
        The object to be checked.

    require_attrs: list of str, optional
        The attributes the object has to have in order to pass as a Dataset.

    Returns
    -------
    bool
        Whether the object is a Dataset or not.
    """
    require_attrs = require_attrs or [
        "data_vars",
        "coords",
        "dims",
        "to_array",
    ]

    return all([hasattr(obj, name) for name in require_attrs])


class ExampleDataStore:
    """ Storage interface for example data. """

    base_url = "https://github.com/phausamann/rbm-data/raw/main/"

    registry = {
        "head": (
            "head.nc",
            "874eddaa51bf775c7311f0046613c6f969adef6e34fe4aea2e1248a75ed3fee3",
        ),
        "left_eye": (
            "left_eye.nc",
            "56d5488fb8d3ff08450663ed0136ac659c1d51eb5340a7e3ed52f5ecf019139c",
        ),
        "right_eye": (
            "right_eye.nc",
            "b038c4cb2f6932e4334f135cdf7e24ff9c3b5789977b2ae0206ba80acf54c647",
        ),
        "rosbag": (
            "example.bag",
            "8d27f5e554f5a0e02e0bec59b60424e582f6104380f96c3f226b4d85c107f2bc",
        ),
    }

    def __getitem__(self, item):
        try:
            import pooch
        except ImportError:
            raise ModuleNotFoundError(
                "pooch must be installed to load example data"
            )

        try:
            dataset, known_hash = self.registry[item]
        except KeyError:
            raise KeyError(f"'{item}' is not a valid example dataset")

        return Path(
            pooch.retrieve(url=self.base_url + dataset, known_hash=known_hash)
        )
