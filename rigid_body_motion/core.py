""""""
import numpy as np


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
            raise IndexError("Axis index out of range")
    elif axis < 0 or axis >= ndim:
        raise IndexError("Axis index out of range")

    return axis


def _resolve_rf(rf):
    """ Retrieve frame by name from registry, if applicable. """
    # TODO test
    # TODO raise error if not ReferenceFrame instance?
    from rigid_body_motion.reference_frames import _registry

    if isinstance(rf, str):
        try:
            return _registry[rf]
        except KeyError:
            raise ValueError('Frame "' + rf + '" not found in registry.')
    else:
        return rf


def _maybe_unpack_dataarray(arr, dim=None, axis=None, timestamps=None):
    """ If input is DataArray, unpack into data, coords and dims. """
    from rigid_body_motion.utils import is_dataarray

    if not is_dataarray(arr):
        if dim is not None:
            raise ValueError("dim argument specified without DataArray input.")
        axis = axis or -1
        coords = None
        dims = None
        name = None
        attrs = None
    else:
        if dim is not None and axis is not None:
            raise ValueError(
                "You can either specify the dim or the axis "
                "argument, not both."
            )
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
                "timestamps argument must be dimension name or None."
            )
        coords = dict(arr.coords)
        dims = arr.dims
        name = arr.name
        attrs = arr.attrs
        arr = arr.data

    return arr, axis, timestamps, coords, dims, name, attrs


def _make_dataarray(arr, coords, dims, name, attrs, ts_arg, ts_out):
    """ Make DataArray out of transformation results. """
    import xarray as xr

    if ts_arg is None:
        # no timestamps specified
        if ts_out is not None:
            coords["time"] = ts_out
            dims = ("time",) + dims
    elif isinstance(ts_arg, str):
        # timestamps specified as coord
        # TODO transpose if time dim is not first?
        if ts_arg not in coords:
            raise ValueError(
                "{} is not a coordinate of this DataArray".format(ts_arg)
            )
        assert ts_out is not None
        if len(coords[ts_arg]) != len(ts_out) or np.any(
            coords[ts_arg] != ts_out
        ):
            # interpolate if timestamps after transform have changed
            for c in coords:
                if ts_arg in coords[c].dims and c != ts_arg:
                    coords[c] = coords[c].interp({ts_arg: ts_out})
            coords[ts_arg] = ts_out
    else:
        # timestamps specified as array
        # TODO time_dim argument
        raise NotImplementedError(
            "timestamps argument must be dimension name or None"
        )

    return xr.DataArray(arr, coords, dims, name, attrs)
