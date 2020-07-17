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
    from rigid_body_motion.reference_frames import ReferenceFrame, _registry

    if isinstance(rf, ReferenceFrame):
        return rf
    elif isinstance(rf, str):
        try:
            return _registry[rf]
        except KeyError:
            raise ValueError(f"Frame '{rf}' not found in registry.")
    else:
        raise TypeError(
            f"Expected frame to be str or ReferenceFrame, "
            f"got {type(rf).__name__}"
        )


def _maybe_unpack_dataarray(
    arr, dim=None, axis=None, time_axis=None, timestamps=None
):
    """ If input is DataArray, unpack into data, coords and dims. """
    from rigid_body_motion.utils import is_dataarray

    if not is_dataarray(arr):
        if dim is not None:
            raise ValueError("dim argument specified without DataArray input")
        axis = axis or -1
        time_axis = time_axis or 0
        time_dim = None
        coords = None
        dims = None
        name = None
        attrs = None
    else:
        if dim is not None and axis is not None:
            raise ValueError(
                "You can either specify the dim or the axis argument, not both"
            )
        elif dim is not None:
            axis = arr.dims.index(dim)
        else:
            axis = axis or -1
            dim = str(arr.dims[axis])
        if isinstance(timestamps, str):
            # TODO convert datetimeindex?
            time_axis = arr.dims.index(timestamps)
            time_dim = timestamps
            timestamps = arr[timestamps].data
        elif timestamps is None:
            if arr.ndim > 1:
                time_axis = time_axis or 0
                time_dim = arr.dims[time_axis]
                timestamps = arr.coords[time_dim]
            else:
                time_dim = None
        elif timestamps is False:
            timestamps = None
            time_dim = None
        else:
            raise NotImplementedError(
                "timestamps argument must be dimension name, None or False"
            )
        coords = dict(arr.coords)
        dims = arr.dims
        name = arr.name
        attrs = arr.attrs.copy()
        arr = arr.data

    return (
        arr,
        axis,
        dim,
        time_axis,
        time_dim,
        timestamps,
        coords,
        dims,
        name,
        attrs,
    )


def _make_dataarray(arr, coords, dims, name, attrs, time_dim, ts_out):
    """ Make DataArray out of transformation results. """
    import xarray as xr

    if time_dim is None:
        # no timestamps specified
        if ts_out is not None:
            coords["time"] = ts_out
            dims = ("time",) + dims
    elif isinstance(time_dim, str):
        # timestamps specified as coord
        # TODO transpose if time dim is not first?
        if time_dim not in coords:
            raise ValueError(
                f"{time_dim} is not a coordinate of this DataArray"
            )
        assert ts_out is not None
        if len(coords[time_dim]) != len(ts_out) or np.any(
            coords[time_dim] != ts_out
        ):
            # interpolate if timestamps after transform have changed
            for c in coords:
                if time_dim in coords[c].dims and c != time_dim:
                    coords[c] = coords[c].interp({time_dim: ts_out})
            coords[time_dim] = ts_out
    else:
        # timestamps specified as array
        # TODO time_dim argument
        raise NotImplementedError(
            "timestamps argument must be dimension name or None"
        )

    return xr.DataArray(arr, coords, dims, name, attrs)
