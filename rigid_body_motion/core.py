""""""
import warnings

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
                    if np.issubdtype(coords[c].dtype, np.number):
                        coords[c] = coords[c].interp({time_dim: ts_out})
                    else:
                        coords[c] = coords[c].sel(
                            {time_dim: ts_out}, method="nearest"
                        )
            coords[time_dim] = ts_out
    else:
        # timestamps specified as array
        # TODO time_dim argument
        raise NotImplementedError(
            "timestamps argument must be dimension name or None"
        )

    return xr.DataArray(arr, coords, dims, name, attrs)


def _transform(
    method,
    arr,
    into,
    outof,
    dim,
    axis,
    timestamps,
    time_axis,
    what="reference_frame",
    **kwargs,
):
    """ Base transform method. """
    (
        arr,
        axis,
        dim,
        time_axis,
        time_dim,
        ts_in,
        coords,
        dims,
        name,
        attrs,
    ) = _maybe_unpack_dataarray(
        arr, dim=dim, axis=axis, time_axis=time_axis, timestamps=timestamps
    )

    if method is None:
        method_lookup = {
            "position": "transform_points",
            "translation": "transform_points",
            "orientation": "transform_quaternions",
            "rotation": "transform_quaternions",
        }
        try:
            # TODO warn if method doesn't match attrs["motion_type"]
            method = method_lookup[attrs["motion_type"]]
        except (KeyError, TypeError):
            raise ValueError(
                f"'method' must be specified unless you provide a DataArray "
                f"whose ``attrs`` contain a 'motion_type' entry "
                f"containing any of {method_lookup.keys()}"
            )

    if outof is None:
        if attrs is not None and what in attrs:
            outof = _resolve_rf(attrs[what])
        else:
            raise ValueError(
                f"'outof' must be specified unless you provide a DataArray "
                f"whose ``attrs`` contain a '{what}' entry with "
                f"the name of a registered frame"
            )
    else:
        outof = _resolve_rf(outof)
        if attrs is not None and what in attrs and attrs[what] != outof.name:
            warnings.warn(
                f"You are transforming the '{what}' of the array out of "
                f"{outof.name}, but the current '{what}' the array is "
                f"{attrs[what]}"
            )

    into = _resolve_rf(into)

    if method in ("transform_angular_velocity", "transform_linear_velocity"):
        kwargs["what"] = what

    if attrs is not None:
        attrs[what] = into.name
        attrs["representation_frame"] = into.name

    arr, ts_out = getattr(outof, method)(
        arr,
        into,
        axis=axis,
        timestamps=ts_in,
        time_axis=time_axis,
        return_timestamps=True,
        **kwargs,
    )

    if coords is not None:
        return _make_dataarray(
            arr, coords, dims, name, attrs, time_dim, ts_out
        )
    elif ts_out is not None:
        # TODO not so pretty. Maybe also introduce return_timestamps
        #  parameter and do this when return_timestamps=None
        return arr, ts_out
    else:
        return arr


def _make_transform_or_pose_dataset(
    translation, rotation, frame, timestamps, pose=False
):
    """ Create Dataset with translation and rotation. """
    import xarray as xr

    if pose:
        linear_name = "position"
        angular_name = "orientation"
    else:
        linear_name = "translation"
        angular_name = "rotation"

    if timestamps is not None:
        ds = xr.Dataset(
            {
                linear_name: (["time", "cartesian_axis"], translation),
                angular_name: (["time", "quaternion_axis"], rotation),
            },
            {
                "time": timestamps,
                "cartesian_axis": ["x", "y", "z"],
                "quaternion_axis": ["w", "x", "y", "z"],
            },
        )
    else:
        ds = xr.Dataset(
            {
                linear_name: ("cartesian_axis", translation),
                angular_name: ("quaternion_axis", rotation),
            },
            {
                "cartesian_axis": ["x", "y", "z"],
                "quaternion_axis": ["w", "x", "y", "z"],
            },
        )

    ds.translation.attrs.update(
        {
            "representation_frame": frame.name,
            "reference_frame": frame.name,
            "motion_type": linear_name,
            "long_name": linear_name.capitalize(),
            "units": "m",
        }
    )

    ds.rotation.attrs.update(
        {
            "representation_frame": frame.name,
            "reference_frame": frame.name,
            "motion_type": angular_name,
            "long_name": angular_name.capitalize(),
        }
    )

    return ds


def _make_twist_dataset(
    angular, linear, moving_frame, reference, represent_in, timestamps
):
    """ Create Dataset with linear and angular velocity. """
    import xarray as xr

    twist = xr.Dataset(
        {
            "angular_velocity": (["time", "cartesian_axis"], angular),
            "linear_velocity": (["time", "cartesian_axis"], linear),
        },
        {"time": timestamps, "cartesian_axis": ["x", "y", "z"]},
    )

    twist.angular_velocity.attrs.update(
        {
            "representation_frame": represent_in.name,
            "reference_frame": reference.name,
            "moving_frame": moving_frame.name,
            "motion_type": "angular_velocity",
            "long_name": "Angular velocity",
            "units": "rad/s",
        }
    )

    twist.linear_velocity.attrs.update(
        {
            "representation_frame": represent_in.name,
            "reference_frame": reference.name,
            "moving_frame": moving_frame.name,
            "motion_type": "linear_velocity",
            "long_name": "Linear velocity",
            "units": "m/s",
        }
    )

    return twist
