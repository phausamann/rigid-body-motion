""""""
import warnings
from collections import namedtuple

import numpy as np
from quaternion import (
    as_float_array,
    as_quat_array,
    as_rotation_vector,
    derivative,
    quaternion,
    squad,
)
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

Frame = namedtuple(
    "Frame", ("translation", "rotation", "timestamps", "discrete", "inverse"),
)
Array = namedtuple("Array", ("data", "timestamps"))


class TransformMatcher:
    """ Matcher for timestamps from reference frames and arrays. """

    def __init__(self):
        """ Constructor. """
        self.frames = []
        self.arrays = []

    @classmethod
    def _check_timestamps(cls, timestamps, arr_shape):
        """ Make sure timestamps are monotonic. """
        if timestamps is not None:
            if np.any(np.diff(timestamps.astype(float)) < 0):
                raise ValueError("Timestamps must be monotonic")
            if len(timestamps) != arr_shape[0]:
                raise ValueError(
                    "Number of timestamps must match length of first axis "
                    "of array"
                )

    @classmethod
    def _transform_from_frame(cls, frame, timestamps):
        """ Get the transform from a frame resampled to the timestamps. """
        if timestamps is None and frame.timestamps is not None:
            raise ValueError("Cannot convert timestamped to static transform")

        if frame.timestamps is None:
            if timestamps is None:
                translation = frame.translation
                rotation = frame.rotation
            else:
                translation = np.tile(frame.translation, (len(timestamps), 1))
                rotation = np.tile(frame.rotation, (len(timestamps), 1))
        elif frame.discrete:
            translation = np.tile(frame.translation[0], (len(timestamps), 1))
            for t, ts in zip(frame.translation, frame.timestamps):
                translation[timestamps >= ts, :] = t
            rotation = np.tile(frame.rotation[0], (len(timestamps), 1))
            for r, ts in zip(frame.rotation, frame.timestamps):
                rotation[timestamps >= ts, :] = r
        else:
            # TODO method + optional scipy dependency?
            translation = interp1d(
                frame.timestamps.astype(float), frame.translation, axis=0
            )(timestamps.astype(float))
            rotation = as_float_array(
                squad(
                    as_quat_array(frame.rotation),
                    frame.timestamps.astype(float),
                    timestamps.astype(float),
                )
            )

        return translation, rotation

    @classmethod
    def _resample_array(cls, array, timestamps):
        """ Resample an array to the timestamps. """
        if timestamps is None and array.timestamps is not None:
            raise ValueError("Cannot convert timestamped to static array")

        if array.timestamps is None:
            if timestamps is None:
                return array.data
            else:
                return np.tile(array.data, (len(timestamps), 1))
        else:
            # TODO better way to check if quaternion
            if array.data.shape[-1] == 4:
                return as_float_array(
                    squad(
                        as_quat_array(array.data),
                        array.timestamps.astype(float),
                        timestamps.astype(float),
                    )
                )
            else:
                # TODO method + optional scipy dependency?
                return interp1d(
                    array.timestamps.astype(float), array.data, axis=0
                )(timestamps.astype(float))

    def add_reference_frame(self, frame, inverse=False):
        """ Add a reference frame to the matcher.

        Parameters
        ----------
        frame: ReferenceFrame
            The frame to add.

        inverse: bool, default False
            If True, invert the transformation of the reference frame.
        """
        self._check_timestamps(frame.timestamps, frame.translation.shape)
        self.frames.append(
            Frame(
                frame.translation,
                frame.rotation,
                frame.timestamps,
                frame.discrete,
                inverse,
            )
        )

    def add_array(self, array, timestamps=None):
        """ Add an array to the matcher.

        Parameters
        ----------
        array: array_like
            The array to add.

        timestamps: array_like, optional
            If provided, the timestamps of the array.
        """
        self._check_timestamps(timestamps, array.shape)
        self.arrays.append(Array(array, timestamps))

    def get_range(self):
        """ Get the range for which the transformation is defined.

        Returns
        -------
        first: numeric or None
            The first timestamp for which the transformation is defined.

        last: numeric or None
            The last timestamp for which the transformation is defined.
        """
        first_stamps = []
        for frame in self.frames:
            if frame.timestamps is not None and not frame.discrete:
                first_stamps.append(frame.timestamps[0])
        for array in self.arrays:
            if array.timestamps is not None:
                first_stamps.append(array.timestamps[0])

        last_stamps = []
        for frame in self.frames:
            if frame.timestamps is not None and not frame.discrete:
                last_stamps.append(frame.timestamps[-1])
        for array in self.arrays:
            if array.timestamps is not None:
                last_stamps.append(array.timestamps[-1])

        first = np.max(first_stamps) if len(first_stamps) else None
        last = np.min(last_stamps) if len(last_stamps) else None

        return first, last

    def get_timestamps(self, arrays_first=True):
        """ Get the timestamps for which the transformation is defined.

        Parameters
        ----------
        arrays_first: bool, default True
            If True, the first array in the list defines the sampling of the
            timestamps. Otherwise, the first reference frame in the list
            defines the sampling.

        Returns
        -------
        timestamps: array_like
            The timestamps for which the transformation is defined.
        """
        # TODO specify rf name as priority?
        ts_range = self.get_range()

        # first and last timestamp can be None for only discrete transforms
        if ts_range[0] is None:
            ts_range = (-np.inf, ts_range[1])
        elif ts_range[1] is None:
            ts_range = (ts_range[0], np.inf)

        arrays = [
            array for array in self.arrays if array.timestamps is not None
        ]
        discrete_frames = [
            frame
            for frame in self.frames
            if frame.discrete and frame.timestamps is not None
        ]
        continuous_frames = [
            frame
            for frame in self.frames
            if not frame.discrete and frame.timestamps is not None
        ]

        if arrays_first:
            elements = arrays + continuous_frames
        else:
            elements = continuous_frames + arrays

        if len(elements):
            # The first element with timestamps determines the timestamps
            # TODO check if this fails for datetime timestamps
            timestamps = elements[0].timestamps
            timestamps = timestamps[
                (timestamps >= ts_range[0]) & (timestamps <= ts_range[-1])
            ]
        elif len(discrete_frames):
            # If there are no continuous frames or arrays with timestamps
            # we merge together all discrete timestamps
            timestamps = np.concatenate(
                [d.timestamps for d in discrete_frames]
            )
            timestamps = np.unique(timestamps)
        else:
            timestamps = None

        return timestamps

    def get_transformation(self, timestamps=None, arrays_first=True):
        """ Get the transformation across all reference frames.

        Parameters
        ----------
        timestamps: array_like, shape (n_timestamps,), optional
            Timestamps to which the transformation should be matched. If not
            provided the matcher will call `get_timestamps` for the target
            timestamps.

        arrays_first: bool, default True
            If True and timestamps aren't provided, the first array in the
            list defines the sampling of the timestamps. Otherwise, the first
            reference frame in the list defines the sampling.

        Returns
        -------
        translation: array_like, shape (3,) or (n_timestamps, 3)
            The translation across all reference frames.

        rotation: array_like, shape (4,) or (n_timestamps, 4)
            The rotation across all reference frames.

        timestamps: array_like, shape (n_timestamps,) or None
            The timestamps for which the transformation is defined.
        """
        from rigid_body_motion.utils import rotate_vectors

        if timestamps is None:
            timestamps = self.get_timestamps(arrays_first)

        translation = np.zeros(3) if timestamps is None else np.zeros((1, 3))
        rotation = quaternion(1.0, 0.0, 0.0, 0.0)

        for frame in self.frames:
            t, r = self._transform_from_frame(frame, timestamps)
            if frame.inverse:
                translation = rotate_vectors(
                    1 / as_quat_array(r), translation - np.array(t)
                )
                rotation = 1 / as_quat_array(r) * rotation
            else:
                translation = rotate_vectors(
                    as_quat_array(r), translation
                ) + np.array(t)
                rotation = as_quat_array(r) * rotation

        return translation, as_float_array(rotation), timestamps

    def get_arrays(self, timestamps=None, arrays_first=True):
        """ Get re-sampled arrays

        Parameters
        ----------
        timestamps: array_like, shape (n_timestamps,), optional
            Timestamps to which the arrays should be matched. If not provided
            the matcher will call `get_timestamps` for the target timestamps.

        arrays_first: bool, default True
            If True and timestamps aren't provided, the first array in the
            list defines the sampling of the timestamps. Otherwise, the first
            reference frame in the list defines the sampling.

        Returns
        -------
        *arrays: one or more array_like
            Input arrays, matched to the timestamps.

        timestamps: array_like, shape (n_timestamps,) or None
            The timestamps for which the transformation is defined.
        """
        if timestamps is None:
            timestamps = self.get_timestamps(arrays_first)

        arrays = tuple(
            self._resample_array(array, timestamps) for array in self.arrays
        )

        return (*arrays, timestamps)


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


def _replace_dim(coords, dims, axis, into, dimensionality):
    """ Replace the spatial dimension. """
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


def _maybe_unpack_dataarray(
    arr, dim=None, axis=None, time_axis=None, timestamps=None
):
    """ If input is DataArray, unpack into data, coords and dims. """
    from rigid_body_motion.utils import is_dataarray

    ndim = np.asanyarray(arr).ndim

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

    if timestamps is not None and axis % ndim == time_axis % ndim:
        raise ValueError(
            "Spatial and time dimension refer to the same array axis"
        )

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
    what=None,
    return_timestamps=False,
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

    if what is None:
        if method == "transform_vectors":
            what = "representation_frame"
        else:
            what = "reference_frame"

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
    elif return_timestamps or return_timestamps is None and ts_out is not None:
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

    ds[linear_name].attrs.update(
        {
            "representation_frame": frame.name,
            "reference_frame": frame.name,
            "motion_type": linear_name,
            "long_name": linear_name.capitalize(),
            "units": "m",
        }
    )

    ds[angular_name].attrs.update(
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


def _make_velocity_dataarray(
    velocity, motion_type, moving_frame, reference, represent_in, timestamps
):
    """ Create DataArray with linear or angular velocity. """
    import xarray as xr

    if motion_type not in ("linear", "angular"):
        raise ValueError(
            f"motion_type must be 'linear' or 'angular', got {motion_type}"
        )

    da = xr.DataArray(
        velocity,
        coords={"time": timestamps, "cartesian_axis": ["x", "y", "z"]},
        dims=("time", "cartesian_axis"),
        name=f"{motion_type}_velocity",
    )

    da.attrs.update(
        {
            "representation_frame": represent_in.name,
            "reference_frame": reference.name,
            "moving_frame": moving_frame.name,
            "motion_type": f"{motion_type}_velocity",
            "long_name": f"{motion_type.capitalize()} velocity",
            "units": "rad/s" if motion_type == "angular" else "m/s",
        }
    )

    return da


def _estimate_angular_velocity(
    rotation,
    timestamps,
    axis=-1,
    time_axis=0,
    mode="quaternion",
    outlier_thresh=None,
    cutoff=None,
):
    """ Estimate angular velocity of transform. """
    if np.issubdtype(timestamps.dtype, np.datetime64):
        timestamps = timestamps.astype(float) / 1e9

    axis = axis % rotation.ndim
    time_axis = time_axis % rotation.ndim

    # fix time axis if it's the last axis of the array and will be swapped with
    # axis when converting to quaternion dtype
    if time_axis == rotation.ndim - 1:
        time_axis = axis

    r = np.swapaxes(rotation, axis, -1)

    if mode == "quaternion":
        # any NaNs need to be removed because derivative breaks otherwise
        nan_idx = np.any(
            np.isnan(r),
            axis=tuple(a for a in range(r.ndim) if a != time_axis),
        )
        valid_idx = [slice(None)] * r.ndim
        valid_idx[time_axis] = ~nan_idx
        valid_idx = tuple(valid_idx)
        dq = as_quat_array(
            derivative(r[valid_idx], timestamps[~nan_idx], axis=time_axis)
        )
        q = as_quat_array(r[valid_idx])
        angular = np.nan * np.ones_like(r[..., :-1])
        angular[valid_idx] = as_float_array(2 * q.conjugate() * dq)[..., 1:]
    elif mode == "rotation_vector":
        rv = as_rotation_vector(as_quat_array(r))
        angular = np.gradient(rv, timestamps, axis=time_axis)
    else:
        raise ValueError(
            f"'mode' can be 'quaternion' or 'rotation_vector', got {mode}"
        )

    if outlier_thresh is not None:
        dr = np.linalg.norm(
            np.diff(as_rotation_vector(as_quat_array(r)), n=2, axis=time_axis),
            axis=-1,
        )
        dr = np.hstack((dr, 0.0, 0.0)) + np.hstack((0.0, dr, 0.0))
        angular = interp1d(
            timestamps[dr <= outlier_thresh],
            angular[dr <= outlier_thresh],
            axis=time_axis,
            bounds_error=False,
        )(timestamps)

    if cutoff is not None:
        angular = filtfilt(*butter(7, cutoff), angular, axis=time_axis)

    angular = np.swapaxes(angular, axis, -1)

    # TODO transform representation frame to match linear velocity estimate

    return angular


def _estimate_linear_velocity(
    translation, timestamps, time_axis=0, outlier_thresh=None, cutoff=None
):
    """ Estimate linear velocity of transform. """
    if np.issubdtype(timestamps.dtype, np.datetime64):
        timestamps = timestamps.astype(float) / 1e9

    linear = np.gradient(translation, timestamps, axis=time_axis)

    if outlier_thresh is not None:
        dt = np.linalg.norm(np.diff(translation, n=2, axis=time_axis), axis=1)
        dt = np.hstack((dt, 0.0, 0.0)) + np.hstack((0.0, dt, 0.0))
        linear = interp1d(
            timestamps[dt <= outlier_thresh],
            linear[dt <= outlier_thresh],
            axis=time_axis,
            bounds_error=False,
        )(timestamps)

    if cutoff is not None:
        linear = filtfilt(*butter(7, cutoff), linear, axis=time_axis)

    return linear
