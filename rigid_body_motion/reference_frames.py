""""""
import numpy as np
from anytree import NodeMixin, Walker
from quaternion import (
    as_float_array,
    as_quat_array,
    derivative,
    from_rotation_matrix,
    squad,
)
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from rigid_body_motion.core import _resolve_rf
from rigid_body_motion.utils import qinv, rotate_vectors

_registry = {}


def _register(rf, update=False):
    """ Register a reference frame. """
    if rf.name is None:
        raise ValueError("Reference frame name cannot be None.")
    if rf.name in _registry:
        if update:
            # TODO keep children?
            _registry[rf.name].parent = None
        else:
            raise ValueError(
                f"Reference frame with name {rf.name} is already registered. "
                f"Specify update=True to overwrite."
            )
    # TODO check if name is a cs transform?
    _registry[rf.name] = rf


def _deregister(name):
    """ Deregister a reference frame. """
    if name not in _registry:
        raise ValueError(
            "Reference frame with name " + name + " not found in registry"
        )

    _registry.pop(name)


def register_frame(
    name,
    parent=None,
    translation=None,
    rotation=None,
    timestamps=None,
    inverse=False,
    update=False,
):
    """ Register a new reference frame in the registry.

    Parameters
    ----------
    name: str
        The name of the reference frame.

    parent: str or ReferenceFrame, optional
        The parent reference frame. If str, the frame will be looked up
        in the registry under that name. If not specified, this frame
        will be a root node of a new reference frame tree.

    translation: array_like, optional
        The translation of this frame wrt the parent frame. Not
        applicable if there is no parent frame.

    rotation: array_like, optional
        The rotation of this frame wrt the parent frame. Not
        applicable if there is no parent frame.

    timestamps: array_like, optional
        The timestamps for translation and rotation of this frame. Not
        applicable if this is a static reference frame.

    inverse: bool, default False
        If True, invert the transform wrt the parent frame, i.e. the
        translation and rotation are specified for the parent frame wrt this
        frame.

    update: bool, default False
        If True, overwrite if there is a frame with the same name in the
        registry.
    """
    # TODO make this a class with __call__, from_dataset etc. methods?
    rf = ReferenceFrame(
        name,
        parent=parent,
        translation=translation,
        rotation=rotation,
        timestamps=timestamps,
        inverse=inverse,
    )
    _register(rf, update=update)


def deregister_frame(name):
    """ Remove a reference frame from the registry.

    Parameters
    ----------
    name: str
        The name of the reference frame.
    """
    _deregister(name)


def clear_registry():
    """ Clear the reference frame registry. """
    _registry.clear()


class ReferenceFrame(NodeMixin):
    """ A three-dimensional reference frame. """

    def __init__(
        self,
        name=None,
        parent=None,
        translation=None,
        rotation=None,
        timestamps=None,
        inverse=False,
    ):
        """ Constructor.

        Parameters
        ----------
        name: str, optional
            The name of this reference frame.

        parent: str or ReferenceFrame, optional
            The parent reference frame. If str, the frame will be looked up
            in the registry under that name. If not specified, this frame
            will be a root node of a new reference frame tree.

        translation: array_like, optional
            The translation of this frame wrt the parent frame. Not
            applicable if there is no parent frame.

        rotation: array_like, optional
            The rotation of this frame wrt the parent frame. Not
            applicable if there is no parent frame.

        timestamps : array_like, optional
            The timestamps for translation and rotation of this frame. Not
            applicable if this is a static reference frame.

        inverse: bool, default False
            If True, invert the transform wrt the parent frame, i.e. the
            translation and rotation are specified for the parent frame wrt
            this frame.
        """
        super(ReferenceFrame, self).__init__()

        # TODO check name requirement
        self.name = name

        if parent is not None:
            self.parent = _resolve_rf(parent)
            (
                self.translation,
                self.rotation,
                self.timestamps,
            ) = self._init_arrays(translation, rotation, timestamps, inverse)
        else:
            self.parent = None
            self._verify_root(translation, rotation, timestamps)
            self.translation, self.rotation, self.timestamps = None, None, None

    def __del__(self):
        """ Destructor. """
        if self.name in _registry and _registry[self.name] is self:
            _deregister(self.name)

    def __str__(self):
        """ String representation. """
        return f"<ReferenceFrame '{self.name}'>"

    def __repr__(self):
        """ String representation. """
        return self.__str__()

    @staticmethod
    def _init_arrays(translation, rotation, timestamps, inverse):
        """ Initialize translation, rotation and timestamp arrays. """
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
            if timestamps.ndim != 1:
                raise ValueError("timestamps must be one-dimensional.")
            t_shape = (len(timestamps), 3)
            r_shape = (len(timestamps), 4)
        else:
            t_shape = (3,)
            r_shape = (4,)

        if translation is not None:
            translation = np.asarray(translation)
            if translation.shape != t_shape:
                raise ValueError(
                    f"Expected translation to be of shape {t_shape}, got "
                    f"{translation.shape}"
                )
        else:
            translation = np.zeros(t_shape)

        if rotation is not None:
            rotation = np.asarray(rotation)
            if rotation.shape != r_shape:
                raise ValueError(
                    f"Expected rotation to be of shape {r_shape}, got "
                    f"{rotation.shape}"
                )
        else:
            rotation = np.zeros(r_shape)
            rotation[..., 0] = 1.0

        if inverse:
            rotation = qinv(rotation)
            translation = -rotate_vectors(rotation, translation)

        return translation, rotation, timestamps

    @staticmethod
    def _verify_root(translation, rotation, timestamps):
        """ Verify arguments for root node. """
        # TODO test
        if translation is not None:
            raise ValueError("translation specified without parent frame.")
        if rotation is not None:
            raise ValueError("rotation specified without parent frame.")
        if timestamps is not None:
            raise ValueError("timestamps specified without parent frame.")

    @classmethod
    def _broadcast(cls, arr, timestamps):
        """ Broadcast scalar array along timestamp axis. """
        # TODO test
        return np.tile(arr, (len(timestamps), 1))

    @classmethod
    def _interpolate(cls, source_arr, target_arr, source_ts, target_ts):
        """ Interpolate source array at target array timestamps. """
        # TODO specify time_axis as parameter
        # TODO priority=None/<rf_name>
        # TODO method + optional scipy dependency?
        ts_dtype = target_ts.dtype
        source_ts = source_ts.astype(float)
        target_ts = target_ts.astype(float)

        # TODO sort somewhere and turn these into assertions or use min/max
        #  with boolean indexing
        if np.any(np.diff(source_ts) < 0):
            raise ValueError("source_ts is not sorted.")
        if np.any(np.diff(target_ts) < 0):
            raise ValueError("target_ts is not sorted.")

        # TODO raise error when intersection is empty
        if target_ts[0] < source_ts[0]:
            target_arr = target_arr[target_ts >= source_ts[0]]
            target_ts = target_ts[target_ts >= source_ts[0]]
        if target_ts[-1] > source_ts[-1]:
            target_arr = target_arr[target_ts <= source_ts[-1]]
            target_ts = target_ts[target_ts <= source_ts[-1]]

        if source_arr.shape[1] == 7:
            # ugly edge case of t and r stacked
            source_arr_interp = np.hstack(
                (
                    interp1d(source_ts, source_arr[:, :3], axis=0)(target_ts),
                    as_float_array(
                        squad(
                            as_quat_array(source_arr[:, 3:]),
                            source_ts,
                            target_ts,
                        )
                    ),
                )
            )
        elif source_arr.shape[1] == 4:
            source_arr_interp = as_float_array(
                squad(as_quat_array(source_arr), source_ts, target_ts)
            )
        else:
            source_arr_interp = interp1d(source_ts, source_arr, axis=0)(
                target_ts
            )

        return source_arr_interp, target_arr, target_ts.astype(ts_dtype)

    @classmethod
    def _match_timestamps(cls, arr, arr_ts, rf_t, rf_r, rf_ts):
        """ Match timestamps of array and reference frame. """
        # TODO test
        # TODO policy='from_arr'/'from_rf'
        if rf_ts is None:
            return arr, rf_t, rf_r, arr_ts
        elif arr_ts is None:
            return cls._broadcast(arr, rf_ts), rf_t, rf_r, rf_ts
        elif len(arr_ts) != len(rf_ts) or np.any(arr_ts != rf_ts):
            # abuse interpolate by stacking t and r and splitting afterwards
            rf_tr = np.hstack((rf_t, rf_r))
            arr, rf_tr, rf_ts = cls._interpolate(arr, rf_tr, arr_ts, rf_ts)
            rf_t, rf_r = np.hsplit(rf_tr, [3])
            return arr, rf_t, rf_r, rf_ts
        else:
            return arr, rf_t, rf_r, rf_ts

    @classmethod
    def _match_timestamps_multi(cls, arr_list, ts_list):
        """ Match multiple arrays and timestamps at once. """
        earliest = np.max([ts[0] for ts in ts_list if ts is not None])
        latest = np.min([ts[-1] for ts in ts_list if ts is not None])

        return_ts = None
        for ts in ts_list:
            if return_ts is None and ts is not None:
                return_ts = ts[(ts >= earliest) & (ts <= latest)]
                break
        else:
            return arr_list, None

        return_arr_list = []
        for arr, ts in zip(arr_list, ts_list):
            if ts is None:
                return_arr_list.append(cls._broadcast(arr, return_ts))
            else:
                if arr.shape[1] == 4:
                    arr_t = as_float_array(
                        squad(arr, ts.astype(float), return_ts.astype(float))
                    )
                else:
                    arr_t = interp1d(ts.astype(float), arr, axis=0)(
                        return_ts.astype(float)
                    )
                return_arr_list.append(arr_t)

        return return_arr_list, return_ts

    @classmethod
    def _add_transformation(cls, rf, t, r, ts, inverse=False):
        """ Add transformation of this frame to current transformation. """
        # TODO test
        if rf.timestamps is not None:
            if ts is None:
                translation = rf.translation
                rotation = rf.rotation
                t = cls._broadcast(t, rf.timestamps)
                r = cls._broadcast(r, rf.timestamps)
                ts_new = rf.timestamps
            else:
                translation, t, ts_new = cls._interpolate(
                    rf.translation, t, rf.timestamps, ts
                )
                rotation, r, ts_new = cls._interpolate(
                    rf.rotation, r, rf.timestamps, ts
                )
        else:
            translation = rf.translation
            rotation = rf.rotation
            ts_new = ts

        if inverse:
            q = 1 / as_quat_array(rotation)
            dt = -np.array(translation)
            t = rotate_vectors(q, t + dt)
        else:
            q = as_quat_array(rotation)
            dt = np.array(translation)
            t = rotate_vectors(q, t) + dt

        return t, as_float_array(q * as_quat_array(r)), ts_new

    @classmethod
    def _validate_input(cls, arr, axis, n_axis, timestamps, time_axis):
        """ Validate shape of array and timestamps. """
        # TODO process DataArray (dim=str, timestamps=str)
        arr = np.asarray(arr)

        if arr.shape[axis] != n_axis:
            raise ValueError(
                f"Expected array to have length {n_axis} along axis {axis}, "
                f"got {arr.shape[axis]}"
            )

        if timestamps is not None:
            timestamps = np.asarray(timestamps)
            if timestamps.ndim != 1:
                raise ValueError("timestamps must be one-dimensional")
            if arr.shape[time_axis] != len(timestamps):
                raise ValueError(
                    f"Axis {time_axis} of the array must have the same length "
                    f"as the timestamps"
                )
            arr = np.swapaxes(arr, 0, time_axis)

        return arr, timestamps

    @classmethod
    def _expand_singleton_axes(cls, t_or_r, ndim):
        """ Expand singleton axes for correct broadcasting with array. """
        if t_or_r.ndim > 1:
            for _ in range(ndim - 2):
                t_or_r = np.expand_dims(t_or_r, 1)

        return t_or_r

    def _walk(self, to_rf):
        """ Walk from this frame to a target frame along the tree. """
        to_rf = _resolve_rf(to_rf)
        walker = Walker()
        up, _, down = walker.walk(self, to_rf)
        return up, down

    @classmethod
    def from_dataset(
        cls,
        ds,
        translation,
        rotation,
        timestamps,
        parent,
        name=None,
        inverse=False,
    ):
        """ Construct a reference frame from a Dataset.

        Parameters
        ----------
        ds: xarray Dataset
            The dataset from which to construct the reference frame.

        translation: str
            The name of the variable representing the translation
            wrt the parent frame.

        rotation: str
            The name of the variable representing the rotation
            wrt the parent frame.

        timestamps: str
            The name of the variable or coordinate representing the
            timestamps.

        parent: str or ReferenceFrame
            The parent reference frame. If str, the frame will be looked up
            in the registry under that name.

        name: str, default None
            The name of the reference frame.

        inverse: bool, default False
            If True, invert the transform wrt the parent frame, i.e. the
            translation and rotation are specified for the parent frame wrt
            this frame.

        Returns
        -------
        rf: ReferenceFrame
            The constructed reference frame.
        """
        # TODO raise errors here if dimensions etc. don't match
        return cls(
            name,
            parent,
            ds[translation].data,
            ds[rotation].data,
            ds[timestamps].data,
            inverse=inverse,
        )

    @classmethod
    def from_translation_dataarray(
        cls, da, timestamps, parent, name=None, inverse=False
    ):
        """ Construct a reference frame from a translation DataArray.

        Parameters
        ----------
        da: xarray DataArray
            The array that describes the translation of this frame
            wrt the parent frame.

        timestamps: str
            The name of the variable or coordinate representing the
            timestamps.

        parent: str or ReferenceFrame
            The parent reference frame. If str, the frame will be looked up
            in the registry under that name.

        name: str, default None
            The name of the reference frame.

        inverse: bool, default False
            If True, invert the transform wrt the parent frame, i.e. the
            translation is specified for the parent frame wrt this frame.

        Returns
        -------
        rf: ReferenceFrame
            The constructed reference frame.
        """
        # TODO raise errors here if dimensions etc. don't match
        return cls(
            name,
            parent,
            translation=da.data,
            timestamps=da[timestamps].data,
            inverse=inverse,
        )

    @classmethod
    def from_rotation_dataarray(
        cls, da, timestamps, parent, name=None, inverse=False
    ):
        """ Construct a reference frame from a rotation DataArray.

        Parameters
        ----------
        da: xarray DataArray
            The array that describes the rotation of this frame
            wrt the parent frame.

        timestamps: str
            The name of the variable or coordinate representing the
            timestamps.

        parent: str or ReferenceFrame
            The parent reference frame. If str, the frame will be looked up
            in the registry under that name.

        name: str, default None
            The name of the reference frame.

        inverse: bool, default False
            If True, invert the transform wrt the parent frame, i.e. the
            rotation is specified for the parent frame wrt this frame.

        Returns
        -------
        rf: ReferenceFrame
            The constructed reference frame.
        """
        # TODO raise errors here if dimensions etc. don't match
        return cls(
            name,
            parent,
            rotation=da.data,
            timestamps=da[timestamps].data,
            inverse=inverse,
        )

    @classmethod
    def from_rotation_matrix(cls, mat, parent, name=None, inverse=False):
        """ Construct a static reference frame from a rotation matrix.

        Parameters
        ----------
        mat: array_like, shape (3, 3)
            The rotation matrix that describes the rotation of this frame
            wrt the parent frame.

        parent: str or ReferenceFrame
            The parent reference frame. If str, the frame will be looked up
            in the registry under that name.

        name: str, default None
            The name of the reference frame.

        inverse: bool, default False
            If True, invert the transform wrt the parent frame, i.e. the
            rotation is specified for the parent frame wrt this frame.

        Returns
        -------
        rf: ReferenceFrame
            The constructed reference frame.
        """
        # TODO support moving reference frame
        if mat.shape != (3, 3):
            raise ValueError(
                f"Expected mat to have shape (3, 3), got {mat.shape}"
            )

        return cls(
            name,
            parent,
            rotation=as_float_array(from_rotation_matrix(mat)),
            inverse=inverse,
        )

    def get_transformation(self, to_frame):
        """ Calculate the transformation from this frame to another.

        The transformation is a rotation followed by a translation which,
        when applied to a position and/or orientation represented in this
        reference frame, yields the representation of that
        position/orientation in the target reference frame.

        Parameters
        ----------
        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        Returns
        -------
        t: array_like, shape (3,) or (n_timestamps, 3)
            The translation from this frame to the target frame.

        r: array_like, shape (4,) or (n_timestamps, 4)
            The rotation from this frame to the target frame.

        ts: array_like, shape (n_timestamps,) or None
            The timestamps for which the transformation is defined.
        """
        up, down = self._walk(to_frame)

        t = np.zeros(3)
        r = np.array((1.0, 0.0, 0.0, 0.0))
        ts = None

        for rf in up:
            t, r, ts = self._add_transformation(rf, t, r, ts)

        for rf in down:
            t, r, ts = self._add_transformation(rf, t, r, ts, inverse=True)

        return t, r, ts

    def transform_vectors(
        self,
        arr,
        to_frame,
        axis=-1,
        time_axis=0,
        timestamps=None,
        return_timestamps=False,
    ):
        """ Transform array of vectors from this frame to another.

        Parameters
        ----------
        arr: array_like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the spatial coordinates of the
            vectors.

        time_axis: int, default 0
            The axis of the array representing the timestamps of the vectors.

        timestamps: array_like, optional
            The timestamps of the vectors, corresponding to the `time_axis`
            of the array. If not None, the axis defined by `time_axis` will be
            re-sampled to the timestamps for which the transformation is
            defined.

        return_timestamps: bool, default False
            If True, also return the timestamps after the transformation.

        Returns
        -------
        arr_transformed: array_like
            The transformed array.

        ts: array_like, shape (n_timestamps,) or None
            The timestamps after the transformation.
        """
        arr, arr_ts = self._validate_input(arr, axis, 3, timestamps, time_axis)

        t, r, rf_ts = self.get_transformation(to_frame)

        arr, _, r, ts = self._match_timestamps(arr, arr_ts, t, r, rf_ts)
        r = self._expand_singleton_axes(r, arr.ndim)
        arr = rotate_vectors(r, arr, axis=axis)

        # undo time axis swap
        if time_axis is not None:
            arr = np.swapaxes(arr, 0, time_axis)

        if not return_timestamps:
            return arr
        else:
            return arr, ts

    def transform_points(
        self,
        arr,
        to_frame,
        axis=-1,
        time_axis=0,
        timestamps=None,
        return_timestamps=False,
    ):
        """ Transform array of points from this frame to another.

        Parameters
        ----------
        arr: array_like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the spatial coordinates of the
            points.

        time_axis: int, default 0
            The axis of the array representing the timestamps of the points.

        timestamps: array_like, optional
            The timestamps of the vectors, corresponding to the `time_axis`
            of the array. If not None, the axis defined by `time_axis` will be
            re-sampled to the timestamps for which the transformation is
            defined.

        return_timestamps: bool, default False
            If True, also return the timestamps after the transformation.

        Returns
        -------
        arr_transformed: array_like
            The transformed array.

        ts: array_like, shape (n_timestamps,) or None
            The timestamps after the transformation.
        """
        arr, arr_ts = self._validate_input(arr, axis, 3, timestamps, time_axis)

        t, r, rf_ts = self.get_transformation(to_frame)

        arr, t, r, ts = self._match_timestamps(arr, arr_ts, t, r, rf_ts)
        t = self._expand_singleton_axes(t, arr.ndim)
        r = self._expand_singleton_axes(r, arr.ndim)
        arr = rotate_vectors(r, arr, axis=axis)
        arr = arr + np.array(t)

        # undo time axis swap
        if time_axis is not None:
            arr = np.swapaxes(arr, 0, time_axis)

        if not return_timestamps:
            return arr
        else:
            return arr, ts

    def transform_quaternions(
        self,
        arr,
        to_frame,
        axis=-1,
        time_axis=0,
        timestamps=None,
        return_timestamps=False,
    ):
        """ Transform array of quaternions from this frame to another.

        Parameters
        ----------
        arr: array_like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the spatial coordinates of the
            quaternions.

        time_axis: int, default 0
            The axis of the array representing the timestamps of the
            quaternions.

        timestamps: array_like, optional
            The timestamps of the quaternions, corresponding to the `time_axis`
            of the array. If not None, the axis defined by `time_axis` will be
            re-sampled to the timestamps for which the transformation is
            defined.

        return_timestamps: bool, default False
            If True, also return the timestamps after the transformation.

        Returns
        -------
        arr_transformed: array_like
            The transformed array.

        ts: array_like, shape (n_timestamps,) or None
            The timestamps after the transformation.
        """
        arr, arr_ts = self._validate_input(arr, axis, 4, timestamps, time_axis)

        t, r, rf_ts = self.get_transformation(to_frame)

        arr, _, r, ts = self._match_timestamps(arr, arr_ts, t, r, rf_ts)
        r = self._expand_singleton_axes(r, arr.ndim)
        arr = np.swapaxes(arr, axis, -1)
        arr = as_quat_array(r) * as_quat_array(arr)
        arr = np.swapaxes(as_float_array(arr), -1, axis)

        # undo time axis swap
        if time_axis is not None:
            arr = np.swapaxes(arr, 0, time_axis)

        if not return_timestamps:
            return arr
        else:
            return arr, ts

    def transform_angular_velocity(
        self,
        arr,
        to_frame,
        what="reference_frame",
        axis=-1,
        time_axis=0,
        timestamps=None,
        return_timestamps=False,
        cutoff=None,
    ):
        """ Transform array of angular velocities from this frame to another.

        The array represents the velocity of a moving body or frame wrt a
        reference frame, expressed in a representation frame.

        The transformation changes either the reference frame, the moving
        frame or the representation frame of the velocity from this frame to
        another. In either case, it is assumed that the array is represented in
        the frame that is being changed and will be represented in the new
        frame after the transformation.

        Parameters
        ----------
        arr: array_like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        what: str, default "reference_frame"
            What frame of the velocity to transform. Can be "reference_frame",
            "moving_frame" or "representation_frame".

        axis: int, default -1
            The axis of the array representing the spatial coordinates of the
            velocities.

        time_axis: int, default 0
            The axis of the array representing the timestamps of the
            velocities.

        timestamps: array_like, optional
            The timestamps of the velocities, corresponding to the `time_axis`
            of the array. If not None, the axis defined by `time_axis` will be
            re-sampled to the timestamps for which the transformation is
            defined.

        return_timestamps: bool, default False
            If True, also return the timestamps after the transformation.

        cutoff: float, optional
            Frequency of a low-pass filter applied to linear and angular
            velocity after the twist estimation as a fraction of the Nyquist
            frequency.

        Returns
        -------
        arr_transformed: array_like
            The transformed array.

        ts: array_like, shape (n_timestamps,) or None
            The timestamps after the transformation.
        """
        if what == "reference_frame":
            _, angular, angular_ts = self.lookup_twist(
                to_frame, to_frame, cutoff=cutoff
            )
        elif what == "moving_frame":
            _, angular, angular_ts = _resolve_rf(to_frame).lookup_twist(
                self, to_frame, cutoff=cutoff
            )
        elif what == "representation_frame":
            return self.transform_vectors(
                arr,
                to_frame,
                axis=axis,
                time_axis=time_axis,
                timestamps=timestamps,
                return_timestamps=return_timestamps,
            )
        else:
            raise ValueError(
                f"Expected 'what' to be 'reference_frame' or 'moving_frame', "
                f"got {what}"
            )

        arr, ts = self.transform_vectors(
            arr,
            to_frame,
            axis=axis,
            time_axis=time_axis,
            timestamps=timestamps,
            return_timestamps=True,
        )

        angular, arr, timestamps = self._interpolate(
            angular, arr, angular_ts, ts
        )

        arr += angular

        if return_timestamps:
            return arr, timestamps
        else:
            return arr

    def transform_linear_velocity(
        self,
        arr,
        to_frame,
        what="reference_frame",
        moving_frame=None,
        reference_frame=None,
        axis=-1,
        time_axis=0,
        timestamps=None,
        return_timestamps=False,
        outlier_thresh=None,
        cutoff=None,
    ):
        """ Transform array of linear velocities from this frame to another.

        The array represents the velocity of a moving body or frame wrt a
        reference frame, expressed in a representation frame.

        The transformation changes either the reference frame, the moving
        frame or the representation frame of the velocity from this frame to
        another. In either case, it is assumed that the array is represented in
        the frame that is being changed and will be represented in the new
        frame after the transformation.

        Parameters
        ----------
        arr: array_like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        what: str, default "reference_frame"
            What frame of the velocity to transform. Can be "reference_frame",
            "moving_frame" or "representation_frame".

        moving_frame: str or ReferenceFrame, optional
            The moving frame when transforming the reference frame of the
            velocity.

        reference_frame: str or ReferenceFrame, optional
            The reference frame when transforming the moving frame of the
            velocity.

        axis: int, default -1
            The axis of the array representing the spatial coordinates of the
            velocities.

        time_axis: int, default 0
            The axis of the array representing the timestamps of the
            velocities.

        timestamps: array_like, optional
            The timestamps of the velocities, corresponding to the `time_axis`
            of the array. If not None, the axis defined by `time_axis` will be
            re-sampled to the timestamps for which the transformation is
            defined.

        return_timestamps: bool, default False
            If True, also return the timestamps after the transformation.

        cutoff: float, optional
            Frequency of a low-pass filter applied to linear and angular
            velocity after the twist estimation as a fraction of the Nyquist
            frequency.

        outlier_thresh: float, optional
            Some SLAM-based trackers introduce position corrections when a new
            camera frame becomes available. This introduces outliers in the
            linear velocity estimate. The estimation algorithm used here
            can suppress these outliers by throwing out samples where the
            norm of the second-order differences of the position is above
            `outlier_thresh` and interpolating the missing values. For
            measurements from the Intel RealSense T265 tracker, set this value
            to 1e-3.

        Returns
        -------
        arr_transformed: array_like
            The transformed array.

        ts: array_like, shape (n_timestamps,) or None
            The timestamps after the transformation.
        """
        if what == "reference_frame":
            linear, angular, linear_ts = self.lookup_twist(
                to_frame,
                to_frame,
                cutoff=cutoff,
                outlier_thresh=outlier_thresh,
            )
            angular_ts = linear_ts
            translation, _, translation_ts = _resolve_rf(
                moving_frame
            ).get_transformation(self)

        elif what == "moving_frame":
            to_frame = _resolve_rf(to_frame)
            linear, _, linear_ts = to_frame.lookup_twist(
                self, to_frame, cutoff=cutoff, outlier_thresh=outlier_thresh,
            )
            _, angular, angular_ts = self.lookup_twist(
                reference_frame, to_frame, cutoff=cutoff
            )
            translation, _, translation_ts = to_frame.get_transformation(self)

        elif what == "representation_frame":
            return self.transform_vectors(
                arr,
                to_frame,
                axis=axis,
                time_axis=time_axis,
                timestamps=timestamps,
                return_timestamps=return_timestamps,
            )

        else:
            raise ValueError(
                f"Expected 'what' to be 'reference_frame' or 'moving_frame', "
                f"got {what}"
            )

        arr, ts = self.transform_vectors(
            arr,
            to_frame,
            axis=axis,
            time_axis=time_axis,
            timestamps=timestamps,
            return_timestamps=True,
        )
        translation, translation_ts = self.transform_vectors(
            translation,
            to_frame,
            timestamps=translation_ts,
            return_timestamps=True,
        )

        (arr, linear, angular, translation), ts = self._match_timestamps_multi(
            [arr, linear, angular, translation],
            [ts, linear_ts, angular_ts, translation_ts],
        )

        arr = arr + linear + np.cross(angular, translation)

        if return_timestamps:
            return arr, ts
        else:
            return arr

    @classmethod
    def _estimate_linear_velocity(
        cls, translation, timestamps, outlier_thresh=None, cutoff=None
    ):
        """ Estimate linear velocity of transform. """
        timestamps = timestamps.astype(float) / 1e9

        linear = np.gradient(translation, timestamps, axis=0)

        if outlier_thresh is not None:
            dt = np.linalg.norm(np.diff(translation, n=2, axis=0), axis=1)
            dt = np.hstack((dt, 0.0, 0.0)) + np.hstack((0.0, dt, 0.0))
            linear = interp1d(
                timestamps[dt <= outlier_thresh],
                linear[dt <= outlier_thresh],
                axis=0,
                bounds_error=False,
            )(timestamps)

        if cutoff is not None:
            linear = filtfilt(*butter(7, cutoff), linear, axis=0)

        return linear

    @classmethod
    def _estimate_angular_velocity(cls, rotation, timestamps, cutoff=None):
        """ Estimate angular velocity of transform. """
        timestamps = timestamps.astype(float) / 1e9

        dq = derivative(rotation, timestamps)
        angular = as_float_array(
            2 * as_quat_array(rotation).conjugate() * as_quat_array(dq)
        )[:, 1:]

        if cutoff is not None:
            angular = filtfilt(*butter(7, cutoff), angular, axis=0)

        return angular

    def lookup_twist(
        self,
        reference=None,
        represent_in=None,
        outlier_thresh=None,
        cutoff=None,
    ):
        """ Estimate linear and angular velocity of this frame wrt a reference.

        Parameters
        ----------
        reference: str or ReferenceFrame, optional
            The reference frame wrt which the twist is estimated. Defaults to
            the parent frame.

        represent_in: str or ReferenceFrame, optional
            The reference frame in which the twist is represented. Defaults
            to the parent frame.

        outlier_thresh: float, optional
            Some SLAM-based trackers introduce position corrections when a new
            camera frame becomes available. This introduces outliers in the
            linear velocity estimate. The estimation algorithm used here
            can suppress these outliers by throwing out samples where the
            norm of the second-order differences of the position is above
            `outlier_thresh` and interpolating the missing values. For
            measurements from the Intel RealSense T265 tracker, set this value
            to 1e-3.

        cutoff: float, optional
            Frequency of a low-pass filter applied to linear and angular
            velocity after the estimation as a fraction of the Nyquist
            frequency.

        Returns
        -------
        linear: numpy.ndarray, shape (N, 3)
            Linear velocity of moving frame wrt reference frame, represented
            in representation frame.

        angular: numpy.ndarray, shape (N, 3)
            Angular velocity of moving frame wrt reference frame, represented
            in representation frame.

        timestamps: each numpy.ndarray
            Timestamps of the twist.
        """
        try:
            reference = _resolve_rf(reference or self.parent)
            represent_in = _resolve_rf(represent_in or self.parent)
        except TypeError:
            raise ValueError(f"Frame {self.name} has no parent frame")

        translation, rotation, timestamps = self.get_transformation(reference)

        if timestamps is None:
            raise ValueError("Twist cannot be estimated for static transforms")

        linear = self._estimate_linear_velocity(
            translation, timestamps, outlier_thresh, cutoff
        )
        angular = self._estimate_angular_velocity(rotation, timestamps, cutoff)

        linear, linear_ts = reference.transform_vectors(
            linear, represent_in, timestamps=timestamps, return_timestamps=True
        )
        angular, angular_ts = self.transform_vectors(
            angular,
            represent_in,
            timestamps=timestamps,
            return_timestamps=True,
        )
        angular, linear, timestamps = self._interpolate(
            angular, linear, angular_ts, linear_ts
        )

        return linear, angular, timestamps

    def register(self, update=False):
        """ Register this frame in the registry.

        Parameters
        ----------
        update: bool, default False
            If True, overwrite if there is a frame with the same name in the
            registry.
        """
        _register(self, update=update)

    def deregister(self):
        """ Remove this frame from the registry. """
        _deregister(self.name)
