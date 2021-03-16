""""""
import numpy as np
from anytree import NodeMixin, RenderTree, Walker
from quaternion import as_float_array, as_quat_array, from_rotation_matrix

from rigid_body_motion.core import (
    TransformMatcher,
    _estimate_angular_velocity,
    _estimate_linear_velocity,
    _resolve_rf,
)
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


def render_tree(root):
    """ Render a reference frame tree.

    Parameters
    ----------
    root: str or ReferenceFrame
        The root of the rendered tree.
    """
    for pre, _, node in RenderTree(_resolve_rf(root)):
        print(f"{pre}{node.name}")


def register_frame(
    name,
    parent=None,
    translation=None,
    rotation=None,
    timestamps=None,
    inverse=False,
    discrete=False,
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

    discrete: bool, default False
        If True, transformations with timestamps are assumed to be events.
        Instead of interpolating between timestamps, transformations are
        fixed between their timestamp and the next one.

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
        discrete=discrete,
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
        discrete=False,
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

        timestamps: array_like, optional
            The timestamps for translation and rotation of this frame. Not
            applicable if this is a static reference frame.

        inverse: bool, default False
            If True, invert the transform wrt the parent frame, i.e. the
            translation and rotation are specified for the parent frame wrt
            this frame.

        discrete: bool, default False
            If True, transformations with timestamps are assumed to be events.
            Instead of interpolating between timestamps, transformations are
            fixed between their timestamp and the next one.
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

        if discrete and self.timestamps is None:
            raise ValueError("timestamps must be provided when discrete=True")
        else:
            self.discrete = discrete

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
            # TODO this should be done somewhere else
            arr = np.swapaxes(arr, 0, time_axis)

        return arr, timestamps

    @classmethod
    def _expand_singleton_axes(cls, t_or_r, ndim):
        """ Expand singleton axes for correct broadcasting with array. """
        if t_or_r.ndim > 1:
            for _ in range(ndim - 2):
                t_or_r = np.expand_dims(t_or_r, 1)

        return t_or_r

    @classmethod
    def _match_arrays(cls, arrays, timestamps=None):
        """ Match multiple arrays with timestamps. """
        matcher = TransformMatcher()

        for array in arrays:
            matcher.add_array(*array)

        return matcher.get_arrays(timestamps)

    def _walk(self, to_rf):
        """ Walk from this frame to a target frame along the tree. """
        to_rf = _resolve_rf(to_rf)
        walker = Walker()
        up, _, down = walker.walk(self, to_rf)
        return up, down

    def _get_matcher(self, to_frame, arrays=None):
        """ Get a TransformMatcher from this frame to another. """
        up, down = self._walk(to_frame)

        matcher = TransformMatcher()
        for rf in up:
            matcher.add_reference_frame(rf)
        for rf in down:
            matcher.add_reference_frame(rf, inverse=True)

        if arrays is not None:
            for array in arrays:
                matcher.add_array(*array)

        return matcher

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
        discrete=False,
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

        discrete: bool, default False
            If True, transformations with timestamps are assumed to be events.
            Instead of interpolating between timestamps, transformations are
            fixed between their timestamp and the next one.

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
            discrete=discrete,
        )

    @classmethod
    def from_translation_dataarray(
        cls, da, timestamps, parent, name=None, inverse=False, discrete=False,
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

        discrete: bool, default False
            If True, transformations with timestamps are assumed to be events.
            Instead of interpolating between timestamps, transformations are
            fixed between their timestamp and the next one.

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
            discrete=discrete,
        )

    @classmethod
    def from_rotation_dataarray(
        cls, da, timestamps, parent, name=None, inverse=False, discrete=False,
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

        discrete: bool, default False
            If True, transformations with timestamps are assumed to be events.
            Instead of interpolating between timestamps, transformations are
            fixed between their timestamp and the next one.

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
            discrete=discrete,
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
        """ Alias for lookup_transform.

        See Also
        --------
        ReferenceFrame.lookup_transform
        """
        import warnings

        warnings.warn(
            DeprecationWarning(
                "get_transformation is deprecated, use lookup_transform "
                "instead."
            )
        )

        return self.lookup_transform(to_frame)

    def lookup_transform(self, to_frame):
        """ Look up the transformation from this frame to another.

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

        See Also
        --------
        lookup_transform
        """
        matcher = self._get_matcher(to_frame)

        return matcher.get_transformation()

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
        matcher = self._get_matcher(to_frame, arrays=[(arr, arr_ts)])
        t, r, ts = matcher.get_transformation()
        arr, _ = matcher.get_arrays(ts)

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
        matcher = self._get_matcher(to_frame, arrays=[(arr, arr_ts)])
        t, r, ts = matcher.get_transformation()
        arr, _ = matcher.get_arrays(ts)

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
        matcher = self._get_matcher(to_frame, arrays=[(arr, arr_ts)])
        t, r, ts = matcher.get_transformation()
        arr, _ = matcher.get_arrays(ts)

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

        See Also
        --------
        transform_angular_velocity
        """
        if what == "reference_frame":
            angular, angular_ts = self.lookup_angular_velocity(
                to_frame,
                to_frame,
                cutoff=cutoff,
                allow_static=True,
                return_timestamps=True,
            )

        elif what == "moving_frame":
            angular, angular_ts = _resolve_rf(
                to_frame
            ).lookup_angular_velocity(
                self,
                to_frame,
                cutoff=cutoff,
                allow_static=True,
                return_timestamps=True,
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
                f"Expected 'what' to be 'reference_frame', 'moving_frame' or "
                f"'representation_frame', got {what}"
            )

        arr, ts = self.transform_vectors(
            arr,
            to_frame,
            axis=axis,
            time_axis=time_axis,
            timestamps=timestamps,
            return_timestamps=True,
        )

        arr, angular, ts_out = self._match_arrays(
            [(arr, ts), (angular, angular_ts)]
        )
        arr += angular

        if return_timestamps:
            return arr, ts_out
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
            Suppress outliers by throwing out samples where the
            norm of the second-order differences of the position is above
            `outlier_thresh` and interpolating the missing values.

        Returns
        -------
        arr_transformed: array_like
            The transformed array.

        ts: array_like, shape (n_timestamps,) or None
            The timestamps after the transformation.

        See Also
        --------
        transform_linear_velocity
        """
        if what == "reference_frame":
            linear, angular, linear_ts = self.lookup_twist(
                to_frame,
                to_frame,
                cutoff=cutoff,
                outlier_thresh=outlier_thresh,
                allow_static=True,
                return_timestamps=True,
            )
            angular_ts = linear_ts
            translation, _, translation_ts = _resolve_rf(
                moving_frame
            ).lookup_transform(self)

        elif what == "moving_frame":
            to_frame = _resolve_rf(to_frame)
            linear, linear_ts = to_frame.lookup_linear_velocity(
                self,
                to_frame,
                cutoff=cutoff,
                outlier_thresh=outlier_thresh,
                allow_static=True,
                return_timestamps=True,
            )
            angular, angular_ts = self.lookup_angular_velocity(
                reference_frame,
                to_frame,
                cutoff=cutoff,
                allow_static=True,
                return_timestamps=True,
            )
            translation, _, translation_ts = to_frame.lookup_transform(self)

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
                f"Expected 'what' to be 'reference_frame', 'moving_frame' or "
                f"'representation_frame', got {what}"
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

        arr, linear, angular, translation, ts_out = self._match_arrays(
            [
                (arr, ts),
                (linear, linear_ts),
                (angular, angular_ts),
                (translation, translation_ts),
            ]
        )
        arr = arr + linear + np.cross(angular, translation)

        if return_timestamps:
            return arr, ts_out
        else:
            return arr

    def lookup_twist(
        self,
        reference=None,
        represent_in=None,
        outlier_thresh=None,
        cutoff=None,
        mode="quaternion",
        allow_static=False,
        return_timestamps=False,
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
            Suppress outliers by throwing out samples where the
            norm of the second-order differences of the position is above
            `outlier_thresh` and interpolating the missing values.

        cutoff: float, optional
            Frequency of a low-pass filter applied to linear and angular
            velocity after the estimation as a fraction of the Nyquist
            frequency.

        mode: str, default "quaternion"
            If "quaternion", compute the angular velocity from the quaternion
            derivative. If "rotation_vector", compute the angular velocity from
            the gradient of the axis-angle representation of the rotations.

        allow_static: bool, default False
            If True, return a zero velocity vector and None for timestamps if
            the transform between this frame and the reference frame is static.
            Otherwise, a `ValueError` will be raised.

        return_timestamps: bool, default False
            If True, also return the timestamps of the lookup.

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

        translation, rotation, timestamps = self.lookup_transform(reference)

        if timestamps is None:
            if allow_static:
                return np.zeros(3), np.zeros(3), None
            else:
                raise ValueError(
                    "Twist cannot be estimated for static transforms"
                )

        linear = _estimate_linear_velocity(
            translation,
            timestamps,
            outlier_thresh=outlier_thresh,
            cutoff=cutoff,
        )
        angular = _estimate_angular_velocity(
            rotation, timestamps, cutoff=cutoff, mode=mode
        )

        # linear velocity is represented in reference frame after estimation
        linear, linear_ts = reference.transform_vectors(
            linear, represent_in, timestamps=timestamps, return_timestamps=True
        )
        # angular velocity is represented in moving frame after estimation
        angular, angular_ts = self.transform_vectors(
            angular,
            represent_in,
            timestamps=timestamps,
            return_timestamps=True,
        )
        angular, linear, twist_ts = self._match_arrays(
            [(angular, angular_ts), (linear, linear_ts)],
        )

        if return_timestamps:
            return linear, angular, twist_ts
        else:
            return linear, angular

    def lookup_linear_velocity(
        self,
        reference=None,
        represent_in=None,
        outlier_thresh=None,
        cutoff=None,
        allow_static=False,
        return_timestamps=False,
    ):
        """ Estimate linear velocity of this frame wrt a reference.

        Parameters
        ----------
        reference: str or ReferenceFrame, optional
            The reference frame wrt which the twist is estimated. Defaults to
            the parent frame.

        represent_in: str or ReferenceFrame, optional
            The reference frame in which the twist is represented. Defaults
            to the parent frame.

        outlier_thresh: float, optional
            Suppress outliers by throwing out samples where the
            norm of the second-order differences of the position is above
            `outlier_thresh` and interpolating the missing values.

        cutoff: float, optional
            Frequency of a low-pass filter applied to linear and angular
            velocity after the estimation as a fraction of the Nyquist
            frequency.

        allow_static: bool, default False
            If True, return a zero velocity vector and None for timestamps if
            the transform between this frame and the reference frame is static.
            Otherwise, a `ValueError` will be raised.

        return_timestamps: bool, default False
            If True, also return the timestamps of the lookup.

        Returns
        -------
        linear: numpy.ndarray, shape (N, 3)
            Linear velocity of moving frame wrt reference frame, represented
            in representation frame.

        timestamps: each numpy.ndarray
            Timestamps of the linear velocity.
        """
        try:
            reference = _resolve_rf(reference or self.parent)
            represent_in = _resolve_rf(represent_in or self.parent)
        except TypeError:
            raise ValueError(f"Frame {self.name} has no parent frame")

        translation, _, timestamps = self.lookup_transform(reference)

        if timestamps is None:
            if allow_static:
                return np.zeros(3), None
            else:
                raise ValueError(
                    "Velocity cannot be estimated for static transforms"
                )

        linear = _estimate_linear_velocity(
            translation,
            timestamps,
            outlier_thresh=outlier_thresh,
            cutoff=cutoff,
        )
        # linear velocity is represented in reference frame after estimation
        linear, linear_ts = reference.transform_vectors(
            linear, represent_in, timestamps=timestamps, return_timestamps=True
        )

        if return_timestamps:
            return linear, linear_ts
        else:
            return linear

    def lookup_angular_velocity(
        self,
        reference=None,
        represent_in=None,
        outlier_thresh=None,
        cutoff=None,
        mode="quaternion",
        allow_static=False,
        return_timestamps=False,
    ):
        """ Estimate angular velocity of this frame wrt a reference.

        Parameters
        ----------
        reference: str or ReferenceFrame, optional
            The reference frame wrt which the twist is estimated. Defaults to
            the parent frame.

        represent_in: str or ReferenceFrame, optional
            The reference frame in which the twist is represented. Defaults
            to the parent frame.

        outlier_thresh: float, optional
            Suppress samples where the norm of the second-order differences of
            the rotation is above `outlier_thresh` and interpolate the missing
            values.

        cutoff: float, optional
            Frequency of a low-pass filter applied to linear and angular
            velocity after the estimation as a fraction of the Nyquist
            frequency.

        mode: str, default "quaternion"
            If "quaternion", compute the angular velocity from the quaternion
            derivative. If "rotation_vector", compute the angular velocity from
            the gradient of the axis-angle representation of the rotations.

        allow_static: bool, default False
            If True, return a zero velocity vector and None for timestamps if
            the transform between this frame and the reference frame is static.
            Otherwise, a `ValueError` will be raised.

        return_timestamps: bool, default False
            If True, also return the timestamps of the lookup.

        Returns
        -------
        angular: numpy.ndarray, shape (N, 3)
            Angular velocity of moving frame wrt reference frame, represented
            in representation frame.

        timestamps: each numpy.ndarray
            Timestamps of the angular velocity.
        """
        try:
            reference = _resolve_rf(reference or self.parent)
            represent_in = _resolve_rf(represent_in or self.parent)
        except TypeError:
            raise ValueError(f"Frame {self.name} has no parent frame")

        _, rotation, timestamps = self.lookup_transform(reference)

        if timestamps is None:
            if allow_static:
                return np.zeros(3), None
            else:
                raise ValueError(
                    "Velocity cannot be estimated for static transforms"
                )

        angular = _estimate_angular_velocity(
            rotation,
            timestamps,
            cutoff=cutoff,
            mode=mode,
            outlier_thresh=outlier_thresh,
        )

        # angular velocity is represented in moving frame after estimation
        angular, angular_ts = self.transform_vectors(
            angular,
            represent_in,
            timestamps=timestamps,
            return_timestamps=True,
        )

        if return_timestamps:
            return angular, angular_ts
        else:
            return angular

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
