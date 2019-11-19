""""""
import numpy as np
from scipy.interpolate import interp1d

from anytree import NodeMixin, Walker
from quaternion import \
    quaternion, as_rotation_matrix, as_quat_array, as_float_array

from rigid_body_motion.utils import rotate_vectors

_registry = {}


def _register(rf):
    """ Register a reference frame. """
    if rf.name in _registry:
        raise ValueError(
            'Reference frame with name ' + rf.name + ' is already registered')
    # TODO check if name is a cs transform
    # TODO update=True/False
    _registry[rf.name] = rf


def _deregister(name):
    """ Deregister a reference frame. """
    if name not in _registry:
        raise ValueError('Reference frame with name ' + name +
                         ' not found in registry')

    _registry.pop(name)


def register_frame(
        name, parent=None, translation=None, rotation=None, timestamps=None):
    """ Register a new reference frame in the registry.

    Parameters
    ----------
    name: str
        The name of the reference frame.

    translation: array_like, optional
        The translation of this frame wrt the parent frame. Not
        applicable if there is no parent frame.

    rotation: array_like, optional
        The rotation of this frame wrt the parent frame. Not
        applicable if there is no parent frame.

    timestamps : array_like, optional
        The timestamps for translation and rotation of this frame. Not
        applicable if this is a static reference frame.
    """
    rf = ReferenceFrame(
        name, parent=parent, translation=translation, rotation=rotation,
        timestamps=timestamps)
    _register(rf)


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
    """ A three-dimensional static reference frame. """

    def __init__(self, name, parent=None, translation=None, rotation=None,
                 timestamps=None):
        """ Constructor.

        Parameters
        ----------
        name: str
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
        """
        super(ReferenceFrame, self).__init__()

        # TODO check name requirement
        self.name = name
        self.parent = self._resolve(parent)

        if self.parent is not None:
            self.translation, self.rotation, self.timestamps = \
                self._init_arrays(translation, rotation, timestamps)
        else:
            self._verify_root(translation, rotation, timestamps)
            self.translation, self.rotation, self.timestamps = None, None, None

    def __del__(self):
        """ Destructor. """
        if self.name in _registry and _registry[self.name] is self:
            _deregister(self.name)

    @staticmethod
    def _init_arrays(translation, rotation, timestamps):
        """ Initialize translation, rotation and timestamp arrays. """
        # TODO test
        if timestamps is not None:
            timestamps = np.asarray(timestamps)
            if timestamps.ndim != 1:
                raise ValueError('timestamps must be one-dimensional.')
            t_shape = (len(timestamps), 3)
            r_shape = (len(timestamps), 4)
        else:
            t_shape = (3,)
            r_shape = (4,)

        if translation is not None:
            translation = np.asarray(translation)
            if translation.shape != t_shape:
                raise ValueError(
                    'Expected translation to be of shape {}, got {}'.format(
                        t_shape, translation.shape))
        else:
            translation = np.zeros(t_shape)

        if rotation is not None:
            rotation = np.asarray(rotation)
            if rotation.shape != r_shape:
                raise ValueError(
                    'Expected translation to be of shape {}, got {}'.format(
                        r_shape, rotation.shape))
        else:
            rotation = np.zeros(r_shape)
            rotation[..., 0] = 1.

        return translation, rotation, timestamps

    @staticmethod
    def _verify_root(translation, rotation, timestamps):
        """ Verify arguments for root node. """
        # TODO test
        if translation is not None:
            raise ValueError('translation specified without parent frame.')
        if rotation is not None:
            raise ValueError('rotation specified without parent frame.')
        if timestamps is not None:
            raise ValueError('timestamps specified without parent frame.')

    @staticmethod
    def _resolve(rf):
        """ Retrieve frame by name from registry, if applicable. """
        # TODO test
        # TODO raise error if not ReferenceFrame instance?
        if isinstance(rf, str):
            try:
                return _registry[rf]
            except KeyError:
                raise ValueError(
                    'Frame "' + rf + '" not found in registry.')
        else:
            return rf

    def _walk(self, to_rf):
        """ Walk from this frame to a target frame along the tree. """
        to_rf = self._resolve(to_rf)
        walker = Walker()
        up, _, down = walker.walk(self, to_rf)
        return up, down

    def _get_parent_transformation_matrix(self, inverse=False):
        """"""
        mat = np.eye(4)
        if inverse:
            mat[:3, :3] = as_rotation_matrix(1 / quaternion(*self.rotation))
            mat[:3, 3] = self.translation
            mat[:3, 3] = -mat[:3, 3]
        else:
            mat[:3, :3] = as_rotation_matrix(quaternion(*self.rotation))
            mat[:3, 3] = self.translation
        return mat

    @classmethod
    def _broadcast(cls, arr, timestamps):
        """"""
        return np.tile(arr, (len(timestamps), 1))

    @classmethod
    def _interpolate(cls, arr, source_ts, target_ts):
        """"""
        # TODO SLERP for quaternions
        # TODO specify time_axis as parameter
        # TODO policy='raise'/'intersect'
        # TODO priority=None/<rf_name>
        # TODO method
        return interp1d(source_ts, arr, axis=0)(target_ts)

    @classmethod
    def _match_timestamps(cls, arr, arr_ts, rf_ts):
        """"""
        # TODO policy='from_arr'/'from_rf'
        if rf_ts is None:
            return arr, arr_ts
        elif arr_ts is None:
            return cls._broadcast(arr, rf_ts), rf_ts
        else:
            return cls._interpolate(arr, arr_ts, rf_ts), rf_ts

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
                ts = rf.timestamps
            else:
                translation = cls._interpolate(
                    rf.translation, rf.timestamps, ts)
                rotation = cls._interpolate(
                    rf.rotation, rf.timestamps, ts)
        else:
            translation = rf.translation
            rotation = rf.rotation

        if inverse:
            q = 1 / as_quat_array(rotation)
            dt = -np.array(translation)
            t = rotate_vectors(q, t + dt)
        else:
            q = as_quat_array(rotation)
            dt = np.array(translation)
            t = rotate_vectors(q, t) + dt

        return t, as_float_array(q * as_quat_array(r)), ts

    @classmethod
    def _validate_input(cls, arr, axis, n_axis, timestamps):
        """"""
        # TODO process DataArray (dim=str, timestamps=str)
        arr = np.asarray(arr)

        if arr.shape[axis] != n_axis:
            raise ValueError(
                'Expected array to have length {} along axis {}, '
                'got {}'.format(n_axis, axis, arr.shape[axis]))

        if timestamps is not None:
            # TODO specify time_axis as parameter
            timestamps = np.asarray(timestamps)
            if timestamps.ndim != 1:
                raise ValueError('timestamps must be one-dimensional.')
            if arr.shape[0] != len(timestamps):
                raise ValueError('The first axis of the array must have the '
                                 'same length as the timestamps.')

        return arr, timestamps

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
        r = np.array((1., 0., 0., 0.))
        ts = None

        for rf in up:
            t, r, ts = self._add_transformation(rf, t, r, ts)

        for rf in down:
            t, r, ts = self._add_transformation(rf, t, r, ts, inverse=True)

        return t, r, ts

    def get_transformation_matrix(self, to_frame):
        """ Calculate the transformation matrix from this frame to another.

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
        mat: array, shape (4, 4)
            The transformation matrix from this frame to the target frame.
        """
        up, down = self._walk(to_frame)

        mat = np.eye(4)
        for rf in up:
            mat = np.matmul(mat, rf._get_parent_transformation_matrix(
                inverse=True))
        for rf in down:
            mat = np.matmul(mat, rf._get_parent_transformation_matrix())

        return mat

    def get_transformation_func(self, to_frame):
        """ Get the transformation function from this frame to another.

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
        func: function
            The transformation function from this frame to the target frame.
        """
        t, r, _ = self.get_transformation(to_frame)

        def transformation_func(arr, axis=-1, **kwargs):
            # TODO support quaternion dtype
            if isinstance(arr, tuple):
                return tuple(
                    transformation_func(a, axis=axis, **kwargs) for a in arr)
            elif arr.shape[axis] == 3:
                arr = rotate_vectors(as_quat_array(r), arr, axis=axis)
                t_idx = [np.newaxis] * arr.ndim
                t_idx[axis] = slice(None)
                arr = arr + np.array(t)[tuple(t_idx)]
            elif arr.shape[axis] == 4:
                arr = np.swapaxes(arr, axis, -1)
                arr = as_quat_array(r) * as_quat_array(arr)
                arr = np.swapaxes(as_float_array(arr), -1, axis)
            else:
                raise ValueError(
                    'Expected array to have size 3 or 4 along '
                    'axis {}, actual size is {}'.format(axis, arr.shape[axis]))
            return arr

        return transformation_func

    def transform_vectors(
            self, arr, to_frame, axis=-1, timestamps=None,
            return_timestamps=False):
        """ Transform an array of vectors from this frame to another.

        Parameters
        ----------
        arr: array_like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the coordinates of the vectors.

        timestamps: array_like, optional
            The timestamps of the vectors, corresponding to the first axis
            of the array. If not None, the first axis of the array will be
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
        arr, arr_ts = self._validate_input(arr, axis, 3, timestamps)

        _, r, rf_ts = self.get_transformation(to_frame)

        arr, ts = self._match_timestamps(arr, arr_ts, rf_ts)
        arr = rotate_vectors(as_quat_array(r), arr, axis=axis)

        if not return_timestamps:
            return arr
        else:
            return arr, ts

    def transform_points(
            self, arr, to_frame, axis=-1, timestamps=None,
            return_timestamps=False):
        """ Transform an array of points from this frame to another.

        Parameters
        ----------
        arr: array_like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the coordinates of the points.

        timestamps: array_like, optional
            The timestamps of the vectors, corresponding to the first axis
            of the array. If not None, the first axis of the array will be
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
        arr, arr_ts = self._validate_input(arr, axis, 3, timestamps)

        t, r, rf_ts = self.get_transformation(to_frame)

        arr, ts = self._match_timestamps(arr, arr_ts, rf_ts)
        arr = rotate_vectors(as_quat_array(r), arr, axis=axis)
        arr = arr + np.array(t)

        if not return_timestamps:
            return arr
        else:
            return arr, ts

    def transform_quaternions(
            self, arr, to_frame, axis=-1, timestamps=None,
            return_timestamps=False):
        """ Transform an array of quaternions from this frame to another.

        Parameters
        ----------
        arr: array_like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the coordinates of the
            quaternions.

        timestamps: array_like, optional
            The timestamps of the vectors, corresponding to the first axis
            of the array. If not None, the first axis of the array will be
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
        arr, arr_ts = self._validate_input(arr, axis, 4, timestamps)

        _, r, rf_ts = self.get_transformation(to_frame)

        arr, ts = self._match_timestamps(arr, arr_ts, rf_ts)
        arr = np.swapaxes(arr, axis, -1)
        arr = as_quat_array(r) * as_quat_array(arr)
        arr = np.swapaxes(as_float_array(arr), -1, axis)

        if not return_timestamps:
            return arr
        else:
            return arr, ts

    def register(self):
        """ Register this frame in the registry. """
        # TODO update=True/False
        _register(self)

    def deregister(self):
        """ Remove this frame from the registry. """
        _deregister(self.name)
