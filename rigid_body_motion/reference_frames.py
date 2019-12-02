""""""
from warnings import warn

import numpy as np
from scipy.interpolate import interp1d

from anytree import NodeMixin, Walker
from quaternion import as_quat_array, as_float_array, from_rotation_matrix

from rigid_body_motion.utils import rotate_vectors
from rigid_body_motion.core import _resolve_rf

_registry = {}


def _register(rf, update=False):
    """ Register a reference frame. """
    if rf.name is None:
        raise ValueError('Reference frame name cannot be None.')
    if rf.name in _registry:
        if update:
            # TODO keep children?
            _registry[rf.name].parent = None
        else:
            raise ValueError(
                'Reference frame with name {} is already registered. Specify '
                'update=True to overwrite.'.format(rf.name))
    # TODO check if name is a cs transform
    _registry[rf.name] = rf


def _deregister(name):
    """ Deregister a reference frame. """
    if name not in _registry:
        raise ValueError('Reference frame with name ' + name +
                         ' not found in registry')

    _registry.pop(name)


def register_frame(
        name, parent=None, translation=None, rotation=None, timestamps=None,
        inverse=False, update=False):
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
        name, parent=parent, translation=translation, rotation=rotation,
        timestamps=timestamps, inverse=inverse)
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

    def __init__(self, name=None, parent=None, translation=None, rotation=None,
                 timestamps=None, inverse=False):
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
        self.parent = _resolve_rf(parent)

        if self.parent is not None:
            self.translation, self.rotation, self.timestamps = \
                self._init_arrays(translation, rotation, timestamps, inverse)
        else:
            self._verify_root(translation, rotation, timestamps)
            self.translation, self.rotation, self.timestamps = None, None, None

    def __del__(self):
        """ Destructor. """
        if self.name in _registry and _registry[self.name] is self:
            _deregister(self.name)

    def __str__(self):
        """ String representation. """
        return '<ReferenceFrame \'{}\'>'.format(self.name)

    def __repr__(self):
        """ String representation. """
        return self.__str__()

    @staticmethod
    def _init_arrays(translation, rotation, timestamps, inverse):
        """ Initialize translation, rotation and timestamp arrays. """
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

        if inverse:
            # TODO utils.qinv
            rotation = as_float_array(1 / as_quat_array(rotation))
            translation = -rotate_vectors(as_quat_array(rotation), translation)

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

    @classmethod
    def _broadcast(cls, arr, timestamps):
        """"""
        # TODO test
        return np.tile(arr, (len(timestamps), 1))

    @classmethod
    def _interpolate(cls, source_arr, target_arr, source_ts, target_ts):
        """"""
        # TODO SLERP for quaternions
        # TODO specify time_axis as parameter
        # TODO priority=None/<rf_name>
        # TODO method + optional scipy dependency?
        ts_dtype = target_ts.dtype
        source_ts = source_ts.astype(float)
        target_ts = target_ts.astype(float)

        # TODO sort somewhere and turn these into assertions or use min/max
        #  with boolean indexing
        if np.any(np.diff(source_ts) < 0):
            raise ValueError('source_ts is not sorted.')
        if np.any(np.diff(target_ts) < 0):
            raise ValueError('target_ts is not sorted.')

        # TODO raise error when intersection is empty
        if target_ts[0] < source_ts[0]:
            target_arr = target_arr[target_ts >= source_ts[0]]
            target_ts = target_ts[target_ts >= source_ts[0]]
        if target_ts[-1] > source_ts[-1]:
            target_arr = target_arr[target_ts <= source_ts[-1]]
            target_ts = target_ts[target_ts <= source_ts[-1]]

        source_arr_interp = interp1d(source_ts, source_arr, axis=0)(target_ts)

        return source_arr_interp, target_arr, target_ts.astype(ts_dtype)

    @classmethod
    def _match_timestamps(cls, arr, arr_ts, rf_t, rf_r, rf_ts):
        """"""
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
                    rf.translation, t, rf.timestamps, ts)
                rotation, r, ts_new = cls._interpolate(
                    rf.rotation, r, rf.timestamps, ts)
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

    def _walk(self, to_rf):
        """ Walk from this frame to a target frame along the tree. """
        to_rf = _resolve_rf(to_rf)
        walker = Walker()
        up, _, down = walker.walk(self, to_rf)
        return up, down

    @classmethod
    def from_dataset(
            cls, ds, translation, rotation, timestamps, parent, name=None,
            inverse=False):
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
            name, parent, ds[translation].data, ds[rotation].data,
            ds[timestamps].data, inverse=inverse)

    @classmethod
    def from_translation_dataarray(
            cls, da, timestamps, parent, name=None, inverse=False):
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
            name, parent, translation=da.data, timestamps=da[timestamps].data,
            inverse=inverse)

    @classmethod
    def from_rotation_dataarray(
            cls, da, timestamps, parent, name=None, inverse=False):
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
            name, parent, rotation=da.data, timestamps=da[timestamps].data,
            inverse=inverse)

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
                'Expected mat to have shape (3, 3), got {}'.format(mat.shape))

        return cls(
            name, parent, rotation=as_float_array(from_rotation_matrix(mat)),
            inverse=inverse)

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

        t, r, rf_ts = self.get_transformation(to_frame)

        arr, _, r, ts = self._match_timestamps(arr, arr_ts, t, r, rf_ts)
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
            The timestamps of the points, corresponding to the first axis
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

        arr, t, r, ts = self._match_timestamps(arr, arr_ts, t, r, rf_ts)
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
            The timestamps of the quaternions, corresponding to the first axis
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

        t, r, rf_ts = self.get_transformation(to_frame)

        arr, _, r, ts = self._match_timestamps(arr, arr_ts, t, r, rf_ts)
        arr = np.swapaxes(arr, axis, -1)
        arr = as_quat_array(r) * as_quat_array(arr)
        arr = np.swapaxes(as_float_array(arr), -1, axis)

        if not return_timestamps:
            return arr
        else:
            return arr, ts

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
