""""""
import numpy as np

from anytree import NodeMixin, Walker
from quaternion import \
    quaternion, as_rotation_matrix, as_quat_array, as_float_array, \
    rotate_vectors

_registry = {}


def _register(rf):
    """ Register a reference frame. """
    if rf.name in _registry:
        raise ValueError('Reference frame with name ' + rf.name +
                         ' is already registered')
    # TODO check if name is a cs transform
    _registry[rf.name] = rf


def _deregister(name):
    """ Deregister a reference frame. """
    if name not in _registry:
        raise ValueError('Reference frame with name ' + name +
                         ' not found in registry')

    _registry.pop(name)


def register_frame(name, parent=None, translation=None, rotation=None):
    """ Register a new reference frame in the registry.

    Parameters
    ----------
    name: str
        The name of the reference frame.

    parent: str or ReferenceFrame, optional
        The parent reference frame. If str, the frame will be looked up
        in the registry under that name. If not specified, this frame
        will be a root node of a new reference frame tree.

    translation: tuple, len 3, optional
        The translation of this frame wrt the parent frame. Not
        applicable if there is no parent frame.

    rotation: tuple, len 4, optional
        The rotation of this frame wrt the parent frame. Not
        applicable if there is no parent frame.
    """
    rf = ReferenceFrame(
            name, parent=parent, translation=translation, rotation=rotation)
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

    def __init__(self, name, parent=None, translation=None, rotation=None):
        """ Constructor.

        Parameters
        ----------
        name: str
            The name of this reference frame.

        parent: str or ReferenceFrame, optional
            The parent reference frame. If str, the frame will be looked up
            in the registry under that name. If not specified, this frame
            will be a root node of a new reference frame tree.

        translation: tuple, len 3, optional
            The translation of this frame wrt the parent frame. Not
            applicable if there is no parent frame.

        rotation: tuple, len 4, optional
            The rotation of this frame wrt the parent frame. Not
            applicable if there is no parent frame.
        """
        super(ReferenceFrame, self).__init__()

        # TODO check name requirement
        self.name = name
        self.parent = self._resolve(parent)

        if self.parent is not None:
            if translation is None:
                self.translation = (0., 0., 0.)
            else:
                self.translation = translation
            if rotation is None:
                self.rotation = (1., 0., 0., 0.)
            else:
                self.rotation = rotation
        else:
            if translation is not None:
                raise ValueError(
                    'translation specificied without parent frame.')
            else:
                self.translation = None
            if rotation is not None:
                raise ValueError(
                    'rotation specificied without parent frame.')
            else:
                self.rotation = None

    def __del__(self):
        """ Destructor. """
        if self.name in _registry and _registry[self.name] is self:
            _deregister(self.name)

    def _resolve(self, rf):
        """ Retrieve frame by name from registry, if applicable. """
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

    def _add_transformation(self, rf, t, r, inverse=False):
        """ Add transformation of this frame to current transformation. """
        if inverse:
            q = 1 / quaternion(*rf.rotation)
            dt = -np.array(rf.translation)
            t = rotate_vectors(q, t + dt)
        else:
            q = quaternion(*rf.rotation)
            dt = np.array(rf.translation)
            t = rotate_vectors(q, t) + dt

        return t, q * r

    def register(self):
        """ Register this frame in the registry. """
        _register(self)

    def deregister(self):
        """ Remove this frame from the registry. """
        _deregister(self.name)

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
        t: tuple, len 3
            The translation from this frame to the target frame.

        r: tuple, len 4
            The rotation from this frame to the target frame.
        """
        up, down = self._walk(to_frame)

        t = np.zeros(3)
        r = quaternion(1., 0., 0., 0.)

        for rf in up:
            t, r = self._add_transformation(rf, t, r)

        for rf in down:
            t, r = self._add_transformation(rf, t, r, inverse=True)

        t = tuple(t)
        r = tuple(as_float_array(r))

        return t, r

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
        t, r = self.get_transformation(to_frame)

        def transformation_func(arr, axis=-1, **kwargs):
            # TODO support quaternion dtype
            if isinstance(arr, tuple):
                return tuple(
                    transformation_func(a, axis=axis, **kwargs) for a in arr)
            elif arr.shape[axis] == 3:
                arr = rotate_vectors(quaternion(*r), arr, axis=axis)
                t_idx = [np.newaxis] * arr.ndim
                t_idx[axis] = slice(None)
                arr = arr + np.array(t)[tuple(t_idx)]
            elif arr.shape[axis] == 4:
                arr = np.swapaxes(arr, axis, -1)
                arr = quaternion(*r) * as_quat_array(arr)
                arr = np.swapaxes(as_float_array(arr), -1, axis)
            else:
                raise ValueError(
                    'Expected array to have size 3 or 4 along '
                    'axis {}, actual size is {}'.format(axis, arr.shape[axis]))
            return arr

        return transformation_func

    def transform_vectors(self, arr, to_frame, axis=-1):
        """ Transform an array of vectors from this frame to another.

        Parameters
        ----------
        arr: array-like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the coordinates of the vectors.

        Returns
        -------
        arr_transformed: array-like
            The transformed array.
        """
        _, r = self.get_transformation(to_frame)
        arr = rotate_vectors(quaternion(*r), arr, axis=axis)

        return arr

    def transform_points(self, arr, to_frame, axis=-1):
        """ Transform an array of points from this frame to another.

        Parameters
        ----------
        arr: array-like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the coordinates of the points.

        Returns
        -------
        arr_transformed: array-like
            The transformed array.
        """
        t, r = self.get_transformation(to_frame)
        arr = rotate_vectors(quaternion(*r), arr, axis=axis)
        t_idx = [np.newaxis] * arr.ndim
        t_idx[axis] = slice(None)
        arr = arr + np.array(t)[tuple(t_idx)]

        return arr

    def transform_quaternions(self, arr, to_frame, axis=-1):
        """ Transform an array of quaternions from this frame to another.

        Parameters
        ----------
        arr: array-like
            The array to transform.

        to_frame: str or ReferenceFrame
            The target reference frame. If str, the frame will be looked up
            in the registry under that name.

        axis: int, default -1
            The axis of the array representing the coordinates of the
            quaternions.

        Returns
        -------
        arr_transformed: array-like
            The transformed array.
        """
        t, r = self.get_transformation(to_frame)
        arr = np.swapaxes(arr, axis, -1)
        arr = quaternion(*r) * as_quat_array(arr)
        arr = np.swapaxes(as_float_array(arr), -1, axis)

        return arr
