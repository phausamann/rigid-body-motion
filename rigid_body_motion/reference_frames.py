""""""
import numpy as np

from anytree import NodeMixin, Walker
from quaternion import \
    quaternion, as_rotation_matrix, as_float_array, rotate_vectors

_registry = {}


def _register(rf):
    """"""
    if rf.name in _registry:
        raise ValueError('Reference frame with name ' + rf.name +
                         ' is already registered')
    # TODO check if name is a cs transform
    _registry[rf.name] = rf


def _deregister(name):
    """"""
    if name not in _registry:
        raise ValueError('Reference frame with name ' + name +
                         ' not found in registry')

    _registry.pop(name)


def register_frame(name, parent=None, translation=None, rotation=None):
    """"""
    rf = ReferenceFrame(
            name, parent=parent, translation=translation, rotation=rotation)
    _register(rf)


def deregister_frame(name):
    """"""
    _deregister(name)


def clear_registry():
    """"""
    _registry.clear()


class ReferenceFrame(NodeMixin):
    """"""

    def __init__(self, name, parent=None, translation=None, rotation=None):
        """"""
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
                self.translation = None

    def __del__(self):
        """"""
        if self.name in _registry and _registry[self.name] is self:
            _deregister(self.name)

    def _resolve(self, rf):
        """"""
        if isinstance(rf, str):
            try:
                return _registry[rf]
            except KeyError:
                raise ValueError(
                    'Frame "' + rf + '" not found in registry.')
        else:
            return rf

    def _walk(self, to_rf):
        """"""
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

    def _get_parent_transformation(self, rf, t, r, inverse=False):
        """"""
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
        """"""
        _register(self)

    def deregister(self):
        """"""
        _deregister(self.name)

    def get_transformation(self, to_rf):
        """"""
        up, down = self._walk(to_rf)

        t = np.zeros(3)
        r = quaternion(1., 0., 0., 0.)

        for rf in up:
            t, r = self._get_parent_transformation(rf, t, r)

        for rf in down:
            t, r = self._get_parent_transformation(rf, t, r, inverse=True)

        t = tuple(t)
        r = tuple(as_float_array(r))

        return t, r

    def get_transformation_matrix(self, to_rf):
        """"""
        up, down = self._walk(to_rf)

        mat = np.eye(4)
        for rf in up:
            mat = np.matmul(mat, rf._get_parent_transformation_matrix(
                inverse=True))
        for rf in down:
            mat = np.matmul(mat, rf._get_parent_transformation_matrix())

        return mat

    def get_transformation_func(self, to_rf):
        """"""
        t, r = self.get_transformation(to_rf)

        def transformation_func(arr, axis=-1, **kwargs):
            t_idx = [np.newaxis] * arr.ndim
            t_idx[axis] = slice(None)
            arr = arr + np.array(t)[tuple(t_idx)]
            arr = rotate_vectors(quaternion(*r), arr, axis=axis)
            return arr

        return transformation_func
