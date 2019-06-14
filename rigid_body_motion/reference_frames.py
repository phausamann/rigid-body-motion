""""""
import numpy as np

from anytree import NodeMixin, Walker
from quaternion import \
    quaternion, as_rotation_matrix, from_rotation_matrix, as_float_array

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
    return ReferenceFrame(name, parent=parent, translation=translation,
                          rotation=rotation, register=True)


def deregister_frame(name):
    """"""
    _deregister(name)


def clear_registry():
    """"""
    _registry.clear()


class ReferenceFrame(NodeMixin):
    """"""

    def __init__(self, name, parent=None, translation=None, rotation=None,
                 register=True):
        """"""
        super(ReferenceFrame, self).__init__()

        self.name = name
        self.register = register
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

        if self.register:
            _register(self)

    def __del__(self):
        """"""
        if self.register and self.name in _registry:
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

    def _get_parent_transform_matrix(self, inverse=False):
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

    def get_transform(self, to_rf):
        """"""
        up, down = self._walk(to_rf)

        mat = np.eye(4)
        for rf in up:
            mat = np.matmul(mat, rf._get_parent_transform_matrix(inverse=True))
        for rf in down:
            mat = np.matmul(mat, rf._get_parent_transform_matrix())

        translation = tuple(mat[:3, 3])
        rotation = tuple(as_float_array(from_rotation_matrix(mat[:3, :3])))

        return translation, rotation
