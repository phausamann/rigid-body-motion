import pytest
import numpy as np
from numpy import testing as npt

from quaternion import from_euler_angles, as_float_array

import rigid_body_motion as rbm
from rigid_body_motion.reference_frames import _register, _deregister


def mock_quaternion(*angles):
    """"""
    return as_float_array(from_euler_angles(*angles))


class TestReferenceFrameRegistry(object):
    """"""

    def test_register(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        _register(rf_world)
        assert rbm._rf_registry['world'] is rf_world

        with pytest.raises(ValueError):
            _register(rf_world)

    def test_deregister(self):
        """"""
        _deregister('world')
        assert 'world' not in rbm._rf_registry

        with pytest.raises(ValueError):
            _deregister('not_an_rf')

    def test_register_frame(self):
        """"""
        rbm.register_frame('world')
        assert isinstance(rbm._rf_registry['world'], rbm.ReferenceFrame)

    def test_deregister_frame(self):
        """"""
        rbm.deregister_frame('world')
        assert 'world' not in rbm._rf_registry

    def test_clear_registry(self):
        """"""
        rbm.register_frame('world')
        rbm.clear_registry()
        assert len(rbm._rf_registry) == 0


class TestReferenceFrame(object):
    """"""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """"""
        rbm.clear_registry()
        yield

    def test_constructor(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent=rf_world)
        assert rf_child.parent is rf_world

        _register(rf_world)
        rf_child2 = rbm.ReferenceFrame('child2', parent='world')
        assert rf_child2.parent is rf_world

        # invalid parent
        with pytest.raises(ValueError):
            rbm.ReferenceFrame('child3', parent='not_an_rf')

    def test_destructor(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        _register(rf_world)
        del rf_world
        assert 'world' in rbm._rf_registry
        del rbm._rf_registry['world']
        assert 'world' not in rbm._rf_registry

    def test_register(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_world.register()
        assert 'world' in rbm._rf_registry
        assert rbm._rf_registry['world'] is rf_world

    def test_deregister(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_world.register()
        rf_world.deregister()
        assert 'world' not in rbm._rf_registry

    def test_walk(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent=rf_world)
        rf_child2 = rbm.ReferenceFrame('child2', parent=rf_world)

        up, down = rf_child._walk(rf_child2)
        assert up == (rf_child,)
        assert down == (rf_child2,)

    def test_get_parent_transform_matrix(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent=rf_world,
                                      translation=(1., 0., 0.))

        actual = rf_child._get_parent_transform_matrix()
        expected = np.eye(4)
        expected[0, 3] = 1.
        npt.assert_equal(actual, expected)

        actual = rf_child._get_parent_transform_matrix(inverse=True)
        expected = np.eye(4)
        expected[0, 3] = -1.
        npt.assert_equal(actual, expected)

    def test_get_transform(self):
        """"""
        rf_world = rbm.ReferenceFrame('world')

        # translation only
        rf_child1 = rbm.ReferenceFrame(
            'child1', parent=rf_world, translation=(1., 0., 0.))
        rf_child2 = rbm.ReferenceFrame(
            'child2', parent=rf_world, translation=(-1., 0., 0.))

        t, r = rf_child1.get_transform(rf_child2)
        npt.assert_almost_equal(t, (2., 0., 0.))
        npt.assert_almost_equal(r, (1., 0., 0., 0.))

        # rotation only
        rf_child1 = rbm.ReferenceFrame(
            'child1', parent=rf_world,
            rotation=mock_quaternion(np.pi/4, 0., 0.))
        rf_child2 = rbm.ReferenceFrame(
            'child2', parent=rf_world,
            rotation=mock_quaternion(-np.pi/4, 0., 0.))

        t, r = rf_child1.get_transform(rf_child2)
        npt.assert_almost_equal(t, (0., 0., 0.))
        npt.assert_almost_equal(r, (np.sqrt(2)/2, 0., 0., np.sqrt(2)/2))

        # both
        rf_child1 = rbm.ReferenceFrame(
            'child1', parent=rf_world, translation=(1., 1., 0.))
        rf_child2 = rbm.ReferenceFrame(
            'child2', parent=rf_world, translation=(-1., 0., 0.),
            rotation=mock_quaternion(np.pi/2, np.pi/3, 0.))

        t, r = rf_child1.get_transform(rf_child2)
        npt.assert_almost_equal(t, (0.5, -2., 0.8660254))
        npt.assert_almost_equal(r, (0.6123724356957946, 0.3535533905932737,
                                    -0.35355339059327373, -0.6123724356957945))
