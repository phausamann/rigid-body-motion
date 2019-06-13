import pytest
import numpy as np
from numpy import testing as npt

from quaternion import from_euler_angles, as_float_array

import rigid_body_motion as rbm


def mock_quaternion(*angles):
    """"""
    return as_float_array(from_euler_angles(*angles))


class TestReferenceFrameRegistry(object):
    """"""

    def test_register_reference_frame(self):
        """"""
        rf_world = rbm.ReferenceFrame('world', register=False)
        rbm.register_reference_frame(rf_world)
        assert rbm._rf_registry['world'] is rf_world

        with pytest.raises(ValueError):
            rbm.register_reference_frame(rf_world)

    def test_deregister_reference_frame(self):
        """"""
        rbm.deregister_reference_frame('world')
        assert 'world' not in rbm._rf_registry

        with pytest.raises(ValueError):
            rbm.deregister_reference_frame('not_an_rf')


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
        rf_child2 = rbm.ReferenceFrame('child2', parent='world')

        assert rf_child.parent is rf_world
        assert rf_child2.parent is rf_world
        assert rbm._rf_registry['world'] is rf_world
        assert rbm._rf_registry['child'] is rf_child
        assert rbm._rf_registry['child2'] is rf_child2

        # already registered
        with pytest.raises(ValueError):
            rbm.ReferenceFrame('world')

        # invalid parent
        with pytest.raises(ValueError):
            rbm.ReferenceFrame('child3', parent='not_an_rf')

    def test_destructor(self):
        """"""
        rbm.ReferenceFrame('world')
        del rbm._rf_registry['world']
        assert 'world' not in rbm._rf_registry

    def test_walk(self):
        """"""
        rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent='world')
        rf_child2 = rbm.ReferenceFrame('child2', parent='world')

        up, down = rf_child._walk('child2')
        assert up == (rf_child,)
        assert down == (rf_child2,)

    def test_get_parent_transform_matrix(self):
        """"""
        rbm.ReferenceFrame('world')
        rf_child = rbm.ReferenceFrame('child', parent='world',
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
        rbm.ReferenceFrame('world')

        # translation only
        rf_child = rbm.ReferenceFrame(
            'child', parent='world', translation=(1., 0., 0.))
        rf_child2 = rbm.ReferenceFrame(
            'child2', parent='world', translation=(-1., 0., 0.))

        translation, rotation = rf_child.get_transform(rf_child2)
        npt.assert_almost_equal(translation, (-2., 0., 0.))
        npt.assert_almost_equal(rotation, (1., 0., 0., 0.))

        rf_child = rbm.ReferenceFrame(
            'child3', parent='world',
            rotation=mock_quaternion(np.pi/4, 0., 0.))
        rf_child2 = rbm.ReferenceFrame(
            'child4', parent='world',
            rotation=mock_quaternion(-np.pi/4, 0., 0.))

        translation, rotation = rf_child.get_transform(rf_child2)
        npt.assert_almost_equal(translation, (0., 0., 0.))
        npt.assert_almost_equal(rotation, -mock_quaternion(-np.pi/2, 0., 0.))

        rf_child = rbm.ReferenceFrame(
            'child5', parent='world',
            translation=(1., 0., 0.),
            rotation=mock_quaternion(np.pi/4, 0., 0.))
        rf_child2 = rbm.ReferenceFrame(
            'child6', parent='world',
            translation=(-1., 0., 0.),
            rotation=mock_quaternion(-np.pi/4, 0., 0.))

        translation, rotation = rf_child.get_transform(rf_child2)
        npt.assert_almost_equal(translation,
                                (-1.-np.sqrt(2)/2, np.sqrt(2)/2, 0.))
        npt.assert_almost_equal(rotation, -mock_quaternion(-np.pi/2, 0., 0.))
