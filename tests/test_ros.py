import pytest

import numpy as np
from quaternion import from_euler_angles, as_float_array

import rigid_body_motion as rbm
from rigid_body_motion.ros import Transformer


def mock_quaternion(*angles):
    """"""
    return as_float_array(from_euler_angles(*angles))


@pytest.fixture()
def basic_rf():
    """"""
    rf_world = rbm.ReferenceFrame('world')
    rf_child1 = rbm.ReferenceFrame(
        'child1', parent=rf_world, translation=(1., 0., 0.))
    rf_child2 = rbm.ReferenceFrame(
        'child2', parent=rf_world, rotation=mock_quaternion(np.pi / 4, 0., 0.))

    yield rf_child1


class TestTransformer(object):

    def test_can_transform(self, basic_rf):
        """"""
        transformer = Transformer.from_reference_frame(basic_rf)
        assert transformer.can_transform('child1', 'child2')

    def test_lookup_transform(self, basic_rf):
        """"""
        transformer = Transformer.from_reference_frame(basic_rf)
        t, r = transformer.lookup_transform('child1', 'child2')
        np.testing.assert_allclose(t, (-1., 0., 0.))
        np.testing.assert_allclose(r, mock_quaternion(np.pi / 4, 0., 0.))

    def test_transform_vector(self, basic_rf):
        """"""
        transformer = Transformer.from_reference_frame(basic_rf)
        v = (1., 0., 0.)
        vt = transformer.transform_vector(v, 'child1', 'child2')
        np.testing.assert_allclose(
            vt, (np.sqrt(2.) / 2., np.sqrt(2.) / 2., 0.))

    def test_transform_point(self, basic_rf):
        """"""
        transformer = Transformer.from_reference_frame(basic_rf)
        p = (1., 0., 0.)
        pt = transformer.transform_point(p, 'child1', 'child2')
        np.testing.assert_allclose(pt, (-0.2928932, np.sqrt(2.) / 2., 0.))

    def test_transform_pose(self, basic_rf):
        """"""
        transformer = Transformer.from_reference_frame(basic_rf)
        p = (1., 0., 0.)
        o = mock_quaternion(np.pi / 4, 0., 0.)
        pt, ot = transformer.transform_pose(p, o, 'child1', 'child2')
        np.testing.assert_allclose(pt, (-0.2928932, np.sqrt(2.) / 2., 0.))
        np.testing.assert_allclose(
            ot, (np.sqrt(2.) / 2., 0., 0., np.sqrt(2.) / 2.))
