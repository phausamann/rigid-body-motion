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

    def test_transform_vector(self, basic_rf):
        """"""
        transformer = Transformer.from_reference_frame(basic_rf)
        vector = (1., 0., 0.)
        np.testing.assert_allclose(
            transformer.transform_vector(vector, 'child1', 'child2'),
            (np.sqrt(2.) / 2., np.sqrt(2.) / 2., 0.))

    def test_transform_point(self, basic_rf):
        """"""
        transformer = Transformer.from_reference_frame(basic_rf)
        point = (1., 0., 0.)
        np.testing.assert_allclose(
            transformer.transform_point(point, 'child1', 'child2'),
            (-0.2928932, np.sqrt(2.) / 2., 0.))
