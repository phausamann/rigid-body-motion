import pytest
import numpy as np
import numpy.testing as npt
from .helpers import mock_quaternion, register_rf_tree

import rigid_body_motion as rbm


class TestTransform(object):
    """"""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """"""
        rbm.clear_registry()
        yield

    def test_coordinate_system_transforms(self):
        """"""
        arr = np.ones((10, 2))
        expected = np.tile((np.sqrt(2), np.pi/4), (10, 1))
        actual = rbm.transform(arr, outof='cartesian', into='polar', axis=1)
        npt.assert_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.transform(np.ones((10, 3)), outof='cartesian', into='polar')
        with pytest.raises(ValueError):
            rbm.transform(np.ones((10, 2)), outof='unsupported', into='polar')

    def test_reference_frame_transforms(self):
        """"""
        register_rf_tree(tc1=(1., 0., 0.), tc2=(-1., 0., 0.),
                         rc2=mock_quaternion(np.pi, 0., 0.))

        # child to world
        arr_child = np.ones((10, 3, 5))
        arr_world = rbm.transform(
            arr_child, outof='child1', into='world', axis=1)

        expected = np.ones((10, 3, 5))
        expected[:, 0] = 2.
        npt.assert_almost_equal(arr_world, expected)

        # child2 to world
        arr_child2 = np.ones((10, 3, 5))
        arr_world = rbm.transform(
            arr_child2, outof='child2', into='world', axis=1)

        expected = np.ones((10, 3, 5))
        expected[:, 0] = -2.
        expected[:, 1] = -1.
        npt.assert_almost_equal(arr_world, expected)

        # child to child2
        arr_child2 = np.ones((10, 3, 5))
        arr_child = rbm.transform(
            arr_child2, outof='child2', into='child1', axis=1)

        expected = np.ones((10, 3, 5))
        expected[:, 0] = -3.
        expected[:, 1] = -1.
        npt.assert_almost_equal(arr_child, expected)

    def test_transform_points(self):
        """"""
        register_rf_tree(tc1=(1., 0., 0.), tc2=(-1., 0., 0.),
                         rc2=mock_quaternion(np.pi, 0., 0.))

        arr_child2 = (1., 1., 1.)
        arr_child1 = rbm.transform_points(
            arr_child2, outof='child2', into='child1')
        npt.assert_almost_equal(arr_child1, (-3., -1., 1.))

    def test_transform_quaternions(self):
        """"""
        register_rf_tree(tc1=(1., 0., 0.), tc2=(-1., 0., 0.),
                         rc2=mock_quaternion(np.pi, 0., 0.))

        arr_child2 = (1., 0., 0., 0.)
        arr_child1 = rbm.transform_quaternions(
            arr_child2, outof='child2', into='child1')
        npt.assert_almost_equal(arr_child1, mock_quaternion(np.pi, 0., 0.))

    def test_transform_vectors(self):
        """"""
        register_rf_tree(tc1=(1., 0., 0.), tc2=(-1., 0., 0.),
                         rc2=mock_quaternion(np.pi, 0., 0.))

        arr_child2 = (1., 1., 1.)
        arr_child1 = rbm.transform_vectors(
            arr_child2, outof='child2', into='child1')
        npt.assert_almost_equal(arr_child1, (-1., -1., 1.))
