import pytest
import numpy as np
import numpy.testing as npt

from quaternion import from_euler_angles, as_float_array

import rigid_body_motion as rbm


def mock_quaternion(*angles):
    """"""
    return as_float_array(from_euler_angles(*angles))


class TestTransform(object):
    """"""

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
        rbm.register_frame('world')
        rbm.register_frame('child', parent='world', translation=(1., 0., 0.))
        rbm.register_frame('child2', parent='world', translation=(-1., 0., 0.),
                           rotation=mock_quaternion(np.pi, 0., 0.))

        # child to world
        arr_child = np.ones((10, 3, 5))
        arr_world = rbm.transform(
            arr_child, outof='child', into='world', axis=1)

        expected = np.ones((10, 3, 5))
        expected[:, 0] = 0.
        npt.assert_almost_equal(arr_world, expected)

        # child2 to world
        arr_child2 = np.ones((10, 3, 5))
        arr_world = rbm.transform(
            arr_child2, outof='child2', into='world', axis=1)

        expected = np.ones((10, 3, 5))
        expected[:, 0] = 0.
        expected[:, 1] = -1.
        npt.assert_almost_equal(arr_world, expected)

        # child to child2
        arr_child2 = np.ones((10, 3, 5))
        arr_child = rbm.transform(
            arr_child2, outof='child2', into='child', axis=1)

        expected = np.ones((10, 3, 5))
        expected[:, 0] = -1.
        expected[:, 1] = -1.
        npt.assert_almost_equal(arr_child, expected)
