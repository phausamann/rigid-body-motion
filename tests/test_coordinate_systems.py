import numpy as np
import pytest
from numpy import testing as npt

import rigid_body_motion as rbm


class TestCoordinateSystems:
    def test_cartesian_to_polar(self):
        """"""
        arr = np.ones((10, 2))
        expected = np.tile((np.sqrt(2), np.pi / 4), (10, 1))
        actual = rbm.cartesian_to_polar(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.cartesian_to_polar(np.ones((10, 3)), axis=1)

    def test_polar_to_cartesian(self):
        """"""
        arr = np.tile((np.sqrt(2), np.pi / 4), (10, 1))
        expected = np.ones((10, 2))
        actual = rbm.polar_to_cartesian(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.polar_to_cartesian(np.ones((10, 3)), axis=1)

    def test_cartesian_to_spherical(self):
        """"""
        arr = np.ones((10, 3))
        expected = np.tile((np.sqrt(3), 0.9553166, np.pi / 4), (10, 1))
        actual = rbm.cartesian_to_spherical(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.cartesian_to_spherical(np.ones((10, 2)), axis=1)

    def test_spherical_to_cartesian(self):
        """"""
        arr = np.tile((np.sqrt(3), 0.9553166, np.pi / 4), (10, 1))
        expected = np.ones((10, 3))
        actual = rbm.spherical_to_cartesian(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.spherical_to_cartesian(np.ones((10, 2)), axis=1)
