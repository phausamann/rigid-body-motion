import pytest
import numpy as np
import numpy.testing as npt

import rigid_body_motion as rbm


class TestTransform(object):
    """"""

    def test_supported_transforms(self):
        """"""
        arr = np.ones((10, 2))
        expected = np.tile((np.sqrt(2), np.pi/4), (10, 1))
        actual = rbm.transform(arr, outof='cartesian', into='polar', axis=1)
        npt.assert_equal(actual, expected)

    def test_unsupported_transforms(self):
        """"""
        with pytest.raises(ValueError):
            rbm.transform(np.ones((10, 3)), outof='cartesian', into='polar')
        with pytest.raises(ValueError):
            rbm.transform(np.ones((10, 2)), outof='unsupported', into='polar')


class TestTransformCoordinateSystems(object):
    """"""

    def test_cartesian_to_polar(self):
        """"""
        arr = np.ones((10, 2))
        expected = np.tile((np.sqrt(2), np.pi/4), (10, 1))
        actual = rbm.cartesian_to_polar(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.cartesian_to_polar(np.ones((10, 3)), axis=1)

    def test_polar_to_cartesian(self):
        """"""
        arr = np.tile((np.sqrt(2), np.pi/4), (10, 1))
        expected = np.ones((10, 2))
        actual = rbm.polar_to_cartesian(arr, axis=1)
        npt.assert_almost_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.polar_to_cartesian(np.ones((10, 3)), axis=1)
