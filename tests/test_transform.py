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
