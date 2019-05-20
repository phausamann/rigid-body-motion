import pytest
import numpy as np
import numpy.testing as npt

import rigid_body_motion as rbm


class TestTransformCoordinateSystems(object):
    """"""

    def test_no_source(self):
        """"""
        with pytest.raises(ValueError):
            rbm.transform(np.ones((10, 2))).to_('polar')

    def test_unsupported(self):
        """"""
        with pytest.raises(ValueError):
            rbm.transform(np.ones((10, 2))).from_('unsupported').to_('polar')

    def test_cartesian_to_polar(self):
        """"""
        arr = np.ones((10, 2))
        expected = np.tile((np.sqrt(2), np.pi/4), (10, 1))
        actual = rbm.transform(arr).from_('cartesian').to_('polar')
        npt.assert_equal(actual, expected)

        with pytest.raises(ValueError):
            rbm.transform(np.ones((10, 3))).from_('cartesian').to_('polar')
