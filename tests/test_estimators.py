import pytest
from numpy import testing as npt

import numpy as np

from rigid_body_motion.estimators import shortest_arc_rotation


class TestEstimators(object):

    def test_shortest_arc_rotation(self):
        """"""
        v1 = np.zeros((10, 3))
        v1[:, 0] = 1.
        v2 = np.zeros((10, 3))
        v2[:, 1] = 1.
        q_exp = np.tile((np.sqrt(2)/2, 0., 0., np.sqrt(2)/2), (10, 1))

        npt.assert_allclose(shortest_arc_rotation(v1, v2), q_exp)
