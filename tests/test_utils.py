import pytest
import numpy as np
from numpy import testing as npt

from quaternion import quaternion, as_float_array, from_euler_angles

import rigid_body_motion as rbm
from rigid_body_motion.utils import _resolve_axis, rotate_vectors


class TestUtils(object):

    def test_resolve_axis(self):
        """"""
        assert _resolve_axis(0, 1) == 0
        assert _resolve_axis(-1, 1) == 0
        assert _resolve_axis((0, -1), 2) == (0, 1)
        assert _resolve_axis(None, 2) == (0, 1)

        with pytest.raises(IndexError):
            _resolve_axis(2, 1)
        with pytest.raises(IndexError):
            _resolve_axis((-2, 0), 1)

    def test_qmean(self):
        """"""
        q = np.hstack((
            from_euler_angles(0., 0., np.pi/4),
            from_euler_angles(0., 0., -np.pi/4),
            from_euler_angles(0., np.pi/4, 0.),
            from_euler_angles(0., -np.pi/4, 0.),
            quaternion(1., 0., 0., 0.),
        ))

        qm = rbm.qmean(q)
        npt.assert_allclose(as_float_array(qm),
                            np.array([1., 0., 0., 0.]))

        qm = rbm.qmean(np.tile(q, (10, 1)), axis=1)
        npt.assert_allclose(as_float_array(qm),
                            np.tile(np.array([1., 0., 0., 0.]), (10, 1)))

        with pytest.raises(ValueError):
            rbm.qmean(np.array([1., 0., 0., 0.]))

    def test_rotate(self):
        """"""
        v = np.ones((10, 3))
        q = np.tile(from_euler_angles(0., 0., np.pi / 4), 10)
        vr = np.vstack((np.zeros(10), np.sqrt(2)*np.ones(10), np.ones(10))).T

        # single quaternion, single vector
        vr_act = rotate_vectors(q[0], v[0])
        np.testing.assert_allclose(vr[0], vr_act, rtol=1.)

        # single quaternion, multiple vectors
        vr_act = rotate_vectors(q[0], v)
        np.testing.assert_allclose(vr, vr_act, rtol=1.)

        # single quaternion, explicit axis
        vr_act = rotate_vectors(q[0], v, axis=1)
        np.testing.assert_allclose(vr, vr_act, rtol=1.)

        # multiple quaternions, multiple vectors
        vr_act = rotate_vectors(q, v)
        np.testing.assert_allclose(vr, vr_act)

        # different axis
        vr_act = rotate_vectors(q, v.T, axis=0)
        np.testing.assert_allclose(vr.T, vr_act)

        # singleton expansion
        vr_act = rotate_vectors(q[:, None], v[None, ...])
        np.testing.assert_allclose(np.tile(vr, (10, 1, 1)), vr_act)

        with pytest.raises(ValueError):
            rotate_vectors(q, v.T)

        with pytest.raises(ValueError):
            rotate_vectors(q, np.ones((10, 4)))
