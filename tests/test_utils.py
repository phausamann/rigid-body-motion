import numpy as np
import pytest
from numpy import testing as npt
from quaternion import as_float_array, as_quat_array, quaternion

from rigid_body_motion.testing import make_test_motion
from rigid_body_motion.utils import (
    from_euler_angles,
    qinterp,
    qinv,
    qmean,
    qmul,
    rotate_vectors,
)


class TestUtils(object):
    def test_qinv(self):
        """"""
        q = from_euler_angles(roll=np.pi / 4, return_quaternion=True)

        assert qinv(q) == 1 / q
        npt.assert_equal(qinv(as_float_array(q)), as_float_array(1 / q))

        q_arr = np.tile(as_float_array(q), (10, 1)).T
        npt.assert_equal(qinv(q_arr, 0)[:, 0], as_float_array(1 / q))

    def test_qmul(self):
        """"""
        q1 = from_euler_angles(roll=np.pi / 4, return_quaternion=True)
        q2 = from_euler_angles(roll=np.pi / 4, return_quaternion=True)

        assert qmul(q1, q1) == q1 * q2
        npt.assert_equal(
            qmul(as_float_array(q1), as_float_array(q2)),
            as_float_array(q1 * q2),
        )

        q_arr = np.tile(as_float_array(q1), (10, 1))
        npt.assert_equal(
            qmul(q_arr, as_float_array(q2)),
            as_float_array(as_quat_array(q_arr) * q2),
        )

        # only one quaternion
        with pytest.raises(ValueError):
            qmul(q1)

        # different dtypes
        with pytest.raises(ValueError):
            qmul(q1, as_float_array(q2))

    def test_qmean(self):
        """"""
        q = np.hstack(
            (
                from_euler_angles(roll=np.pi / 4, return_quaternion=True),
                from_euler_angles(roll=-np.pi / 4, return_quaternion=True),
                from_euler_angles(pitch=np.pi / 4, return_quaternion=True),
                from_euler_angles(pitch=-np.pi / 4, return_quaternion=True),
                quaternion(1.0, 0.0, 0.0, 0.0),
            )
        )

        # quaternion dtype
        qm = qmean(q)
        npt.assert_allclose(as_float_array(qm), np.array([1.0, 0.0, 0.0, 0.0]))

        # float dtype
        qm = qmean(as_float_array(q).T, qaxis=0)
        npt.assert_allclose(qm, np.array([1.0, 0.0, 0.0, 0.0]))

        # not all axes
        qm = qmean(np.tile(q, (10, 1)), axis=1)
        npt.assert_allclose(
            as_float_array(qm),
            np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (10, 1)),
        )

    def test_qinterp(self):
        """"""
        _, r_100, ts_100 = make_test_motion(100, fs=100, stack=True)
        _, r_200, ts_200 = make_test_motion(200, fs=200, stack=True)

        # standard case
        r_act = qinterp(r_200, ts_200, ts_100)
        np.testing.assert_allclose(r_act, r_100)

        # quaternion dtype
        r_act = qinterp(as_quat_array(r_200), ts_200, ts_100)
        np.testing.assert_allclose(as_float_array(r_act), r_100)

        # different axis
        r_act = qinterp(r_200.T, ts_200, ts_100, qaxis=0)
        np.testing.assert_allclose(r_act, r_100.T)

        # higher dimensions
        r_act = qinterp(np.tile(r_200, (5, 1, 1)), ts_200, ts_100, axis=1)
        np.testing.assert_allclose(r_act, np.tile(r_100, (5, 1, 1)))

    def test_rotate_vectors(self):
        """"""
        v = np.ones((10, 3))
        q = np.tile(
            from_euler_angles(yaw=np.pi / 4, return_quaternion=True), 10
        )
        vr = np.vstack((np.zeros(10), np.sqrt(2) * np.ones(10), np.ones(10))).T

        # single quaternion, single vector
        vr_act = rotate_vectors(q[0], v[0])
        np.testing.assert_allclose(vr[0], vr_act, rtol=1.0)

        # single quaternion, multiple vectors
        vr_act = rotate_vectors(q[0], v)
        np.testing.assert_allclose(vr, vr_act, rtol=1.0)

        # single quaternion, explicit axis
        vr_act = rotate_vectors(q[0], v, axis=1)
        np.testing.assert_allclose(vr, vr_act, rtol=1.0)

        # multiple quaternions, multiple vectors
        vr_act = rotate_vectors(q, v)
        np.testing.assert_allclose(vr, vr_act)

        # different axis
        vr_act = rotate_vectors(q, v.T, axis=0)
        np.testing.assert_allclose(vr.T, vr_act)

        # singleton expansion
        vr_act = rotate_vectors(q[:, None], v[None, ...])
        np.testing.assert_allclose(np.tile(vr, (10, 1, 1)), vr_act)

        # float dtype
        vr_act = rotate_vectors(as_float_array(q[0]), v[0])
        np.testing.assert_allclose(vr[0], vr_act, rtol=1.0)

        with pytest.raises(ValueError):
            rotate_vectors(q, v.T)

        with pytest.raises(ValueError):
            rotate_vectors(q, np.ones((10, 4)))

    def test_from_euler_angles(self):
        """"""
        rpy_r = (np.pi / 2, 0, 0)
        q_r = (np.sqrt(2) / 2, np.sqrt(2) / 2, 0, 0)
        q_rp = (0.5, 0.5, 0.5, 0.5)
        q_ypr = (np.sqrt(2) / 2, 0, np.sqrt(2) / 2, 0)

        # single rpy
        q = from_euler_angles(rpy_r)
        np.testing.assert_almost_equal(q, q_r)

        # rpy array
        q = from_euler_angles(np.tile(rpy_r, (10, 1)))
        np.testing.assert_almost_equal(q, np.tile(q_r, (10, 1)))

        # different axis
        q = from_euler_angles(np.tile(rpy_r, (10, 1)).T, axis=0)
        np.testing.assert_almost_equal(q, np.tile(q_r, (10, 1)).T)

        # single roll
        q = from_euler_angles(roll=np.pi / 2)
        np.testing.assert_almost_equal(q, q_r)

        # single roll, pitch & yaw
        q = from_euler_angles(roll=np.pi / 2, pitch=np.pi / 2)
        np.testing.assert_almost_equal(q, q_rp)

        # single roll, pitch, yaw + different order
        q = from_euler_angles(
            roll=np.pi / 2, pitch=np.pi / 2, yaw=np.pi / 2, order="ypr"
        )
        np.testing.assert_almost_equal(q, q_ypr)

        # no input
        with pytest.raises(ValueError):
            from_euler_angles()

        # rpy and roll together
        with pytest.raises(ValueError):
            from_euler_angles(rpy_r, roll=np.pi / 2)

        # inconsistent shapes
        with pytest.raises(ValueError):
            from_euler_angles(roll=np.ones(10), pitch=1)
