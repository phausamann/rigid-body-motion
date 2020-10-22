import numpy as np
import pytest
from numpy import testing as npt
from quaternion import (
    as_float_array,
    as_quat_array,
    from_euler_angles,
    quaternion,
)

from rigid_body_motion.utils import qinv, qmean, qmul, rotate_vectors


class TestUtils(object):
    def test_qinv(self):
        """"""
        q = from_euler_angles(0.0, 0.0, np.pi / 4)

        assert qinv(q) == 1 / q
        npt.assert_equal(qinv(as_float_array(q)), as_float_array(1 / q))

        q_arr = np.tile(as_float_array(q), (10, 1)).T
        npt.assert_equal(qinv(q_arr, 0)[:, 0], as_float_array(1 / q))

    def test_qmul(self):
        """"""
        q1 = from_euler_angles(0.0, 0.0, np.pi / 4)
        q2 = from_euler_angles(0.0, 0.0, np.pi / 4)

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
                from_euler_angles(0.0, 0.0, np.pi / 4),
                from_euler_angles(0.0, 0.0, -np.pi / 4),
                from_euler_angles(0.0, np.pi / 4, 0.0),
                from_euler_angles(0.0, -np.pi / 4, 0.0),
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

    def test_rotate_vectors(self):
        """"""
        v = np.ones((10, 3))
        q = np.tile(from_euler_angles(0.0, 0.0, np.pi / 4), 10)
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
