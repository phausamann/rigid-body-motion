import pytest
from numpy import testing as npt

import numpy as np
import xarray as xr

from quaternion import quaternion, as_float_array, from_euler_angles

import rigid_body_motion as rbm
from rigid_body_motion.utils import \
    _resolve_axis, rotate_vectors, _maybe_unpack_dataarray, _make_dataarray


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

    def test_maybe_unpack_datarray(self):
        """"""
        # ndarray
        arr = np.ones((10, 3))
        arr_out, axis, timestamps, coords, dims = _maybe_unpack_dataarray(arr)
        assert arr_out is arr
        assert axis == -1
        assert timestamps is None
        assert coords is None

        # dim argument with ndarray
        with pytest.raises(ValueError):
            _maybe_unpack_dataarray(arr, dim='cartesian_axis')

        # DataArray
        da = xr.DataArray(
            arr, {'time': np.arange(10)}, ('time', 'cartesian_axis'))
        arr_out, axis, timestamps, coords, dims = _maybe_unpack_dataarray(
            da, dim='cartesian_axis', timestamps='time')
        npt.assert_equal(arr_out, arr)
        assert axis == 1
        npt.assert_equal(timestamps, np.arange(10))
        assert all(c in da.coords for c in coords)
        assert dims is da.dims

        # static DataArray
        da = xr.DataArray(arr[0], dims=('cartesian_axis',))
        arr_out, axis, timestamps, coords, dims = _maybe_unpack_dataarray(
            da, dim='cartesian_axis', timestamps=None)
        npt.assert_equal(arr_out, arr[0])
        assert axis == 0
        assert timestamps is None
        assert all(c in da.coords for c in coords)
        assert dims is da.dims

        # dim and axis argument
        with pytest.raises(ValueError):
            _maybe_unpack_dataarray(da, dim='cartesian_axis', axis=-1)

    def test_make_dataarray(self):
        """"""
        arr = np.ones((10, 3))

        # no input timestamps
        da_out = _make_dataarray(arr, {}, ('cartesian_axis',), None,
                                 np.arange(10))
        npt.assert_equal(da_out, arr)
        assert da_out.dims == ('time', 'cartesian_axis')
        npt.assert_equal(da_out.coords['time'], np.arange(10))

        # timestamps from coord
        da_in = xr.DataArray(
            arr, dims=('time', 'cartesian_axis'),
            coords={'time': np.arange(10),
                    'test_coord': ('time', 2*np.arange(10))})
        da_out = _make_dataarray(arr[:5], dict(da_in.coords), da_in.dims,
                                 'time', np.arange(5) + 2.5)
        npt.assert_allclose(da_out, arr[:5])
        assert da_out.dims == ('time', 'cartesian_axis')
        npt.assert_equal(da_out.coords['time'], np.arange(5)+2.5)
        npt.assert_allclose(da_out.coords['test_coord'], 2*np.arange(5)+5)
