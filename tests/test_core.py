import numpy as np
import pytest
from numpy import testing as npt

from rigid_body_motion.core import (
    _make_dataarray,
    _maybe_unpack_dataarray,
    _replace_dim,
    _resolve_axis,
)


class TestCore:
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

    def test_replace_dim(self):
        """"""
        xr = pytest.importorskip("xarray")

        da = xr.DataArray(
            np.ones((10, 3)),
            {
                "time": np.arange(10),
                "old_dim": ["a", "b", "c"],
                "old_dim_coord": ("old_dim", np.arange(3)),
            },
            ("time", "old_dim"),
        )
        dims = da.dims
        coords = dict(da.coords)

        new_coords, new_dims = _replace_dim(coords, dims, -1, "cartesian", 2)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["cartesian_axis"] == ["x", "y"]
        assert new_dims == ("time", "cartesian_axis")

        new_coords, new_dims = _replace_dim(coords, dims, -1, "polar", 2)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["polar_axis"] == ["r", "phi"]
        assert new_dims == ("time", "polar_axis")

        new_coords, new_dims = _replace_dim(coords, dims, -1, "cartesian", 3)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["cartesian_axis"] == ["x", "y", "z"]
        assert new_dims == ("time", "cartesian_axis")

        new_coords, new_dims = _replace_dim(coords, dims, -1, "spherical", 3)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["spherical_axis"] == ["r", "theta", "phi"]
        assert new_dims == ("time", "spherical_axis")

        new_coords, new_dims = _replace_dim(coords, dims, -1, "quaternion", 3)
        assert set(new_coords.keys()) == set(new_dims)
        assert new_coords["quaternion_axis"] == ["w", "x", "y", "z"]
        assert new_dims == ("time", "quaternion_axis")

    def test_maybe_unpack_dataarray(self):
        """"""
        xr = pytest.importorskip("xarray")

        # ndarray
        arr = np.ones((10, 3))
        (
            arr_out,
            axis,
            dim,
            time_axis,
            time_dim,
            timestamps,
            coords,
            dims,
            name,
            attrs,
        ) = _maybe_unpack_dataarray(arr)
        assert arr_out is arr
        assert axis == -1
        assert dim is None
        assert time_axis == 0
        assert time_dim is None
        assert timestamps is None
        assert coords is None
        assert dims is None
        assert name is None
        assert attrs is None

        # dim argument with ndarray
        with pytest.raises(ValueError):
            _maybe_unpack_dataarray(arr, dim="cartesian_axis")

        # DataArray
        da = xr.DataArray(
            arr,
            {"time": np.arange(10)},
            ("time", "cartesian_axis"),
            "da",
            {"test_attr": 0},
        )
        (
            arr_out,
            axis,
            dim,
            time_axis,
            time_dim,
            timestamps,
            coords,
            dims,
            name,
            attrs,
        ) = _maybe_unpack_dataarray(da, dim="cartesian_axis")
        npt.assert_equal(arr_out, arr)
        assert axis == 1
        assert dim == "cartesian_axis"
        assert time_axis == 0
        assert time_dim == "time"
        npt.assert_equal(timestamps, np.arange(10))
        assert all(c in da.coords for c in coords)
        assert dims is da.dims
        assert name is da.name
        assert attrs is not da.attrs
        assert attrs == da.attrs

        # static DataArray
        da_static = xr.DataArray(arr[0], dims=("cartesian_axis",))
        (
            arr_out,
            axis,
            dim,
            time_axis,
            time_dim,
            timestamps,
            coords,
            dims,
            name,
            attrs,
        ) = _maybe_unpack_dataarray(da_static, dim="cartesian_axis")
        assert axis == 0
        assert dim == "cartesian_axis"
        assert time_axis is None
        assert time_dim is None
        assert timestamps is None

        # multi-dimensional static DataArray
        (
            arr_out,
            axis,
            dim,
            time_axis,
            time_dim,
            timestamps,
            coords,
            dims,
            name,
            attrs,
        ) = _maybe_unpack_dataarray(da, timestamps=False)
        assert axis == -1
        assert dim == "cartesian_axis"
        assert time_axis is None
        assert time_dim is None
        assert timestamps is None

        # dim and axis argument at the same time
        with pytest.raises(ValueError):
            _maybe_unpack_dataarray(da, dim="cartesian_axis", axis=-1)

    def test_make_dataarray(self):
        """"""
        xr = pytest.importorskip("xarray")
        arr = np.ones((10, 3))

        # no input timestamps
        da_out = _make_dataarray(
            arr,
            {},
            ("cartesian_axis",),
            "da",
            {"test_attr": 0},
            None,
            np.arange(10),
        )
        npt.assert_equal(da_out, arr)
        assert da_out.dims == ("time", "cartesian_axis")
        npt.assert_equal(da_out.coords["time"], np.arange(10))
        assert da_out.name == "da"
        assert da_out.attrs["test_attr"] == 0

        # timestamps from coord
        da_in = xr.DataArray(
            arr,
            dims=("time", "cartesian_axis"),
            coords={
                "time": np.arange(10),
                "test_coord": ("time", 2 * np.arange(10)),
            },
        )
        da_out = _make_dataarray(
            arr[:5],
            dict(da_in.coords),
            da_in.dims,
            None,
            None,
            "time",
            np.arange(5) + 2.5,
        )
        npt.assert_allclose(da_out, arr[:5])
        assert da_out.dims == ("time", "cartesian_axis")
        npt.assert_equal(da_out.coords["time"], np.arange(5) + 2.5)
        npt.assert_allclose(da_out.coords["test_coord"], 2 * np.arange(5) + 5)

        # invalid coord name
        with pytest.raises(ValueError):
            _make_dataarray(
                arr,
                dict(da_in.coords),
                da_in.dims,
                None,
                None,
                "not_a_coord",
                None,
            )

        # non-numeric coord
        da_in = xr.DataArray(
            arr,
            dims=("time", "cartesian_axis"),
            coords={
                "time": np.arange(10),
                "test_coord": ("time", ["A"] * 5 + ["B"] * 5),
            },
        )
        da_out = _make_dataarray(
            arr[:5],
            dict(da_in.coords),
            da_in.dims,
            None,
            None,
            "time",
            np.arange(5) + 2.5,
        )
        npt.assert_allclose(da_out, arr[:5])
        assert da_out.dims == ("time", "cartesian_axis")
        npt.assert_equal(da_out.coords["time"], np.arange(5) + 2.5)
        npt.assert_equal(
            da_out.coords["test_coord"], np.array(["A", "A", "B", "B", "B"])
        )
