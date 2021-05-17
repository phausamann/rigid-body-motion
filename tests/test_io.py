import numpy as np
import pytest

from rigid_body_motion.io import load_optitrack


class TestIO:
    def test_load_optitrack(self, optitrack_path):
        """"""
        pytest.importorskip("pandas")

        # numpy
        data = load_optitrack(optitrack_path)
        assert set(data.keys()) == {"RigidBody 01", "RigidBody 02"}
        assert data["RigidBody 01"][0].shape == (227, 3)
        assert data["RigidBody 01"][1].shape == (227, 4)
        assert data["RigidBody 01"][2].shape == (227,)
        assert np.issubdtype(data["RigidBody 01"][2].dtype, np.datetime64)

        # pandas
        data = load_optitrack(optitrack_path, "pandas")
        assert set(data.keys()) == {"RigidBody 01", "RigidBody 02"}
        assert data["RigidBody 01"].shape == (227, 7)

    def test_load_optitrack_xr(self, optitrack_path):
        """"""
        pytest.importorskip("pandas")
        pytest.importorskip("xarray")

        data = load_optitrack(optitrack_path, "xarray")
        assert set(data.keys()) == {"RigidBody 01", "RigidBody 02"}
        assert data["RigidBody 01"].position.shape == (227, 3)
        assert data["RigidBody 01"].orientation.shape == (227, 4)
