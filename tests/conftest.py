import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from quaternion import as_float_array, from_euler_angles

import rigid_body_motion as rbm
from rigid_body_motion.testing import make_test_motion

test_data_dir = Path(__file__).parent / "test_data"


def load_csv(filepath):
    """"""
    df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    contents = [[tuple(r) for r in df[c].values] for c in df.columns.levels[0]]
    return list(zip(*contents))


@pytest.fixture(autouse=True)
def clear_registry():
    """"""
    rbm.clear_registry()
    yield


@pytest.fixture(params=load_csv(test_data_dir / "rf_test_grid.csv"))
def rf_grid(request):
    """"""
    return request.param


@pytest.fixture(params=load_csv(test_data_dir / "transform_test_grid.csv"))
def transform_grid(request):
    """"""
    return request.param


@pytest.fixture()
def mock_quaternion():
    def _mock_quaternion(*angles):
        """"""
        return as_float_array(from_euler_angles(*angles))

    return _mock_quaternion


@pytest.fixture()
def mock_frame():
    rbm.register_frame("world")

    def _mock_frame(
        t=None,
        r=None,
        ts=None,
        name="mock",
        parent="world",
        inverse=False,
        discrete=False,
    ):
        """"""
        if t is None and r is None:
            t = (0.0, 0.0, 0.0)
            r = (1.0, 0.0, 0.0, 0.0)

        return rbm.ReferenceFrame(name, parent, t, r, ts, inverse, discrete)

    yield _mock_frame

    rbm.clear_registry()


@pytest.fixture()
def get_rf_tree():
    def _get_rf_tree(
        tc1=(0.0, 0.0, 0.0),
        rc1=(1.0, 0.0, 0.0, 0.0),
        tc2=(0.0, 0.0, 0.0),
        rc2=(1.0, 0.0, 0.0, 0.0),
    ):
        """"""
        rf_world = rbm.ReferenceFrame("world")
        rf_child1 = rbm.ReferenceFrame(
            "child1", parent=rf_world, translation=tc1, rotation=rc1
        )
        rf_child2 = rbm.ReferenceFrame(
            "child2", parent=rf_world, translation=tc2, rotation=rc2
        )

        return rf_world, rf_child1, rf_child2

    return _get_rf_tree


@pytest.fixture()
def register_rf_tree():
    def _register_rf_tree(
        tc1=(0.0, 0.0, 0.0),
        rc1=(1.0, 0.0, 0.0, 0.0),
        tc2=(0.0, 0.0, 0.0),
        rc2=(1.0, 0.0, 0.0, 0.0),
    ):
        """"""
        rbm.register_frame("world")
        rbm.register_frame(
            "child1", parent="world", translation=tc1, rotation=rc1
        )
        rbm.register_frame(
            "child2", parent="world", translation=tc2, rotation=rc2
        )

    return _register_rf_tree


@pytest.fixture()
def compensated_tree():
    """"""
    n_samples = 10000
    stack = False

    rbm.register_frame("world", update=True)

    t, r, ts = make_test_motion(n_samples, stack=stack)
    rbm.ReferenceFrame(
        translation=t, rotation=r, timestamps=ts, parent="world", name="head",
    ).register(update=True)

    it, ir, _ = make_test_motion(
        n_samples, inverse=True, stack=stack, offset=(1.0, 0.0, 0.0)
    )
    rbm.ReferenceFrame(
        translation=it, rotation=ir, timestamps=ts, parent="head", name="eyes",
    ).register(update=True)


@pytest.fixture()
def icp_test_data():
    """"""
    return np.load(test_data_dir / "icp_test_data.npz")


@pytest.fixture()
def head_dataset():
    """"""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    pytest.importorskip("pooch")

    return xr.load_dataset(rbm.example_data["head"])


@pytest.fixture()
def left_eye_dataset():
    """"""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    pytest.importorskip("pooch")

    return xr.load_dataset(rbm.example_data["left_eye"])


@pytest.fixture()
def head_rf_tree(head_dataset):
    """"""
    rbm.register_frame("world")
    rbm.ReferenceFrame.from_dataset(
        head_dataset, "position", "orientation", "time", "world", "head",
    ).register()


@pytest.fixture()
def rosbag_path():
    """"""
    yield test_data_dir / "test.bag"
    shutil.rmtree(test_data_dir / "cache", ignore_errors=True)


@pytest.fixture()
def optitrack_path():
    """"""
    yield test_data_dir / "optitrack.csv"
    shutil.rmtree(test_data_dir / "cache", ignore_errors=True)


@pytest.fixture()
def export_folder():
    """"""
    export_folder = test_data_dir / "exports"
    yield export_folder
    shutil.rmtree(export_folder, ignore_errors=True)
