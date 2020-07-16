from pathlib import Path

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
    n_samples = 1000

    rbm.register_frame("world", update=True)

    t, r, ts = make_test_motion(n_samples, stack=False)
    rbm.ReferenceFrame(
        translation=t, rotation=r, timestamps=ts, parent="world", name="head",
    ).register(update=True)

    it, ir, _ = make_test_motion(n_samples, inverse=True, stack=False)
    rbm.ReferenceFrame(
        translation=it, rotation=ir, timestamps=ts, parent="head", name="eyes",
    ).register(update=True)


@pytest.fixture()
def head_dataset():
    """"""
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")

    return xr.load_dataset(rbm.example_data["head"])
