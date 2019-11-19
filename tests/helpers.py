import os

import pandas as pd
from quaternion import as_float_array, from_euler_angles

import rigid_body_motion as rbm

test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')


def load_csv(filepath):
    """"""
    df = pd.read_csv(filepath, header=[0, 1], index_col=0)
    l = [[tuple(r) for r in df[c].values] for c in df.columns.levels[0]]
    return list(zip(*l))


def rf_test_grid(step=1):
    """"""
    grid = load_csv(os.path.join(test_data_dir, 'rf_test_grid.csv'))
    return grid[::step]


def transform_test_grid(step=4):
    """"""
    grid = load_csv(os.path.join(test_data_dir, 'transform_test_grid.csv'))
    return grid[::step]


def mock_quaternion(*angles):
    """"""
    return as_float_array(from_euler_angles(*angles))


def get_rf_tree(tc1=(0., 0., 0.), rc1=(1., 0., 0., 0.),
                tc2=(0., 0., 0.), rc2=(1., 0., 0., 0.)):
    """"""
    rf_world = rbm.ReferenceFrame('world')
    rf_child1 = rbm.ReferenceFrame(
        'child1', parent=rf_world, translation=tc1, rotation=rc1)
    rf_child2 = rbm.ReferenceFrame(
        'child2', parent=rf_world, translation=tc2, rotation=rc2)

    return rf_world, rf_child1, rf_child2


def register_rf_tree(tc1=(0., 0., 0.), rc1=(1., 0., 0., 0.),
                     tc2=(0., 0., 0.), rc2=(1., 0., 0., 0.)):
    """"""
    rbm.register_frame('world')
    rbm.register_frame('child1', parent='world', translation=tc1, rotation=rc1)
    rbm.register_frame('child2', parent='world', translation=tc2, rotation=rc2)
