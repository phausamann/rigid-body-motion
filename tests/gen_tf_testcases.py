""" Generate test cases for reference frame transforms.

This script requires a ROS environment in order to run properly.
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import rospy
import tf
from quaternion import as_float_array, from_euler_angles

import rigid_body_motion as rbm
from rigid_body_motion.ros import Transformer
from rigid_body_motion.ros.msg import static_rf_to_transform_msg

test_data_dir = Path(__file__).parent / "test_data"


def mock_quaternion(*angles):
    """"""
    return as_float_array(from_euler_angles(*angles))


def get_rf_tree(
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


def get_transformer(
    tc1=(0.0, 0.0, 0.0),
    rc1=(1.0, 0.0, 0.0, 0.0),
    tc2=(0.0, 0.0, 0.0),
    rc2=(1.0, 0.0, 0.0, 0.0),
):
    """"""
    rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
    return Transformer.from_reference_frame(rf_world)


def get_transformation(
    tc1=(0.0, 0.0, 0.0),
    rc1=(1.0, 0.0, 0.0, 0.0),
    tc2=(0.0, 0.0, 0.0),
    rc2=(1.0, 0.0, 0.0, 0.0),
):
    """"""
    transformer = get_transformer(tc1, rc1, tc2, rc2)
    t, r = transformer.lookup_transform("child2", "child1")

    index = pd.MultiIndex.from_tuples(
        list(product(("tc1", "tc2", "t"), ("x", "y", "z")))
        + list(product(("rc1", "rc2", "r"), ("w", "x", "y", "z"))),
        names=["var", "dim"],
    )

    return pd.Series(np.hstack((tc1, tc2, t, rc1, rc2, r)), index=index)


def transform_pose(
    tc1=(0.0, 0.0, 0.0),
    rc1=(1.0, 0.0, 0.0, 0.0),
    tc2=(0.0, 0.0, 0.0),
    rc2=(1.0, 0.0, 0.0, 0.0),
    p=(0.0, 0.0, 0.0),
    o=(1.0, 0.0, 0.0, 0.0),
):
    """"""
    transformer = get_transformer(tc1, rc1, tc2, rc2)
    pt, ot = transformer.transform_pose(p, o, "child2", "child1")

    index = pd.MultiIndex.from_tuples(
        list(product(("tc1", "tc2", "p", "pt"), ("x", "y", "z")))
        + list(product(("rc1", "rc2", "o", "ot"), ("w", "x", "y", "z"))),
        names=["var", "dim"],
    )

    return pd.Series(
        np.hstack((tc1, tc2, p, pt, rc1, rc2, o, ot)), index=index
    )


def get_twist(
    rf_world, transformer, t=(0.0, 0.0, 0.0), r=(1.0, 0.0, 0.0, 0.0), time=0.0
):
    """"""
    rf_child1 = rbm.ReferenceFrame(
        "child1", parent=rf_world, translation=t, rotation=r
    )

    transformer.setTransform(static_rf_to_transform_msg(rf_child1, time=time))

    try:
        v, w = transformer.lookupTwist(
            "child1", "world", rospy.Time(), rospy.Duration(0.01)
        )
    except tf.ExtrapolationException:
        v, w = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    index = pd.MultiIndex.from_tuples(
        list(product(("t", "v", "w"), ("x", "y", "z")))
        + ["rw", "rx", "ry", "rz"],
        names=["var", "dim"],
    )

    return pd.Series(np.hstack((t, v, w, r)), index=index)


if __name__ == "__main__":
    rg = np.arange(2)
    it = list(
        product(rg, rg, rg, rg * np.pi / 3, rg * np.pi / 3, rg * np.pi / 3)
    )

    # transformations
    df = pd.DataFrame(
        get_transformation(
            tc1=(x, y, z),
            tc2=(y, z, x),
            rc1=mock_quaternion(rx, ry, rz),
            rc2=mock_quaternion(ry, rz, rx),
        )
        for x, y, z, rx, ry, rz in it
    )

    df.to_csv(test_data_dir / "rf_test_grid.csv")

    # transformed poses
    df = pd.DataFrame(
        transform_pose(
            tc1=(x, y, z),
            tc2=(y, z, x),
            p=(z, x, y),
            rc1=mock_quaternion(rx, ry, rz),
            rc2=mock_quaternion(ry, rz, rx),
            o=mock_quaternion(rz, rx, ry),
        )
        for x, y, z, rx, ry, rz in it
    )

    df.to_csv(test_data_dir / "transform_test_grid.csv")

    # twist sequences
    twist_rg = np.linspace(0.0, 10.0, 100)
    it = list(
        product(twist_rg, (0.0,), (0.0,), rg * np.pi / 3, (0.0,), (0.0,))
    )
    times = np.arange(len(it))

    rf_world = rbm.ReferenceFrame("world")
    transformer = tf.TransformerROS(True)
    df = pd.DataFrame(
        get_twist(
            rf_world,
            transformer,
            t=(x, y, z),
            r=mock_quaternion(rx, ry, rz),
            time=time,
        )
        for (x, y, z, rx, ry, rz), time in zip(it, times)
    )
    df.index = pd.to_datetime(times, unit="s")

    df.to_csv(test_data_dir / "twist_test_grid.csv")
