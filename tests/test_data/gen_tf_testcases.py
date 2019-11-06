""" Generate test cases for reference frame transforms.

This script requires a Python 2.7 ROS enviroment in order to run properly.
"""
from __future__ import print_function

import os

import numpy as np
import pandas as pd

import rospy
import tf
from geometry_msgs.msg import PoseStamped, TransformStamped, Vector3, \
    Point, Quaternion

import rigid_body_motion as rbm
from quaternion import from_euler_angles, as_float_array
from itertools import product


test_data_dir = os.path.dirname(os.path.realpath(__file__))


def mock_quaternion(*angles):
    """"""
    return as_float_array(from_euler_angles(*angles))


def rf_to_transform_msg(rf):
    """"""
    msg = TransformStamped()
    msg.header.frame_id = rf.parent.name
    msg.child_frame_id = rf.name
    msg.transform.translation = Vector3(*rf.translation)
    r = rf.rotation
    msg.transform.rotation = Quaternion(r[1], r[2], r[3], r[0])

    return msg


def make_pose_msg(p, o, frame_id):
    """"""
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position = Point(*p)
    msg.pose.orientation = Quaternion(o[1], o[2], o[3], o[0])

    return msg


def get_transformation(tc1=(0., 0., 0.), rc1=(1., 0., 0., 0.),
                       tc2=(0., 0., 0.), rc2=(1., 0., 0., 0.)):
    """"""
    rf_world = rbm.ReferenceFrame('world')
    rf_child1 = rbm.ReferenceFrame(
        'child1', parent=rf_world, translation=tc1, rotation=rc1)
    rf_child2 = rbm.ReferenceFrame(
        'child2', parent=rf_world, translation=tc2, rotation=rc2)

    transformer = tf.TransformerROS(True, rospy.Duration(10.0))
    transformer.setTransform(rf_to_transform_msg(rf_child1))
    transformer.setTransform(rf_to_transform_msg(rf_child2))

    t, r = transformer.lookupTransform('child2', 'child1', rospy.Time(0))
    t = tuple(t)
    r = (r[3], r[0], r[1], r[2])

    index = pd.MultiIndex.from_tuples(
        list(product(('tc1', 'tc2', 't'), ('x', 'y', 'z'))) +
        list(product(('rc1', 'rc2', 'r'), ('w', 'x', 'y', 'z'))),
        names=['var', 'dim'])

    return pd.Series(np.hstack((tc1, tc2, t, rc1, rc2, r)), index=index)


def transform_pose(tc1=(0., 0., 0.), rc1=(1., 0., 0., 0.),
                   tc2=(0., 0., 0.), rc2=(1., 0., 0., 0.),
                   p=(0., 0., 0.), o=(1., 0., 0., 0.)):
    """"""
    rf_world = rbm.ReferenceFrame('world')
    rf_child1 = rbm.ReferenceFrame(
        'child1', parent=rf_world, translation=tc1, rotation=rc1)
    rf_child2 = rbm.ReferenceFrame(
        'child2', parent=rf_world, translation=tc2, rotation=rc2)

    transformer = tf.TransformerROS(True, rospy.Duration(10.0))
    transformer.setTransform(rf_to_transform_msg(rf_child1))
    transformer.setTransform(rf_to_transform_msg(rf_child2))

    pose = make_pose_msg(p, o, 'child1')
    pose_t = transformer.transformPose('child2', pose)
    pt = pose_t.pose.position
    pt = (pt.x, pt.y, pt.z)
    ot = pose_t.pose.orientation
    ot = (ot.w, ot.x, ot.y, ot.z)

    index = pd.MultiIndex.from_tuples(
        list(product(('tc1', 'tc2', 'p', 'pt'), ('x', 'y', 'z'))) +
        list(product(('rc1', 'rc2', 'o', 'ot'), ('w', 'x', 'y', 'z'))),
        names=['var', 'dim'])

    return pd.Series(np.hstack((tc1, tc2, p, pt, rc1, rc2, o, ot)),
                     index=index)


if __name__ == '__main__':

    r = np.arange(2)
    it = product(r, r, r, r*np.pi/3, r*np.pi/3, r*np.pi/3)

    df = pd.DataFrame(get_transformation(tc1=(x, y, z), tc2=(y, z, x),
                                         rc1=mock_quaternion(rx, ry, rz),
                                         rc2=mock_quaternion(ry, rz, rx))
                      for x, y, z, rx, ry, rz in it)

    df.to_csv(os.path.join(test_data_dir, 'rf_test_grid.csv'))

    it = product(r, r, r, r*np.pi/3, r*np.pi/3, r*np.pi/3)

    df = pd.DataFrame(transform_pose(tc1=(x, y, z), tc2=(y, z, x), p=(z, x, y),
                                     rc1=mock_quaternion(rx, ry, rz),
                                     rc2=mock_quaternion(ry, rz, rx),
                                     o=mock_quaternion(rz, rx, ry))
                      for x, y, z, rx, ry, rz in it)

    df.to_csv(os.path.join(test_data_dir, 'transform_test_grid.csv'))
