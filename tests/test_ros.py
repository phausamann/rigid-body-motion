import numpy as np
import pytest

import rigid_body_motion as rbm


class RospyPublisher:
    def __init__(
        self,
        name,
        data_class,
        subscriber_listener=None,
        tcp_nodelay=False,
        latch=False,
        headers=None,
        queue_size=None,
    ):
        """ Monkey patch drop in for rospy.Publisher. """
        self.name = name
        self.data_class = data_class
        self.subscriber_listener = subscriber_listener
        self.tcp_nodelay = tcp_nodelay
        self.latch = latch
        self.headers = headers
        self.queue_size = queue_size

        self._last_msg = None
        self._pub_count = 0

    def publish(self, msg):
        """"""
        self._last_msg = msg
        self._pub_count += 1


@pytest.fixture(autouse=True)
def patch_rospy_publisher(monkeypatch):
    """"""
    rospy = pytest.importorskip("rospy")
    monkeypatch.setattr(rospy, "Publisher", RospyPublisher)


# -- transformer module -- #
@pytest.fixture()
def Transformer():
    """"""
    tf = pytest.importorskip("rigid_body_motion.ros.transformer")
    return tf.Transformer


class TestTransformer:
    def test_moving_reference_frame(self, Transformer, head_dataset):
        """"""
        rbm.register_frame("world")
        rf_head = rbm.ReferenceFrame.from_dataset(
            head_dataset, "position", "orientation", "time", "world", "head",
        )

        Transformer.from_reference_frame(rf_head)

    def test_can_transform(self, Transformer, get_rf_tree):
        """"""
        rf_world, _, _ = get_rf_tree()
        transformer = Transformer.from_reference_frame(rf_world)
        assert transformer.can_transform("child1", "child2")

    def test_lookup_transform(self, rf_grid, Transformer, get_rf_tree):
        """"""
        r, rc1, rc2, t, tc1, tc2 = rf_grid
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        t_act, r_act = transformer.lookup_transform("child2", "child1")
        np.testing.assert_allclose(t_act, t)
        np.testing.assert_allclose(r_act, r)

    def test_transform_vector(self, transform_grid, Transformer, get_rf_tree):
        """"""
        o, ot, p, pt, rc1, rc2, tc1, tc2 = transform_grid
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        vt_act = transformer.transform_vector(p, "child2", "child1")
        v0t = transformer.transform_point((0.0, 0.0, 0.0), "child2", "child1")
        vt = np.array(pt) - np.array(v0t)
        # large relative differences at machine precision
        np.testing.assert_allclose(vt_act, vt, rtol=1.0)

    def test_transform_point(self, transform_grid, Transformer, get_rf_tree):
        """"""
        o, ot, p, pt, rc1, rc2, tc1, tc2 = transform_grid
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        pt_act = transformer.transform_point(p, "child2", "child1")
        np.testing.assert_allclose(pt_act, pt)

    def test_transform_quaternion(
        self, transform_grid, Transformer, get_rf_tree
    ):
        """"""
        o, ot, p, pt, rc1, rc2, tc1, tc2 = transform_grid
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        ot_act = transformer.transform_quaternion(o, "child2", "child1")
        # large relative differences at machine precision
        np.testing.assert_allclose(ot_act, ot, rtol=1.0)

    def test_transform_pose(self, transform_grid, Transformer, get_rf_tree):
        """"""
        o, ot, p, pt, rc1, rc2, tc1, tc2 = transform_grid
        rf_world, _, _ = get_rf_tree(tc1, rc1, tc2, rc2)
        transformer = Transformer.from_reference_frame(rf_world)
        pt_act, ot_act = transformer.transform_pose(p, o, "child2", "child1")
        np.testing.assert_allclose(pt_act, pt)
        # large relative differences at machine precision
        np.testing.assert_allclose(ot_act, ot, rtol=1.0)


@pytest.fixture()
def ReferenceFrameTransformBroadcaster():
    """"""
    tf = pytest.importorskip("rigid_body_motion.ros.transformer")
    return tf.ReferenceFrameTransformBroadcaster


class TestReferenceFrameTransformBroadcaster:
    def test_constructor(
        self,
        ReferenceFrameTransformBroadcaster,
        register_rf_tree,
        head_rf_tree,
    ):
        """"""
        # moving frame
        broadcaster = ReferenceFrameTransformBroadcaster(
            "head", publish_pose=True, publish_twist=True
        )
        assert broadcaster.frame.name == "head"
        assert broadcaster.base.name == "world"
        assert broadcaster.pose_publisher is not None
        assert broadcaster.twist_publisher is not None

        # static frame
        rbm.clear_registry()
        register_rf_tree()
        broadcaster = ReferenceFrameTransformBroadcaster("child1")
        assert broadcaster.frame.name == "child1"
        assert broadcaster.base.name == "world"

    def test_publish(
        self, ReferenceFrameTransformBroadcaster, head_rf_tree,
    ):
        """"""
        broadcaster = ReferenceFrameTransformBroadcaster(
            "head", publish_pose=True, publish_twist=True
        )
        broadcaster.publish()
        assert broadcaster.broadcaster.pub_tf._pub_count == 1
        assert broadcaster.pose_publisher._pub_count == 1
        assert broadcaster.twist_publisher._pub_count == 1


# -- visualization module -- #
@pytest.fixture()
def visualization():
    """"""
    return pytest.importorskip("rigid_body_motion.ros.visualization")


class TestVisualization:
    def test_hex_to_rgba(self, visualization):
        """"""
        color_msg = visualization.hex_to_rgba("#ffffffff")
        assert color_msg.r == 1.0
        assert color_msg.g == 1.0
        assert color_msg.b == 1.0
        assert color_msg.a == 1.0

    def test_get_marker(self, visualization):
        """"""
        marker_msg = visualization.get_marker()
        assert marker_msg.type == 4
        assert marker_msg.header.frame_id == "world"


class TestReferenceFrameMarkerPublisher:
    def test_constructor(self, visualization, head_rf_tree):
        """"""
        publisher = visualization.ReferenceFrameMarkerPublisher("head")
        assert publisher.topic == "/head/path"

    def test_publish(self, visualization, head_rf_tree):
        """"""
        publisher = visualization.ReferenceFrameMarkerPublisher("head")
        publisher.publish()
        assert publisher.publisher._pub_count == 1


# -- io module -- #
@pytest.fixture()
def RosbagReader():
    """"""
    io = pytest.importorskip("rigid_body_motion.ros.io")
    return io.RosbagReader


@pytest.fixture()
def RosbagWriter():
    """"""
    io = pytest.importorskip("rigid_body_motion.ros.io")
    return io.RosbagWriter


class TestRosbagReader:
    def test_get_msg_type(self, RosbagReader, rosbag_path):
        """"""
        import rosbag

        bag = rosbag.Bag(rosbag_path)

        assert (
            RosbagReader._get_msg_type(bag, "/camera/odom/sample")
            == "nav_msgs/Odometry"
        )
        assert (
            RosbagReader._get_msg_type(bag, "/vicon/t265_tracker/t265_tracker")
            == "geometry_msgs/TransformStamped"
        )

    def test_get_topics_and_types(self, RosbagReader, rosbag_path):
        """"""
        with RosbagReader(rosbag_path) as reader:
            info = reader.get_topics_and_types()

        assert info == {
            "/camera/accel/sample": "sensor_msgs/Imu",
            "/camera/gyro/sample": "sensor_msgs/Imu",
            "/camera/odom/sample": "nav_msgs/Odometry",
            "/vicon/t265_tracker/t265_tracker": "geometry_msgs/"
            "TransformStamped",
        }

    def test_load_msgs(self, RosbagReader, rosbag_path):
        """"""
        # odometry
        reader = RosbagReader(rosbag_path)
        with reader:
            odometry = reader.load_messages("/camera/odom/sample")

        assert set(odometry.keys()) == {
            "timestamps",
            "position",
            "orientation",
            "linear_velocity",
            "angular_velocity",
        }
        assert all(v.shape[0] == 228 for v in odometry.values())

        # pose
        with reader:
            pose = reader.load_messages("/vicon/t265_tracker/t265_tracker")

        assert set(pose.keys()) == {
            "timestamps",
            "position",
            "orientation",
        }
        assert all(v.shape[0] == 57 for v in pose.values())

    def test_load_dataset(self, RosbagReader, rosbag_path):
        """"""
        pytest.importorskip("xarray")

        reader = RosbagReader(rosbag_path)
        with reader:
            ds = reader.load_dataset("/camera/odom/sample", cache=True)

        assert (
            rosbag_path.parent / "cache" / "_camera_odom_sample.nc"
        ).exists()

        assert set(ds.data_vars) == {
            "position",
            "orientation",
            "linear_velocity",
            "angular_velocity",
        }

        assert set(ds.coords) == {
            "time",
            "cartesian_axis",
            "quaternion_axis",
        }

        assert ds.sizes == {
            "time": 228,
            "cartesian_axis": 3,
            "quaternion_axis": 4,
        }

        # regression test for using header timestamps if available
        assert np.mean(np.diff(ds.time)).astype(int) == 4998061

    def test_write_netcdf(self, RosbagReader, rosbag_path, export_folder):
        """"""
        pytest.importorskip("xarray")
        pytest.importorskip("netCDF4")

        output_file = export_folder / "test.nc"
        with RosbagReader(rosbag_path) as reader:
            reader.export("/camera/odom/sample", output_file)

        assert output_file.exists()


class TestRosbagWriter:
    def test_write_transform_stamped(self, RosbagWriter, tmpdir):
        """"""
        rosbag_path = tmpdir / "test.bag"

        with RosbagWriter(rosbag_path) as writer:
            writer.write_transform_stamped(
                np.arange(10),
                np.zeros((10, 3)),
                np.zeros((10, 4)),
                "/test",
                "world",
                "body",
            )

        assert rosbag_path.exists()

    def test_write_transform_stamped_dataset(
        self, RosbagWriter, head_dataset, tmpdir
    ):
        """"""
        rosbag_path = tmpdir / "test.bag"

        with RosbagWriter(rosbag_path) as writer:
            writer.write_transform_stamped_dataset(
                head_dataset.isel(time=range(100)), "/test", "world", "body",
            )

        assert rosbag_path.exists()


# -- utils module -- #
class MockPublisher:
    def __init__(self, n_msgs):
        """"""
        self.timestamps = np.arange(n_msgs).astype(np.datetime64)
        self.pub_count = 0

    def publish(self, idx=None):
        """"""
        self.pub_count += 1


@pytest.fixture()
def mock_publisher():
    """"""
    return MockPublisher(100)


class TestUtils:
    def test_play_publisher(self, mock_publisher):
        """"""
        utils = pytest.importorskip("rigid_body_motion.ros.utils")
        utils.play_publisher(mock_publisher)
