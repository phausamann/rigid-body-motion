from pathlib import Path

import numpy as np
import pandas as pd
import rosbag

from rigid_body_motion.ros.msg import (
    make_transform_msg,
    unpack_point_msg,
    unpack_quaternion_msg,
    unpack_vector_msg,
)


class RosbagReader:
    """ Reader for motion topics from rosbag files. """

    def __init__(self, bag_file):
        """ Constructor.

        Parameters
        ----------
        bag_file: str
            Path to rosbag file.
        """
        self.bag_file = Path(bag_file)

        self._bag = None

    def __enter__(self):
        self._bag = rosbag.Bag(self.bag_file, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._bag.close()
        self._bag = None

    @staticmethod
    def _get_msg_type(bag, topic):
        """ Get type of message. """
        return bag.get_type_and_topic_info(topic).topics[topic].msg_type

    def _get_filename(self, output_file, extension):
        """ Get export filename and create folder. """
        if output_file is None:
            folder, filename = self.bag_file.parent, self.bag_file.stem
            filename = folder / f"{filename}.{extension}"
        else:
            folder = output_file.parent
            filename = output_file

        folder.mkdir(parents=True, exist_ok=True)

        return filename

    @staticmethod
    def _write_netcdf(ds, filename, dtype="int32"):
        """ Write dataset to netCDF file. """
        comp = {
            "zlib": True,
            "dtype": dtype,
            "scale_factor": 0.0001,
            "_FillValue": np.iinfo(dtype).min,
        }

        encoding = {}
        for v in ds.data_vars:
            encoding[v] = comp

        ds.to_netcdf(filename, encoding=encoding)

    def get_topics_and_types(self):
        """ Get topics and corresponding message types included in rosbag.

        Returns
        -------
        topics: dict
            Names of topics and corresponding message types included in the
            rosbag.
        """
        if self._bag is None:
            raise RuntimeError(
                "get_topics must be called from within the RosbagReader "
                "context manager"
            )

        info = self._bag.get_type_and_topic_info()

        return {k: v[0] for k, v in info[1].items()}

    def load_messages(self, topic):
        """ Load messages from topic as dict.

        Only nav_msgs/Odometry and geometry_msgs/TransformStamped topics are
        supported so far.

        Parameters
        ----------
        topic: str
            Name of the topic to load.

        Returns
        -------
        messages: dict
            Dict containing arrays of timestamps and other message contents.
        """
        if self._bag is None:
            raise RuntimeError(
                "load_messages must be called from within the RosbagReader "
                "context manager"
            )

        msg_type = self._get_msg_type(self._bag, topic)

        if msg_type == "nav_msgs/Odometry":
            arr = np.array(
                [
                    (
                        (msg.header.stamp if msg._has_header else ts).to_sec(),
                        *unpack_point_msg(msg.pose.pose.position),
                        *unpack_quaternion_msg(msg.pose.pose.orientation),
                        *unpack_vector_msg(msg.twist.twist.linear),
                        *unpack_vector_msg(msg.twist.twist.angular),
                    )
                    for _, msg, ts in self._bag.read_messages(topics=topic)
                ]
            )
            return_vals = {
                "timestamps": arr[:, 0],
                "position": arr[:, 1:4],
                "orientation": arr[:, 4:8],
                "linear_velocity": arr[:, 8:11],
                "angular_velocity": arr[:, 11:],
            }

        elif msg_type == "geometry_msgs/TransformStamped":
            arr = np.array(
                [
                    (
                        (msg.header.stamp if msg._has_header else ts).to_sec(),
                        *unpack_point_msg(msg.transform.translation),
                        *unpack_quaternion_msg(msg.transform.rotation),
                    )
                    for _, msg, ts in self._bag.read_messages(topics=topic)
                ]
            )
            return_vals = {
                "timestamps": arr[:, 0],
                "position": arr[:, 1:4],
                "orientation": arr[:, 4:8],
            }

        else:
            raise ValueError(f"Unsupported message type {msg_type}")

        return return_vals

    def load_dataset(self, topic, cache=False):
        """ Load messages from topic as xarray.Dataset.

        Only nav_msgs/Odometry and geometry_msgs/TransformStamped topics are
        supported so far.

        Parameters
        ----------
        topic: str
            Name of the topic to load.

        cache: bool, default False
            If True, cache the dataset in ``cache/<topic>.nc`` in the same
            folder as the rosbag.

        Returns
        -------
        ds: xarray.Dataset
            Messages as dataset.
        """
        # TODO attrs
        import xarray as xr

        if cache:
            filepath = (
                self.bag_file.parent
                / "cache"
                / f"{topic.replace('/', '_')}.nc"
            )
            if not filepath.exists():
                self.export(topic, filepath)
            return xr.open_dataset(filepath)

        motion = self.load_messages(topic)

        coords = {
            "cartesian_axis": ["x", "y", "z"],
            "quaternion_axis": ["w", "x", "y", "z"],
            "time": pd.to_datetime(motion["timestamps"], unit="s"),
        }

        data_vars = {
            "position": (["time", "cartesian_axis"], motion["position"]),
            "orientation": (
                ["time", "quaternion_axis"],
                motion["orientation"],
            ),
        }

        if "linear_velocity" in motion:
            data_vars.update(
                (
                    {
                        "linear_velocity": (
                            ["time", "cartesian_axis"],
                            motion["linear_velocity"],
                        ),
                        "angular_velocity": (
                            ["time", "cartesian_axis"],
                            motion["angular_velocity"],
                        ),
                    }
                )
            )

        ds = xr.Dataset(data_vars, coords)

        return ds

    def export(self, topic, output_file=None):
        """ Export messages from topic as netCDF4 file.

        Parameters
        ----------
        topic: str
            Topic to read.

        output_file: str, optional
            Path to output file. By default, the path to the bag file, but with
            a different extension depending on the export format.
        """
        ds = self.load_dataset(topic, cache=False)
        self._write_netcdf(ds, self._get_filename(output_file, "nc"))


class RosbagWriter:
    """ Writer for motion topics to rosbag files. """

    def __init__(self, bag_file):
        """ Constructor.

        Parameters
        ----------
        bag_file: str
            Path to rosbag file.
        """
        self.bag_file = Path(bag_file)

        self._bag = None

    def __enter__(self):
        self._bag = rosbag.Bag(self.bag_file, "w")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._bag.close()
        self._bag = None

    def write_transform_stamped(
        self, timestamps, translation, rotation, topic, frame, child_frame
    ):
        """ Write multiple geometry_msgs/TransformStamped messages.

        Parameters
        ----------
        timestamps: array_like, shape (n_timestamps,)
            Array of timestamps.

        translation: array_like, shape (n_timestamps, 3)
            Array of translations.

        rotation: array_like, shape (n_timestamps, 4)
            Array of rotations.

        topic: str
            Topic of the messages.

        frame: str
            Parent frame of the transform.

        child_frame: str
            Child frame of the transform.
        """
        # check timestamps
        timestamps = np.asarray(timestamps)
        if timestamps.ndim != 1:
            raise ValueError("timestamps must be one-dimensional")

        # check translation
        translation = np.asarray(translation)
        if translation.shape != (len(timestamps), 3):
            raise ValueError(
                f"Translation must have shape ({len(timestamps)}, 3), "
                f"got {translation.shape}"
            )

        # check rotation
        rotation = np.asarray(rotation)
        if rotation.shape != (len(timestamps), 4):
            raise ValueError(
                f"Rotation must have shape ({len(timestamps)}, 4), "
                f"got {rotation.shape}"
            )

        # write messages to bag
        for ts, t, r in zip(timestamps, translation, rotation):
            msg = make_transform_msg(t, r, frame, child_frame, ts)
            self._bag.write(topic, msg)

    def write_transform_stamped_dataset(
        self,
        ds,
        topic,
        frame,
        child_frame,
        timestamps="time",
        translation="position",
        rotation="orientation",
    ):
        """ Write a dataset as geometry_msgs/TransformStamped messages.

        Parameters
        ----------
        ds: xarray.Dataset
            Dataset containing timestamps, translation and rotation

        topic: str
            Topic of the messages.

        frame: str
            Parent frame of the transform.

        child_frame: str
            Child frame of the transform.

        timestamps: str, default 'time'
            Name of the dimension containing the timestamps.

        translation: str, default 'position'
            Name of the variable containing the translation.

        rotation: str, default 'orientation'
            Name of the variable containing the rotation.
        """
        if np.issubdtype(ds[timestamps].dtype, np.datetime64):
            timestamps = ds[timestamps].astype(float) / 1e9
        else:
            timestamps = ds[timestamps]

        self.write_transform_stamped(
            timestamps,
            ds[translation],
            ds[rotation],
            topic,
            frame,
            child_frame,
        )
