from pathlib import Path

import numpy as np
import pandas as pd
import rosbag

from rigid_body_motion.ros.msg import (
    unpack_point_msg,
    unpack_quaternion_msg,
    unpack_vector_msg,
)


class RosbagReader:
    """ Reader for motion topics from rosbag files. """

    def __init__(self, bag_file, topic):
        """ Constructor.

        Parameters
        ----------
        bag_file: str
            Path to rosbag file.

        topic: str
            Topic to export.

        output_file: str, optional
            Path to output file. By default, the path to the bag file, but with
            a different extension depending on the export format.
        """
        self.bag_file = Path(bag_file)
        self.topic = topic

        self.msg_type = self._get_msg_type(self.bag_file, self.topic)

    @staticmethod
    def _get_msg_type(bag_file, topic):
        """ Get type of message. """
        with rosbag.Bag(bag_file, "r") as bag:
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

    def load_messages(self):
        """ Load messages from topic as dict.

        Returns
        -------
        messages: dict
            Dict containing timestamp arrays

        """
        if self.msg_type == "nav_msgs/Odometry":
            with rosbag.Bag(self.bag_file, "r") as bag:
                arr = np.array(
                    [
                        (
                            ts.to_sec(),
                            *unpack_point_msg(msg.pose.pose.position),
                            *unpack_quaternion_msg(msg.pose.pose.orientation),
                            *unpack_vector_msg(msg.twist.twist.linear),
                            *unpack_vector_msg(msg.twist.twist.angular),
                        )
                        for _, msg, ts in bag.read_messages(topics=self.topic)
                    ]
                )
            return_vals = {
                "timestamps": arr[:, 0],
                "position": arr[:, 1:4],
                "orientation": arr[:, 4:8],
                "linear_velocity": arr[:, 8:11],
                "angular_velocity": arr[:, 11:],
            }

        elif self.msg_type == "geometry_msgs/TransformStamped":
            with rosbag.Bag(self.bag_file, "r") as bag:
                arr = np.array(
                    [
                        (
                            ts.to_sec(),
                            *unpack_point_msg(msg.transform.translation),
                            *unpack_quaternion_msg(msg.transform.rotation),
                        )
                        for _, msg, ts in bag.read_messages(topics=self.topic)
                    ]
                )
                return_vals = {
                    "timestamps": arr[:, 0],
                    "position": arr[:, 1:4],
                    "orientation": arr[:, 4:8],
                }

        else:
            raise ValueError(f"Unsupported message type {self.msg_type}")

        return return_vals

    def load_dataset(self):
        """ Load messages from topic as xarray.Dataset.

        Returns
        -------
        ds: xarray.Dataset
            Messages as dataset.
        """
        # TODO attrs
        import xarray as xr

        motion = self.load_messages()

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

    def export(self, output_file=None):
        """ Export messages from topic as netCDF4 file. """
        ds = self.load_dataset()
        self._write_netcdf(ds, self._get_filename(output_file, "nc"))
