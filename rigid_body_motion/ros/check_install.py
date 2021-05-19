import traceback

if __name__ == "__main__":

    try:
        import geometry_msgs.msg  # noqa
        import rospy  # noqa
        import std_msgs.msg  # noqa
        import visualization_msgs.msg  # noqa

        try:
            import rospkg  # noqa
            import tf2_geometry_msgs  # noqa
            import tf2_ros  # noqa
            from tf.msg import tfMessage  # noqa
        except rospkg.ResourceNotFound:
            raise ImportError(
                "The rospkg module was found but tf2_ros failed to import, "
                "make sure you've set up the necessary environment variables"
            )

    except ImportError:
        print(
            f"Some dependencies are not correctly installed. "
            f"See the traceback below for more info.\n\n"
            f"{traceback.format_exc()}"
        )

    else:
        print("All dependencies correctly installed!")
