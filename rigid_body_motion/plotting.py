import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    """ Colored arrows representing coordinate system. """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """ Constructor. """
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """ Draw to the given renderer. """
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def _add_frame(ax, frame, world_frame=None, arrow_len=1.0):
    """ Add coordinates representing a reference frame. """
    from rigid_body_motion import transform_points

    o = [0.0, 0.0, 0.0]
    x = [arrow_len, 0.0, 0.0]
    y = [0.0, arrow_len, 0.0]
    z = [0.0, 0.0, arrow_len]

    if world_frame is not None:
        o = transform_points(o, outof=frame, into=world_frame)
        x = transform_points(x, outof=frame, into=world_frame)
        y = transform_points(y, outof=frame, into=world_frame)
        z = transform_points(z, outof=frame, into=world_frame)

    arrow_prop_dict = dict(
        mutation_scale=20, arrowstyle="->", shrinkA=0, shrinkB=0
    )
    x_arrow = Arrow3D(
        [o[0], x[0]], [o[1], x[1]], [o[2], x[2]], **arrow_prop_dict, color="r"
    )
    ax.add_artist(x_arrow)
    y_arrow = Arrow3D(
        [o[0], y[0]], [o[1], y[1]], [o[2], y[2]], **arrow_prop_dict, color="g"
    )
    ax.add_artist(y_arrow)
    z_arrow = Arrow3D(
        [o[0], z[0]], [o[1], z[1]], [o[2], z[2]], **arrow_prop_dict, color="b"
    )
    ax.add_artist(z_arrow)

    # manually update axis limits
    x_lim_old = ax.get_xlim3d()
    y_lim_old = ax.get_ylim3d()
    z_lim_old = ax.get_zlim3d()
    x_lim_new = [
        np.min((x_lim_old[0], o[0], x[0], y[0], z[0])),
        np.max((x_lim_old[1], o[0], x[0], y[0], z[0])),
    ]
    y_lim_new = [
        np.min((y_lim_old[0], o[1], x[1], y[1], z[1])),
        np.max((y_lim_old[1], o[1], x[1], y[1], z[1])),
    ]
    z_lim_new = [
        np.min((z_lim_old[0], o[2], x[2], y[2], z[2])),
        np.max((z_lim_old[1], o[2], x[2], y[2], z[2])),
    ]
    ax.set_xlim3d(x_lim_new)
    ax.set_ylim3d(y_lim_new)
    ax.set_ylim3d(z_lim_new)

    return [x_arrow, y_arrow, z_arrow]


def _set_axes_equal(ax):
    """ Make axes of 3D plot have equal scale.

    from https://stackoverflow.com/a/31364297
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_reference_frame(
    frame, world_frame=None, ax=None, figsize=(6, 6), arrow_len=1.0
):
    """ Plot a 3D coordinate system representing a static reference frame.

    Parameters
    ----------
    frame: str or ReferenceFrame
        The reference frame to plot. If str, the frame will be looked up
        in the registry under that name.

    world_frame: str or ReferenceFrame, optional
        If specified, the world reference frame that defines the origin of the
        plot. If str, the frame will be looked up in the registry under that
        name.

    ax: matplotlib.axes.Axes instance, optional
        If provided, plot the points onto these axes.

    figsize:
        If `ax` is not provided, create a figure of this size.

    arrow_len:
        Length of the arrows of the coordinate system.

    Returns
    -------
    arrows: list of Arrow3D
        A list of three arrows representing the plotted coordinate system.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    if frame.timestamps is not None:
        raise NotImplementedError("Can only plot static reference frames")

    arrows = _add_frame(ax, frame, world_frame, arrow_len=arrow_len)

    _set_axes_equal(ax)

    return arrows


def plot_points(arr, ax=None, figsize=(6, 6), fmt=None, **kwargs):
    """ Plot an array of 3D points.

    Parameters
    ----------
    arr: array_like, shape (N, 3)
        Array of 3D points to plot.

    ax: matplotlib.axes.Axes instance, optional
        If provided, plot the points onto these axes.

    figsize:
        If `ax` is not provided, create a figure of this size.

    fmt: str, optional
        A format string, e.g. 'ro' for red circles.

    kwargs:
        Additional keyword arguments passed to ax.plot().

    Returns
    -------
    lines: list of Line3D
        A list of lines representing the plotted data.
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

    lines = ax.plot(arr[:, 0], arr[:, 1], arr[:, 2], fmt, **kwargs)

    _set_axes_equal(ax)

    return lines
