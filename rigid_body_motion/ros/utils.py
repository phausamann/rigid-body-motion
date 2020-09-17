""""""
import numpy as np


def play_publisher(publisher, step=1, speed=1.0, skip=None, timestamps=None):
    """ Interactive widget for playing back messages from a publisher.

    Parameters
    ----------
    publisher: object
        Any object with a ``publish`` method that accepts an ``idx`` parameter
        and publishes a message corresponding to that index.

    step: int, default 1
        Difference in indexes between consecutive messages, e.g. if ``step=2``
        every second message will be published.

    speed: float, default 1.0
        Playback speed.

    skip: int, optional
        Number of messages to skip with the forward and backward buttons.

    timestamps: array_like, datetime64 dtype, optional
        Timestamps of publisher messages that determine time difference between
        messages and total number of messages. The time difference is
        calculated as the mean difference between the timestamps, i.e. it
        assumes that the timestamps are more or less regular. If not provided,
        the publisher must have a ``timestamps`` attribute which will be used
        instead.
    """
    from IPython.core.display import display
    from ipywidgets import widgets

    if timestamps is None:
        timestamps = np.asarray(publisher.timestamps)

    interval = np.mean(np.diff(timestamps.astype(float) / 1e6)) / speed

    # position bar
    s_idx = widgets.IntSlider(
        min=0, max=len(timestamps) - 1, value=0, description="Index"
    )

    # forward button
    def button_plus(name):
        s_idx.value += skip or step if s_idx.value < s_idx.max else 0

    forward = widgets.Button(
        description="►►", layout=widgets.Layout(width="50px")
    )
    forward.on_click(button_plus)

    # backward button
    def button_minus(name):
        s_idx.value -= skip or step if s_idx.value < s_idx.max else 0

    backward = widgets.Button(
        description="◄◄", layout=widgets.Layout(width="50px")
    )
    backward.on_click(button_minus)

    # play button
    play = widgets.Play(
        interval=int(interval * step),
        value=0,
        min=s_idx.min,
        max=s_idx.max,
        step=step,
        description="Press play",
        disabled=False,
    )
    widgets.jslink((play, "value"), (s_idx, "value"))

    # layout
    ui = widgets.HBox([s_idx, backward, play, forward])
    out = widgets.interactive_output(publisher.publish, {"idx": s_idx})
    display(ui, out)
