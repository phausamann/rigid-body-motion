import numpy as np
import pytest

import rigid_body_motion as rbm

plt = pytest.importorskip("matplotlib.pyplot")


class TestPlotting:
    def test_plot_reference_frame(self):
        rf_world = rbm.ReferenceFrame("world")
        rf_observer = rbm.ReferenceFrame(
            "observer",
            parent=rf_world,
            translation=(5, 0, 0),
            rotation=(np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2),
        )

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        rbm.plot.reference_frame(rf_world, ax=ax)
        rbm.plot.reference_frame(rf_observer, ax=ax)

        fig.tight_layout()
        plt.show()
