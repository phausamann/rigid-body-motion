import pytest

import rigid_body_motion  # noqa

xr = pytest.importorskip("xarray")


class TestDataArrayAccessor:
    def test_qinterp(self, head_dataset, left_eye_dataset):
        """"""
        da = head_dataset.orientation.rbm.qinterp(time=left_eye_dataset.time)

        assert da.shape == (left_eye_dataset.sizes["time"], 4)
        xr.testing.assert_equal(da.time, left_eye_dataset.time)

        with pytest.raises(ValueError):
            head_dataset.orientation.rbm.qinterp(
                not_a_dim=left_eye_dataset.time
            )

        with pytest.raises(ValueError):
            head_dataset.position.rbm.qinterp(time=left_eye_dataset.time)

        with pytest.raises(NotImplementedError):
            head_dataset.orientation.rbm.qinterp(
                time=left_eye_dataset.time, other_dim=range(10)
            )
