import numpy as np
import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

from .utils import qinterp, qinv


@xr.register_dataarray_accessor("rbm")
class DataArrayAccessor:
    """ Accessor for DataArrays. """

    def __init__(self, obj):
        """ Constructor. """
        self._obj = obj

    def qinterp(self, coords=None, qdim="quaternion_axis", **coords_kwargs):
        """ Quaternion interpolation.

        Parameters
        ----------
        coords: dict, optional
            Mapping from dimension names to the new coordinates.
            New coordinate can be a scalar, array-like or DataArray.

        qdim: str, default "quaternion_axis"
            Name of the dimension representing the quaternions.

        **coords_kwargs : {dim: coordinate, ...}, optional
            The keyword arguments form of ``coords``.
            One of coords or coords_kwargs must be provided.

        Returns
        -------
        interpolated: xr.DataArray
            New array on the new coordinates.

        Examples
        --------
        >>> import xarray as xr
        >>> import rigid_body_motion as rbm
        >>> ds_head = xr.load_dataset(rbm.example_data["head"])
        >>> ds_left_eye = xr.load_dataset(rbm.example_data["left_eye"])
        >>> ds_head.orientation.rbm.qinterp(time=ds_left_eye.time) # doctest:+ELLIPSIS
        <xarray.DataArray 'orientation' (time: 113373, quaternion_axis: 4)>
        array(...)
        Coordinates:
          * time             (time) datetime64[ns] ...
          * quaternion_axis  (quaternion_axis) object 'w' 'x' 'y' 'z'
        Attributes:
            long_name:  Orientation
        """  # noqa
        coords = either_dict_or_kwargs(coords, coords_kwargs, "interp")

        if len(coords) != 1:
            raise NotImplementedError(
                "qinterp only works along a single dimension so far"
            )

        interp_dim = next(iter(coords))
        if interp_dim not in self._obj.dims:
            raise ValueError(
                f"{interp_dim} is not a dimension of this DataArray"
            )

        if np.asanyarray(coords[interp_dim]).ndim != 1:
            raise NotImplementedError(
                "qinterp only supports one-dimensional coords so far"
            )

        if qdim not in self._obj.dims:
            raise ValueError(f"{qdim} is not a dimension of this DataArray")

        # interpolate
        arr = self._obj.values
        t_in = self._obj.coords[interp_dim]
        t_out = coords[interp_dim]
        axis = self._obj.dims.index(interp_dim)
        qaxis = self._obj.dims.index(qdim)
        arr_out = qinterp(arr, t_in, t_out, axis, qaxis)

        # update coords either by interpolating or selecting nearest for
        # non-numerical coords
        coords_out = dict(self._obj.coords)
        for c in coords_out:
            if c == interp_dim:
                coords_out[c] = t_out
            elif interp_dim in coords_out[c].dims:
                if np.issubdtype(coords_out[c].dtype, np.number):
                    coords_out[c] = coords_out[c].interp(coords)
                else:
                    coords_out[c] = coords_out[c].sel(coords, method="nearest")

        interpolated = xr.DataArray(
            arr_out,
            coords_out,
            self._obj.dims,
            self._obj.name,
            self._obj.attrs,
        )

        return interpolated

    def qinv(self, qdim="quaternion_axis"):
        """ Quaternion inverse.

        Parameters
        ----------
        qdim: str, default "quaternion_axis"
            Name of the dimension representing the quaternions.

        Returns
        -------
        inverse: xr.DataArray
            New array with inverted quaternions.

        Examples
        --------
        >>> import xarray as xr
        >>> import rigid_body_motion as rbm
        >>> ds_head = xr.load_dataset(rbm.example_data["head"])
        >>> ds_head.orientation.rbm.qinv() # doctest:+ELLIPSIS
        <xarray.DataArray 'orientation' (time: 66629, quaternion_axis: 4)>
        array(...)
        Coordinates:
          * time             (time) datetime64[ns] ...
          * quaternion_axis  (quaternion_axis) object 'w' 'x' 'y' 'z'
        Attributes:
            long_name:  Orientation
        """
        if qdim not in self._obj.dims:
            raise ValueError(f"{qdim} is not a dimension of this DataArray")

        qaxis = self._obj.dims.index(qdim)
        inverse = self._obj.copy()
        inverse.values = qinv(self._obj.values, qaxis)

        return inverse
