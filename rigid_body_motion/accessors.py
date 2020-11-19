import xarray as xr
from xarray.core.utils import either_dict_or_kwargs

from .utils import qinterp


@xr.register_dataarray_accessor("rbm")
class DataArrayAccessor:
    """ Accessor for DataArrays. """

    def __init__(self, obj):
        """ Constructor. """
        self._obj = obj

    def qinterp(self, coords=None, dim="quaternion_axis", **coords_kwargs):
        """ Quaternion interpolation.

        Parameters
        ----------
        coords: dict, optional
            Mapping from dimension names to the new coordinates.
            New coordinate can be a scalar, array-like or DataArray.

        dim: str, default "quaternion_axis"
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
          * quaternion_axis  (quaternion_axis) object 'w' 'x' 'y' 'z'
          * time             (time) datetime64[ns] ...
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

        if dim not in self._obj.dims:
            raise ValueError(f"{dim} is not a dimension of this DataArray")

        arr = self._obj.values
        t_in = self._obj.coords[interp_dim]
        t_out = coords[interp_dim]
        axis = self._obj.dims.index(interp_dim)
        qaxis = self._obj.dims.index(dim)

        arr_out = qinterp(arr, t_in, t_out, axis, qaxis)

        interpolated = self._obj.interp(coords)
        interpolated.values = arr_out

        return interpolated
