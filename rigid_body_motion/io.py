import numpy as np
from quaternion import as_float_array, from_rotation_vector


def load_optitrack(filepath, import_format="numpy"):
    """ Load rigid body poses from OptiTrack csv export.

    Parameters
    ----------
    filepath: str or path-like
        Path to csv file.

    import_format: {"numpy", "pandas", "xarray"}, default "numpy"
        Import format for rigid body poses. "numpy" returns a (position,
        orientation, timestamps) tuple for each body, "pandas" returns a
        DataFrame and "xarray" returns a Dataset.

    Returns
    -------
    data_dict: dict
        Dictionary with one entry for each rigid body. See ``import_format``
        for the format of each entry.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ModuleNotFoundError("Install pandas to use optitrack import")

    # parse header with metadata
    header = pd.read_csv(filepath, nrows=1, header=None).values
    meta = {k: v for k, v in zip(header[0][0::2], header[0][1::2])}

    # parse body with poses of rigid bodies and marker positions
    df = pd.read_csv(
        filepath, header=[2, 3, 5, 6], index_col=0, skip_blank_lines=False
    )

    # compute timestamps
    timestamps = pd.to_datetime(
        meta["Capture Start Time"], format="%Y-%m-%d %I.%M.%S.%f %p"
    ) + pd.to_timedelta(df.values[:, 0], unit="s")

    # only import rigid bodies, not markers
    df = df["Rigid Body"]

    # set index to timestamps and multi-index levels
    df = df.set_index(timestamps).rename_axis("time")
    df.columns = df.columns.set_names(["body ID", "motion type", "axis"])

    # split dataframe into dict with one entry for each rigid body
    data_dict = {
        idx: gp.xs(idx, level=0, axis=1)
        for idx, gp in df.groupby(level=0, axis=1)
    }

    # convert cm to m (if applicable)
    if meta.get("Length Units", None) == "Centimeters":
        for name in data_dict:
            data_dict[name]["Position"] /= 100

    # convert orientation to quaternions
    for name in data_dict:
        r = np.deg2rad(data_dict[name]["Rotation"].values)
        qx = from_rotation_vector(
            r[:, 0][:, np.newaxis] * np.array([1, 0, 0])[np.newaxis, :]
        )
        qy = from_rotation_vector(
            r[:, 1][:, np.newaxis] * np.array([0, 1, 0])[np.newaxis, :]
        )
        qz = from_rotation_vector(
            r[:, 2][:, np.newaxis] * np.array([0, 0, 1])[np.newaxis, :]
        )
        if meta.get("Rotation Type", "XYZ") == "XYZ":
            q = as_float_array(qx * qy * qz)
        else:
            raise ValueError("Unsupported rotation type")

        position = data_dict[name]["Position"]
        position.columns = position.columns.str.lower()
        orientation = pd.DataFrame(
            q, index=position.index, columns=["w", "x", "y", "z"]
        )
        data_dict[name] = pd.concat(
            {"position": position, "orientation": orientation}, axis=1,
        )

    # return in requested format
    if import_format == "pandas":
        return data_dict

    elif import_format == "numpy":
        return {
            k: (v["position"].values, v["orientation"].values, v.index.values)
            for k, v in data_dict.items()
        }

    elif import_format == "xarray":
        import xarray as xr

        for name in data_dict:
            position = xr.DataArray(
                data_dict[name]["position"],
                name="position",
                dims=("time", "cartesian_axis"),
            )
            orientation = xr.DataArray(
                data_dict[name].orientation,
                name="orientation",
                dims=("time", "quaternion_axis"),
            )
            data_dict[name] = xr.merge((position, orientation))

        return data_dict

    else:
        raise ValueError(f"Unsupported import format: {import_format}")
