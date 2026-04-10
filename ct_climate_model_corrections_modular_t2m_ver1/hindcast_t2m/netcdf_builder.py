import os
from datetime import datetime

import numpy as np
import pandas as pd
from netCDF4 import Dataset
from ncBuilder import ncBuilder


def create_nc(path_out, lons, lats, times, data, units, variable_name):
    os.makedirs(os.path.dirname(path_out), exist_ok=True)
    if os.path.isfile(path_out):
        os.remove(path_out)

    times = pd.to_datetime(times).to_pydatetime()
    now = datetime.utcnow().strftime("%Y%m%d%H")

    var = {
        f"{variable_name}": {
            "dims": ("time", "latitude", "longitude"),
            "dtype": np.float64,
            "long_name": f"{variable_name}",
            "standard_name": f"{variable_name}",
            "units": f"{units}",
            "variable_kw": {
                "least_significant_digit": 2,
                "timeseries": True,
            },
        }
    }

    nc_out = Dataset(path_out, "w")
    ncBuilder.create_nc(nc_out, lats, lons, time=times, vars=var, comp_lvl=9)

    for attr in ["Conventions", "Metadata_Conventions", "HISTORY"]:
        if attr in nc_out.ncattrs():
            delattr(nc_out, attr)

    nc_out.institution = "Climatempo - MetOps"
    nc_out.source = "Hydra"
    nc_out.description = f"netcdf file created by Hydra in {now}"
    nc_out.title = "climatempo - MetOps Netcdf file | from Hydra"
    nc_out.history = f"Created in {now}"

    ncBuilder.update_nc(nc_out, f"{variable_name}", data)

    nc_out.sync()
    nc_out.close()
    return True
