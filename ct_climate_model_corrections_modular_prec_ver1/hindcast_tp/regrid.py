from __future__ import annotations
from pathlib import Path
from typing import Optional
import xarray as xr
from .config import HindcastConfig
from .io import load_reference_grid

def regrid_dataset(out: xr.Dataset, ref_grid_path: Path, cfg: HindcastConfig, weights_file: Optional[Path]) -> xr.Dataset:
    import xesmf as xe
    ds_grid = load_reference_grid(ref_grid_path)
    src_grid = xr.Dataset({"lat": out["latitude"], "lon": out["longitude"]})

    kwargs = dict(method=cfg.regrid_method, periodic=cfg.regrid_periodic, reuse_weights=False)
    if weights_file is not None:
        weights_file.parent.mkdir(parents=True, exist_ok=True)
        if cfg.reuse_weights and weights_file.exists():
            kwargs["reuse_weights"] = True
        kwargs["filename"] = str(weights_file)

    regridder = xe.Regridder(src_grid, ds_grid, **kwargs)
    out_regrid = regridder(out)

    if "lat" in out_regrid.coords and "latitude" not in out_regrid.coords:
        out_regrid = out_regrid.rename({"lat": "latitude"})
    if "lon" in out_regrid.coords and "longitude" not in out_regrid.coords:
        out_regrid = out_regrid.rename({"lon": "longitude"})
    out_regrid["total_precipitation"] = out_regrid["total_precipitation"].transpose("lead", "latitude", "longitude")
    return out_regrid
