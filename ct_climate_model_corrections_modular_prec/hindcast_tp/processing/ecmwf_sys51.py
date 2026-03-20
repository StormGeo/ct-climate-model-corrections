from __future__ import annotations
import xarray as xr
from .time_step_like import TimeStepLikeProcessor
from ..config import HindcastConfig

class ECMWFSys51Processor(TimeStepLikeProcessor):
    name = "ecmwf_sys51"

    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        if cfg.originating_centre.lower() == "ecmwf" and str(cfg.system) == "51":
            return super().can_handle(ds, cfg)
        centre = (ds.attrs.get("GRIB_centre") or "").lower()
        system = ds.attrs.get("GRIB_system")
        return centre == "ecmwf" and (system is None or str(system) == "51") and super().can_handle(ds, cfg)
