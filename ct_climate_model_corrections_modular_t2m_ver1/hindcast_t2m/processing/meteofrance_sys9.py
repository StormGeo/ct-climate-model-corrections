from __future__ import annotations

import xarray as xr

from .time_step_like import TimeStepLikeProcessor
from ..config import HindcastConfig


class MeteoFranceSys9Processor(TimeStepLikeProcessor):
    """Météo-France sys9 temperature (time/step/valid_time layout).

    Month alignment is handled in the pipeline (requesting previous month when needed).
    """

    name = "lfpw_sys9"

    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        if cfg.originating_centre.lower() in ("lfpw", "meteofrance", "meteo-france") and str(cfg.system) == "9":
            return super().can_handle(ds, cfg)

        centre = (ds.attrs.get("GRIB_centre") or "").lower()
        return centre == "lfpw" and super().can_handle(ds, cfg)
