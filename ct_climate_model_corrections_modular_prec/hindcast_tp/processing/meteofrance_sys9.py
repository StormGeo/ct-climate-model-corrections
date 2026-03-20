# hindcast_tp/processing/meteofrance_sys9.py
from __future__ import annotations

import numpy as np
import xarray as xr

from .base import BaseProcessor
from .time_step_like import seconds_in_month_from_ym
from ..config import HindcastConfig
from ..io import normalize_coords_latlon


class MeteoFranceSys9Processor(BaseProcessor):
    name = "lfpw_sys9"

    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        # Prefer config-based selection
        if cfg.originating_centre.lower() in ("lfpw", "meteofrance", "meteo-france") and str(cfg.system) == "9":
            return ("time" in ds.dims and "step" in ds.dims and "valid_time" in ds.variables)

        # Fallback using GRIB centre attr
        centre = (ds.attrs.get("GRIB_centre") or "").lower()
        return centre == "lfpw" and ("time" in ds.dims and "step" in ds.dims and "valid_time" in ds.variables)

    def process_grib(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        ds = normalize_coords_latlon(ds)

        if cfg.input_var not in ds.data_vars:
            raise RuntimeError(f"Expected variable '{cfg.input_var}' in GRIB. Found: {list(ds.data_vars)}")
        if "valid_time" not in ds.variables:
            raise RuntimeError("Expected 'valid_time(time, step)' in the dataset.")

        # Normalize dim names to standard internal names
        rename = {}
        if "number" in ds.dims:
            rename["number"] = "member"
        if "time" in ds.dims:
            rename["time"] = "init_time"
        if "step" in ds.dims:
            rename["step"] = "step_raw"
        if rename:
            ds = ds.rename(rename)

        rate = ds[cfg.input_var]   # m/s
        vt = ds["valid_time"]      # (init_time, step_raw)

        # -------- Météo-France sys9 rules --------
        # 1) Target month is month(valid_time) directly (NO "-1 day")
        vt_np = vt.values.astype("datetime64[ns]")
        target_ym = vt_np.astype("datetime64[M]")

        # 2) Lead is the difference in months (NO "+1")
        init_ym = ds["init_time"].values.astype("datetime64[M]")
        lead = (target_ym.astype("int64") - init_ym.astype("int64")[:, None])

        lead_da = xr.DataArray(lead, dims=vt.dims, coords=vt.coords, name="lead")
        mask = (lead_da >= 1) & (lead_da <= 6)

        # Seconds in the target month (for converting rate -> monthly total)
        flat_ym = target_ym.ravel()
        sec = np.array([seconds_in_month_from_ym(ym) for ym in flat_ym], dtype=np.float64).reshape(target_ym.shape)
        sec_da = xr.DataArray(sec, dims=vt.dims, coords=vt.coords, name="seconds_in_target_month")

        # Convert to mm/month and keep only leads 1..6
        tp_mm = (rate * sec_da * 1000.0).where(mask)
        tp_mm = tp_mm.assign_coords(lead=lead_da)
        tp_mm.name = "total_precipitation"
        tp_mm.attrs["units"] = "mm"
        tp_mm.attrs["long_name"] = "Total precipitation (monthly accumulated)"
        tp_mm.attrs["note"] = (
            "Météo-France sys9 (lfpw): target_month = month(valid_time) and "
            "lead = month(target) - month(init) (no +1). Duplicate valid_time "
            "days inside the same month are averaged per lead."
        )

        # Robust duplicate handling: stack (init_time, step_raw) -> groupby lead
        tp_s = tp_mm.stack(sample=("init_time", "step_raw"))
        lead_s = lead_da.stack(sample=("init_time", "step_raw"))

        valid = ~np.isnan(lead_s.values)
        tp_s = tp_s.isel(sample=valid)
        lead_s = lead_s.isel(sample=valid)

        tp_s = tp_s.assign_coords(lead=("sample", lead_s.values.astype(np.int64)))
        tp_by_lead = tp_s.groupby("lead").mean("sample", skipna=True)

        # Ensure we always output leads 1..6 (missing leads become NaN)
        tp_by_lead = tp_by_lead.reindex(lead=[1, 2, 3, 4, 5, 6])

        # Hindcast climatology over ensemble members (and init_time already collapsed via sample)
        tp_clim = tp_by_lead.mean(dim="member", skipna=True)

        out = xr.Dataset({"total_precipitation": tp_clim})
        out = out.assign_coords(month=("lead", np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)))
        out["total_precipitation"] = out["total_precipitation"].transpose("lead", "latitude", "longitude")
        return out
