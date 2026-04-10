from __future__ import annotations
import numpy as np
import xarray as xr
from .base import BaseProcessor
from ..config import HindcastConfig
from ..io import normalize_coords_latlon

def seconds_in_month_from_ym(ym: np.datetime64) -> float:
    start = ym.astype("datetime64[ns]")
    end = (ym + np.timedelta64(1, "M")).astype("datetime64[ns]")
    return float((end - start) / np.timedelta64(1, "s"))

class TimeStepLikeProcessor(BaseProcessor):
    name = "time_step_like"

    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        return ("time" in ds.dims and "step" in ds.dims and "valid_time" in ds.variables)

    def process_grib(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        ds = normalize_coords_latlon(ds)

        if cfg.input_var not in ds.data_vars:
            raise RuntimeError(f"Expected variable '{cfg.input_var}' in GRIB. Found: {list(ds.data_vars)}")
        if "valid_time" not in ds.variables:
            raise RuntimeError("Expected 'valid_time(time, step)' in the dataset.")

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

        # ECMWF default behavior: month(valid_time - 1 day)
        vt_np = vt.values.astype("datetime64[ns]")
        target_np = vt_np - np.timedelta64(1, "D")
        target_ym = target_np.astype("datetime64[M]")

        init_ym = ds["init_time"].values.astype("datetime64[M]")
        lead = (target_ym.astype("int64") - init_ym.astype("int64")[:, None]) + 1
        lead_da = xr.DataArray(lead, dims=vt.dims, coords=vt.coords, name="lead")
        mask = (lead_da >= 1) & (lead_da <= 6)

        flat_ym = target_ym.ravel()
        sec = np.array([seconds_in_month_from_ym(ym) for ym in flat_ym], dtype=np.float64).reshape(target_ym.shape)
        sec_da = xr.DataArray(sec, dims=vt.dims, coords=vt.coords, name="seconds_in_target_month")

        tp_mm = (rate * sec_da * 1000.0).where(mask)
        tp_mm = tp_mm.assign_coords(lead=lead_da)
        tp_mm.name = "total_precipitation"
        tp_mm.attrs["units"] = "mm"
        tp_mm.attrs["long_name"] = "Total precipitation (monthly accumulated)"
        tp_mm.attrs["note"] = (
            "From CDS monthly_mean tprate (m/s): total = tprate * seconds_in_target_month * 1000; "
            "target_month = month(valid_time - 1 day)."
        )

        lead_fields = []
        for k in range(1, 7):
            sel = tp_mm.where(tp_mm["lead"] == k, drop=True)
            if "step_raw" in sel.dims:
                sel = sel.mean("step_raw", skipna=True)
            sel = sel.reset_coords(drop=True)
            sel = sel.expand_dims({"lead": [k]})
            lead_fields.append(sel)

        tp_lead = xr.concat(lead_fields, dim="lead", coords="minimal", compat="override")
        tp_clim = tp_lead.mean(dim=["member", "init_time"], skipna=True)

        out = xr.Dataset({"total_precipitation": tp_clim})
        out = out.assign_coords(month=("lead", np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)))
        out["total_precipitation"] = out["total_precipitation"].transpose("lead", "latitude", "longitude")
        return out
