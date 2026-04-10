from __future__ import annotations

import numpy as np
import xarray as xr

from .base import BaseProcessor
from ..config import HindcastConfig
from ..io import normalize_coords_latlon


def _pick_temp_var(ds: xr.Dataset, preferred: str) -> str:
    """Pick the temperature variable from a GRIB-opened dataset."""
    if preferred in ds.data_vars:
        return preferred
    for cand in ("t2m", "2t", "t2m_mean", "t2m_min", "t2m_max", "2m_temperature", "air_temperature"):
        if cand in ds.data_vars:
            return cand
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise RuntimeError(f"Expected temperature variable in GRIB. Found: {list(ds.data_vars)}")


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Convert K->°C when appropriate (unit hint or value range)."""
    units = str(da.attrs.get("units", "")).lower()
    if "k" in units and "deg" not in units:
        out = da - 273.15
        out.attrs = da.attrs.copy()
        out.attrs["units"] = "degC"
        return out

    # Heuristic: Kelvin-like values
    try:
        v = float(da.isel({d: 0 for d in da.dims if d not in ("latitude", "longitude")}).mean().values)
        if v > 100.0:
            out = da - 273.15
            out.attrs = da.attrs.copy()
            out.attrs["units"] = "degC"
            return out
    except Exception:
        pass

    out = da.copy()
    if "units" not in out.attrs:
        out.attrs["units"] = "degC"
    return out


class TimeStepLikeProcessor(BaseProcessor):
    """Generic processor for CDS seasonal-monthly-single-levels temperature fields.

    Expected layout:
      - dims include time, step and variables include valid_time(time, step)
      - variable is a monthly statistic (mean/min/max) for 2m temperature

    Output:
      <out_var_name>(lead, latitude, longitude) in degC
      coords: lead=1..6, month(lead)=1..6 (calendar month per lead is carried by correction stage)
    """

    name = "time_step_like"

    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        return ("time" in ds.dims and "step" in ds.dims and "valid_time" in ds.variables)

    def process_grib(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        ds = normalize_coords_latlon(ds)

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

        vname = _pick_temp_var(ds, cfg.input_var)
        t = _to_celsius(ds[vname])
        vt = ds["valid_time"]  # (init_time, step_raw)

        # ECMWF-like behavior: target month = month(valid_time - 1 day)
        vt_np = vt.values.astype("datetime64[ns]")
        target_np = vt_np - np.timedelta64(1, "D")
        target_ym = target_np.astype("datetime64[M]")

        init_ym = ds["init_time"].values.astype("datetime64[M]")
        lead = (target_ym.astype("int64") - init_ym.astype("int64")[:, None]) + 1
        lead_da = xr.DataArray(lead, dims=vt.dims, coords=vt.coords, name="lead")
        mask = (lead_da >= 1) & (lead_da <= 6)

        t = t.where(mask)
        t = t.assign_coords(lead=lead_da)
        t.name = str(cfg.out_var_name)
        t.attrs["long_name"] = "2m air temperature (monthly statistic)"

        lead_fields = []
        for k in range(1, 7):
            sel = t.where(t["lead"] == k, drop=True)
            if "step_raw" in sel.dims:
                sel = sel.mean("step_raw", skipna=True)
            sel = sel.reset_coords(drop=True)
            sel = sel.expand_dims({"lead": [k]})
            lead_fields.append(sel)

        t_lead = xr.concat(lead_fields, dim="lead", coords="minimal", compat="override")
        # Hindcast climatology across members + init dates
        t_clim = t_lead.mean(dim=[d for d in ("member", "init_time") if d in t_lead.dims], skipna=True)

        out = xr.Dataset({str(cfg.out_var_name): t_clim})
        out = out.assign_coords(month=("lead", np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)))
        out[str(cfg.out_var_name)] = out[str(cfg.out_var_name)].transpose("lead", "latitude", "longitude")
        return out
