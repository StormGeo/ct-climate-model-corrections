# hindcast_tp/processing/ukmo_sys603.py
from __future__ import annotations

import numpy as np
import xarray as xr

from .base import BaseProcessor
from .time_step_like import seconds_in_month_from_ym
from ..config import HindcastConfig
from ..io import normalize_coords_latlon


def _to_datetime64_ns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]")
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    sec = arr.astype("int64")
    return epoch + sec.astype("timedelta64[s]").astype("timedelta64[ns]")


def _ym_month_int(dt64_ns: np.ndarray) -> np.ndarray:
    # datetime64[ns] -> datetime64[M] -> int
    return dt64_ns.astype("datetime64[M]").astype("int64")


class UKMOSystem603Processor(BaseProcessor):
    """
    UK Met Office (centre egrr) – system 603
    Monthly mean precipitation rate (tprate)

    Supports:
      A) indexing_time/forecastMonth layout
      B) time/step (+ valid_time) layout (sparse) with streaming + sparse-safe selection
    """

    name = "egrr_sys603"

    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        if str(cfg.originating_centre).lower() in (
            "egrr",
            "ukmo",
            "uk-met-office",
            "ukmetoffice",
            "metoffice",
        ) and str(cfg.system) == "603":
            return True
        centre = (ds.attrs.get("GRIB_centre") or "").lower()
        return centre == "egrr"

    def process_grib(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        ds = normalize_coords_latlon(ds)

        if cfg.input_var not in ds:
            raise RuntimeError(f"Variable '{cfg.input_var}' not found. Found: {list(ds.data_vars)}")

        has_fcmonth = "forecastMonth" in ds.dims
        has_indexing = "indexing_time" in ds.dims
        has_time_step = "time" in ds.dims and "step" in ds.dims

        if has_fcmonth and has_indexing:
            return self._process_layout_forecastMonth(ds, cfg)

        if has_time_step:
            return self._process_layout_time_step(ds, cfg)

        raise RuntimeError(f"UKMO sys603: unsupported layout. dims={dict(ds.dims)} coords={list(ds.coords)}")

    # ------------------------------------------------------------------
    # Layout A: indexing_time / forecastMonth
    # ------------------------------------------------------------------
    def _process_layout_forecastMonth(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        rename = {}
        if "number" in ds.dims:
            rename["number"] = "member"
        if "indexing_time" in ds.dims:
            rename["indexing_time"] = "init_time"
        if "forecastMonth" in ds.dims:
            rename["forecastMonth"] = "lead_raw"
        if rename:
            ds = ds.rename(rename)

        for dim in ("init_time", "lead_raw", "latitude", "longitude"):
            if dim not in ds.dims:
                raise RuntimeError(f"UKMO sys603 expected dim '{dim}'. Got dims={ds.dims}")

        rate = ds[cfg.input_var]  # m s-1

        lead_raw = ds["lead_raw"].values.astype(int)
        lead_oper = lead_raw + 1 if lead_raw.min() == 0 else lead_raw
        lead = np.arange(1, 7, dtype=int)

        init_ym = _ym_month_int(_to_datetime64_ns(ds["init_time"].values))  # (init_time,)
        target_ym_int = init_ym[:, None] + lead[None, :]  # (init_time, lead)
        target_ym = target_ym_int.astype("datetime64[M]")

        sec = np.array([seconds_in_month_from_ym(m) for m in target_ym.ravel()], dtype=np.float64).reshape(
            target_ym.shape
        )

        pieces = []
        for l in lead:
            idx = np.where(lead_oper == l)[0]
            if idx.size == 0:
                pieces.append(rate.isel(lead_raw=0) * np.nan)
            else:
                pieces.append(rate.isel(lead_raw=int(idx[0])))

        rate_sel = xr.concat(pieces, dim=xr.DataArray(lead, dims="lead", name="lead"))
        sec_da = xr.DataArray(sec, dims=("init_time", "lead"), coords={"init_time": ds["init_time"], "lead": lead})

        tp = rate_sel * sec_da * 1000.0  # mm/month

        # climatology over member + init_time
        if "member" in tp.dims:
            tp = tp.mean(dim=("member", "init_time"), skipna=True)
        else:
            tp = tp.mean(dim=("init_time",), skipna=True)

        cal_month = np.array([int(str(m).split("-")[1]) for m in target_ym[0]], dtype=int)

        out = xr.Dataset({"total_precipitation": tp})
        out = out.assign_coords(lead=("lead", lead))
        out = out.assign_coords(month=("lead", cal_month))
        out = out.assign_coords(calendar_month=("lead", cal_month))
        out["total_precipitation"] = out["total_precipitation"].transpose("lead", "latitude", "longitude")
        out["total_precipitation"].attrs.update(
            {
                "units": "mm",
                "long_name": "Total precipitation (monthly accumulated)",
                "source": "UKMO (egrr) system 603; from tprate (m s-1) using seconds in target month.",
            }
        )
        return out

    # ------------------------------------------------------------------
    # Layout B: time / step (SPARSE) – streaming + sparse-safe selection
    # ------------------------------------------------------------------
    def _process_layout_time_step(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        # Normalize ensemble dim name
        if "number" in ds.dims and "member" not in ds.dims:
            ds = ds.rename({"number": "member"})

        rate = ds[cfg.input_var]
        if "valid_time" not in ds:
            raise RuntimeError("UKMO sys603 time/step layout requires 'valid_time' coordinate.")
        vt = ds["valid_time"]

        # init months from time
        init_ym = _ym_month_int(_to_datetime64_ns(ds["time"].values))  # (time,)
        # target months from valid_time
        target_ym = _ym_month_int(_to_datetime64_ns(vt.values))  # (time,step) as int months

        # physical lead in months
        lead_phys = (target_ym - init_ym[:, None]).astype(np.int64)  # (time,step)

        # Rebase so min positive lead becomes 1 (UKMO monthly_mean behaves like France-like)
        mask_pos = (lead_phys >= 1) & (lead_phys <= 24)
        if not np.any(mask_pos):
            raise RuntimeError("UKMO sys603: could not compute valid leads from time/valid_time")

        min_lead = int(np.nanmin(lead_phys[mask_pos]))
        lead_norm = lead_phys - (min_lead - 1)  # now starts at 1
        keep = (lead_norm >= 1) & (lead_norm <= 6)

        lat = ds["latitude"].values
        lon = ds["longitude"].values

        out_sum = np.zeros((6, lat.size, lon.size), dtype=np.float64)
        out_cnt = np.zeros(6, dtype=np.int64)
        cal_month = np.full(6, -1, dtype=np.int64)

        # 1) find time indices that have ANY finite values in early steps (sparse cube!)
        valid_time_idx: list[int] = []
        for ti in range(ds.sizes["time"]):
            ok = False
            for si in range(min(10, ds.sizes["step"])):
                try:
                    sample = rate.isel(time=ti, step=si, latitude=slice(0, 2), longitude=slice(0, 2))
                    if "member" in sample.dims:
                        sample = sample.isel(member=0)
                    v = sample.values
                    if np.isfinite(v).any():
                        ok = True
                        break
                except Exception:
                    continue
            if ok:
                valid_time_idx.append(ti)

        if not valid_time_idx:
            raise RuntimeError("UKMO sys603: no valid time indices found (GRIB cube is fully sparse/NaN).")

        # 2) For each valid init time, for each lead, pick FIRST step that is finite
        for ti in valid_time_idx:
            for l in range(1, 7):
                step_candidates = np.where((lead_norm[ti, :] == l) & keep[ti, :])[0]
                if step_candidates.size == 0:
                    continue

                chosen = None
                for si in step_candidates:
                    try:
                        sample = rate.isel(time=ti, step=int(si), latitude=slice(0, 2), longitude=slice(0, 2))
                        if "member" in sample.dims:
                            sample = sample.isel(member=0)
                        v = sample.values
                        if np.isfinite(v).any():
                            chosen = int(si)
                            break
                    except Exception:
                        continue

                if chosen is None:
                    continue

                # int-month -> datetime64[M]
                ym = np.array(target_ym[ti, chosen], dtype="int64").astype("datetime64[M]")
                sec = seconds_in_month_from_ym(ym)

                if "member" in rate.dims:
                    da = rate.isel(time=ti, step=chosen).mean("member", skipna=True)
                else:
                    da = rate.isel(time=ti, step=chosen)

                vals = da.values
                if not np.isfinite(vals).any():
                    continue

                out_sum[l - 1] += vals.astype(np.float64) * float(sec) * 1000.0
                out_cnt[l - 1] += 1

                if cal_month[l - 1] < 0:
                    # ym is 'YYYY-MM'
                    cal_month[l - 1] = int(str(ym).split("-")[1])

        out = np.full_like(out_sum, np.nan, dtype=np.float64)
        for i in range(6):
            if out_cnt[i] > 0:
                out[i] = out_sum[i] / float(out_cnt[i])

        da_out = xr.DataArray(
            out,
            dims=("lead", "latitude", "longitude"),
            coords={"lead": np.arange(1, 7, dtype=np.int64), "latitude": lat, "longitude": lon},
            name="total_precipitation",
            attrs={
                "units": "mm",
                "long_name": "Total precipitation (monthly accumulated)",
                "source": "UKMO (egrr) system 603; from tprate (m s-1) using seconds in target month (sparse-safe).",
            },
        )

        ds_out = xr.Dataset({"total_precipitation": da_out})
        ds_out = ds_out.assign_coords(month=("lead", cal_month))
        ds_out = ds_out.assign_coords(calendar_month=("lead", cal_month))
        ds_out["total_precipitation"] = ds_out["total_precipitation"].transpose("lead", "latitude", "longitude")
        return ds_out
