from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import CorrectionConfig
from ..utils import extract_year_from_path

import warnings

# Silencia warnings do xESMF
warnings.filterwarnings("ignore", category=UserWarning, module=r"xesmf\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"ESMF\..*")
warnings.filterwarnings("ignore", category=UserWarning, module=r"esmpy\..*")

try:
    import xesmf as xe
except Exception:  # pragma: no cover
    xe = None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _std_latlon(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ds = ds.rename({"lon": "longitude"})
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise RuntimeError(f"Missing latitude/longitude coords. Coords={list(ds.coords)}")
    return ds


def _sort_latlon(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds


def list_doys_in_month_2001(month: int) -> List[int]:
    start = np.datetime64(f"2001-{month:02d}-01")
    end = np.datetime64("2002-01-01") if month == 12 else np.datetime64(f"2001-{month+1:02d}-01")
    ndays = int((end - start).astype("timedelta64[D]").astype(np.int64))
    doy0 = int(((start.astype("datetime64[D]") - np.datetime64("2001-01-01")) / np.timedelta64(1, "D")) + 1)
    return [doy0 + i for i in range(ndays)]


def doy_of_2001_from_yyyy_mm_dd(tday: np.datetime64, max_doy: int) -> int:
    """Map a real calendar date to the DOY index used by the observed daily climatology.

    - Uses a non-leap dummy year (2001) for month/day mapping.
    - Special case: Feb 29. If climatology has only 365 days (max_doy < 366),
      map Feb 29 to Feb 28 (DOY 59) to avoid shifting the rest of the year.
      If climatology provides DOY 366, map Feb 29 to its own DOY (60).
    """
    s = str(tday.astype("datetime64[D]"))
    mm = int(s[5:7])
    dd = int(s[8:10])

    if mm == 2 and dd == 29:
        if int(max_doy) < 366:
            return 59  # use Feb 28 climatology
        # leap climatology: compute DOY in a leap dummy year (2000)
        d2000 = np.datetime64("2000-02-29").astype("datetime64[D]")
        return int(((d2000 - np.datetime64("2000-01-01")) / np.timedelta64(1, "D")) + 1)

    # regular month/day: safe in 2001
    d2001 = np.datetime64(f"2001-{mm:02d}-{dd:02d}").astype("datetime64[D]")
    return int(((d2001 - np.datetime64("2001-01-01")) / np.timedelta64(1, "D")) + 1)



class ForecastCorrectionPipeline:
    """Daily temperature correction pipeline (bias-add + daily-clim clipping).

    Logic matches the precipitation module structure:
      1) Load observed DAILY climatology and build a fast DOY lookup.
      2) For each DOY folder:
           - load processed hindcast (monthly, by lead)
           - load raw forecast, aggregate to daily mean if sub-daily
           - regrid to reference grid (climatology grid)
           - apply lead-wise additive bias: corr = fore + (obs_month_mean - hind_mean)
           - clip around daily obs climatology: corr ∈ [clim_daily-Δ, clim_daily+Δ]
           - write corrected forecast to <out_root>/<var>/<year>/<doy>/...
    """

    def __init__(self, cfg: CorrectionConfig):
        self.cfg = cfg

        self.forecast_root = cfg.forecast_root.expanduser().resolve()
        self.hindcast_root = cfg.hindcast_root.expanduser().resolve()
        self.clim_file = cfg.clim_file.expanduser().resolve()
        self.out_root = cfg.out_root.expanduser().resolve()

        if not self.clim_file.exists():
            raise FileNotFoundError(f"Climatology file not found: {self.clim_file}")

        # year for output foldering (follow precip module behavior)
        try:
            self.out_year = extract_year_from_path(self.forecast_root)
        except Exception:
            self.out_year = extract_year_from_path(self.hindcast_root)

        # load observed climatology and ref grid
        self._clim_lookup, self._max_doy, self._ny, self._nx, self._lat_ref, self._lon_ref = (
            self._load_obs_climatology_daily_fast()
        )
        self._obs_month_mean = self._build_obs_month_means()

        self._regridders: Dict[Tuple, xe.Regridder] = {}  # cache per source grid

    # ------------------------------------------------------------------
    # Observed daily climatology
    # ------------------------------------------------------------------
    def _load_obs_climatology_daily_fast(self) -> Tuple[np.ndarray, int, int, int, xr.DataArray, xr.DataArray]:
        ds = xr.open_dataset(self.clim_file)
        ds = _std_latlon(ds)
        ds = _sort_latlon(ds)

        if self.cfg.var_name not in ds.data_vars:
            if len(ds.data_vars) == 1:
                vname = list(ds.data_vars)[0]
            else:
                raise KeyError(
                    f"Variable '{self.cfg.var_name}' not found in clim_file. Vars: {list(ds.data_vars)}"
                )
        else:
            vname = self.cfg.var_name

        da = ds[vname]
        if "time" not in da.dims:
            raise ValueError("Observed DAILY climatology must have 'time' dimension.")

        da = da.transpose("time", "latitude", "longitude").astype("float32").load()

        n = int(da.sizes["time"])
        if n < 365:
            raise ValueError(f"Climatology expected >=365 timesteps, got time={n}.")

        doys = da["time"].dt.dayofyear.values.astype(np.int32)
        max_doy = int(doys.max())
        ny = int(da.sizes["latitude"])
        nx = int(da.sizes["longitude"])

        clim_lookup = np.zeros((max_doy + 1, ny, nx), dtype=np.float32)
        filled = np.zeros((max_doy + 1,), dtype=bool)

        vals = da.values
        for i in range(vals.shape[0]):
            d = int(doys[i])
            if 1 <= d <= max_doy and (not filled[d]):
                clim_lookup[d, :, :] = vals[i, :, :]
                filled[d] = True

        nfilled = int(filled.sum()) - int(filled[0])
        if nfilled < 365:
            raise ValueError(f"Daily climatology insufficient: filled {nfilled} DOYs (expected >=365).")

        return clim_lookup, max_doy, ny, nx, ds["latitude"], ds["longitude"]

    def _build_obs_month_means(self) -> Dict[int, np.ndarray]:
        out: Dict[int, np.ndarray] = {}
        for m in range(1, 13):
            doys_m = np.array(list_doys_in_month_2001(m), dtype=np.int32)
            doys_m = doys_m[doys_m <= self._max_doy]
            if doys_m.size < 1:
                out[m] = np.zeros((self._ny, self._nx), dtype=np.float32)
            else:
                out[m] = np.mean(self._clim_lookup[doys_m, :, :], axis=0).astype(np.float32)
        return out

    # ------------------------------------------------------------------
    # Regrid
    # ------------------------------------------------------------------
    @staticmethod
    def _grid_key(ds: xr.Dataset) -> Tuple:
        ds = _std_latlon(ds)
        ds = _sort_latlon(ds)
        lat = ds["latitude"].values
        lon = ds["longitude"].values
        # Key by shape + endpoints (enough to avoid mixing grids)
        return (int(lat.size), int(lon.size), float(lat[0]), float(lat[-1]), float(lon[0]), float(lon[-1]))

    def _build_regridder(self, ds_in: xr.Dataset, weights_file: Optional[Path]) -> xe.Regridder:
        if xe is None:
            raise RuntimeError("xesmf is required for regridding.")

        ds_in = _std_latlon(ds_in)
        ds_in = _sort_latlon(ds_in)
        if weights_file is None:
            grid_in = xr.Dataset(
                {"lat": (("lat",), ds_in["latitude"].values), "lon": (("lon",), ds_in["longitude"].values)}
            )
            grid_out = xr.Dataset({"lat": (("lat",), self._lat_ref.values), "lon": (("lon",), self._lon_ref.values)})
            return xe.Regridder(grid_in, grid_out, "bilinear")

        # weights_file can be a directory; build a unique filename per source grid shape
        if weights_file.suffix != ".nc":
            ensure_dir(weights_file)
            ds_in2 = _std_latlon(ds_in)
            ds_in2 = _sort_latlon(ds_in2)
            ny = int(ds_in2["latitude"].size)
            nx = int(ds_in2["longitude"].size)
            weights_file = weights_file / f"regrid_weights_{ny}x{nx}.nc"

        grid_in = xr.Dataset(
            {"lat": (("lat",), ds_in["latitude"].values), "lon": (("lon",), ds_in["longitude"].values)}
        )
        grid_out = xr.Dataset({"lat": (("lat",), self._lat_ref.values), "lon": (("lon",), self._lon_ref.values)})

        ensure_dir(weights_file.parent)
        reuse = weights_file.exists()
        regridder = xe.Regridder(
            grid_in,
            grid_out,
            "bilinear",
            filename=str(weights_file),
            reuse_weights=reuse,
        )
        return regridder

    def _regrid_to_ref(self, ds: xr.Dataset, weights_file: Optional[Path]) -> xr.Dataset:
        ds = _std_latlon(ds)
        ds = _sort_latlon(ds)
        gkey = self._grid_key(ds)
        regridder = self._regridders.get(gkey)
        if regridder is None:
            regridder = self._build_regridder(ds, weights_file=weights_file)
            self._regridders[gkey] = regridder

        out = regridder(ds)
        if "lat" in out.coords and "latitude" not in out.coords:
            out = out.rename({"lat": "latitude"})
        if "lon" in out.coords and "longitude" not in out.coords:
            out = out.rename({"lon": "longitude"})
        return _sort_latlon(out)

    # ------------------------------------------------------------------
    # Hindcast / forecast IO
    # ------------------------------------------------------------------
    def _load_hindcast(self, doy_subdir: str, weights_file: Optional[Path]) -> xr.Dataset:
        files = sorted(self.hindcast_root.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No hindcast .nc files found in: {self.hindcast_root}")

        ds = xr.open_dataset(files[0])
        ds = _std_latlon(ds)
        ds = _sort_latlon(ds)

        if self.cfg.var_name not in ds:
            raise KeyError(f"Variable '{self.cfg.var_name}' not found in hindcast file: {files[0]}")
        if "lead" not in ds.dims:
            raise ValueError(f"Hindcast must have 'lead' dimension. Dims: {ds.dims}")
        if "month" not in ds.coords:
            raise ValueError("Hindcast must have 'month' coordinate (month per lead).")

        ds_in = ds[[self.cfg.var_name, "month"]].copy()
        ds_in[self.cfg.var_name].data = np.ascontiguousarray(ds_in[self.cfg.var_name].data)
        return self._regrid_to_ref(ds_in, weights_file=weights_file)

    @staticmethod
    def _forecast_to_daily_mean(da: xr.DataArray) -> xr.DataArray:
        if "time" not in da.dims:
            raise ValueError("Forecast variable must have 'time' dimension.")
        out = da.resample(time="1D").mean(keep_attrs=True)
        out = out.astype("float32")
        out.attrs = da.attrs.copy()
        out.attrs["units"] = "degC"
        return out

    def _build_time_units(self, time_values: np.ndarray) -> str:
        t0 = pd.Timestamp(time_values[0]).strftime("%Y-%m-%d %H:%M:%S")
        return f"hours since {t0}"

    def _format_corrected_output(self, ds: xr.Dataset) -> tuple[xr.Dataset, dict]:
        ds = ds.copy()

        var_name = self.cfg.var_name
        fill_value = np.float32(-9.99e8)
        time_units = self._build_time_units(ds["time"].values)
        now = datetime.utcnow().strftime("%Y%m%d%H")

        ds[var_name] = ds[var_name].astype("float32")

        ds["time"].attrs.pop("units", None)
        ds["time"].attrs.pop("calendar", None)

        if "units" in ds["time"].encoding:
            del ds["time"].encoding["units"]
        if "calendar" in ds["time"].encoding:
            del ds["time"].encoding["calendar"]

        ds["time"].attrs["standard_name"] = "time"
        ds["time"].attrs["axis"] = "T"

        ds["latitude"].attrs = {
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north",
            "grads_dim": "Y",
        }

        ds["longitude"].attrs = {
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east",
            "grads_dim": "x",
        }

        ds[var_name].attrs = {
            "units": "C",
            "missing_value": fill_value,
        }

        ds.attrs = {
            "institution": "Climatempo - MetOps",
            "source": "Hydra",
            "description": f"netcdf file created by Hydra in {now}",
            "title": "climatempo - MetOps Netcdf file | from Hydra",
            "history": f"Created in {now}",
        }

        encoding = {
            var_name: {
                "_FillValue": fill_value,
                "dtype": "float32",
                "least_significant_digit": 2,
            },
            "time": {
                "dtype": "float32",
                "units": time_units,
                "calendar": "standard",
            },
        }

        return ds, encoding

    def _preprocess_forecast_daily(self, forecast_path: Path, hind: xr.Dataset, weights_file: Optional[Path]) -> Tuple[xr.DataArray, np.datetime64, str]:
        ds = xr.open_dataset(forecast_path)
        ds = _std_latlon(ds)
        ds = _sort_latlon(ds)

        if self.cfg.var_name not in ds:
            raise KeyError(f"Variable '{self.cfg.var_name}' not found in forecast file: {forecast_path}")

        if "time" not in ds.coords or ds["time"].size < 1:
            raise ValueError("Forecast must have non-empty 'time' coordinate.")

        init_stamp = self._init_stamp_from_filename(forecast_path)
        da = ds[self.cfg.var_name]

        t_init = ds["time"].values[0].astype("datetime64[D]")

        # Cut sub-daily timesteps by lead range BEFORE daily aggregation
        t_sub_days = ds["time"].values.astype("datetime64[D]")
        lead0_sub = (t_sub_days.astype("datetime64[M]").astype(int) - t_init.astype("datetime64[M]").astype(int))
        lead_arr_sub = (lead0_sub + 1).astype(np.int32)

        n_hind_leads = int(hind[self.cfg.var_name].sizes["lead"])
        max_lead_out = n_hind_leads + int(self.cfg.extend_months)
        valid_idx_sub = np.where((lead_arr_sub >= 1) & (lead_arr_sub <= max_lead_out))[0]
        if valid_idx_sub.size < 1:
            raise RuntimeError("No timesteps are within hindcast lead range (including extend_months).")

        da_slice = da.isel(time=valid_idx_sub)

        # Aggregate to daily mean ONLY if forecast is sub-daily.
        t = da_slice["time"].values.astype("datetime64[h]")
        if t.size < 2:
            raise ValueError("Forecast has < 2 timesteps; cannot infer frequency.")

        dt = np.unique(np.diff(t).astype(int))  # hours
        if dt.size == 1 and dt[0] == 24:
            da_daily = da_slice.astype("float32")
            da_daily.attrs = da_slice.attrs.copy()
            da_daily.attrs["units"] = "degC"
        else:
            da_daily = self._forecast_to_daily_mean(da_slice)

        # Regrid to ref
        ds_daily = da_daily.to_dataset(name=self.cfg.var_name).copy()
        ds_daily[self.cfg.var_name].data = np.ascontiguousarray(ds_daily[self.cfg.var_name].data)
        ds_daily_rg = self._regrid_to_ref(ds_daily, weights_file=weights_file)
        out = ds_daily_rg[self.cfg.var_name].astype("float32")

        return out, t_init, init_stamp

    # ------------------------------------------------------------------
    # Correction
    # ------------------------------------------------------------------
    def _correct_daily(self, fore_d_ref: xr.DataArray, hind: xr.Dataset, t_init: np.datetime64) -> xr.Dataset:
        hind_m = hind[self.cfg.var_name].astype("float32")
        hm = hind["month"]
        n_hind_leads = int(hind_m.sizes["lead"])
        max_lead_out = n_hind_leads + int(self.cfg.extend_months)

        # limit output days
        t_daily_all = fore_d_ref["time"].values.astype("datetime64[D]")
        if t_daily_all.size < 1:
            raise RuntimeError("Daily forecast became empty after preprocessing.")

        max_days = int(self.cfg.max_days_output)
        t0_day = t_daily_all[0]
        t_end = t0_day + np.timedelta64(max_days - 1, "D")
        idx_lim = np.where((t_daily_all >= t0_day) & (t_daily_all <= t_end))[0]
        if idx_lim.size < 1:
            raise RuntimeError("max_days_output removed all days.")
        fore_d_ref = fore_d_ref.isel(time=idx_lim)

        t_days = fore_d_ref["time"].values.astype("datetime64[D]")
        lead0 = (t_days.astype("datetime64[M]").astype(int) - t_init.astype("datetime64[M]").astype(int))
        lead_arr = (lead0 + 1).astype(np.int32)
        keep = np.where((lead_arr >= 1) & (lead_arr <= max_lead_out))[0]
        if keep.size < 1:
            raise RuntimeError("No daily steps left within lead range (including extend_months).")
        fore_d_ref = fore_d_ref.isel(time=keep)
        t_days = fore_d_ref["time"].values.astype("datetime64[D]")
        lead_arr = lead_arr[keep]

        # map to DOY in 2001 for clim lookup
        doys = np.array([doy_of_2001_from_yyyy_mm_dd(t, self._max_doy) for t in t_days], dtype=np.int32)
        if (doys > self._max_doy).any():
            raise RuntimeError("Found DOY outside climatology lookup.")
        clim_obs_all = self._clim_lookup[doys, :, :]

        lead_used = np.minimum(lead_arr, n_hind_leads).astype(np.int32)

        corr_np = fore_d_ref.values.astype(np.float32, copy=True)
        hind_np = hind_m.values.astype(np.float32)
        delta_clip = np.float32(self.cfg.clip_delta_c)

        # Apply correction lead-wise
        for L in range(1, n_hind_leads + 1):
            idx_all = np.where(lead_used == L)[0]
            if idx_all.size < 1:
                continue

            hind_mean_m = hind_np[L - 1, :, :]

            # For leads within hindcast range, month is taken from hindcast month coord.
            if (lead_arr[idx_all] <= n_hind_leads).all():
                month_tag = int(hm.isel(lead=L - 1).values)
                obs_mean_m = self._obs_month_mean[month_tag]

                bias_m = (obs_mean_m - hind_mean_m).astype(np.float32)
                corr_np[idx_all, :, :] = corr_np[idx_all, :, :] + bias_m[None, :, :]

                if self.cfg.clip_with_abs_limit:
                    clim_blk = clim_obs_all[idx_all, :, :].astype(np.float32)
                    cmin = (clim_blk - delta_clip).astype(np.float32)
                    cmax = (clim_blk + delta_clip).astype(np.float32)
                    corr_np[idx_all, :, :] = np.clip(corr_np[idx_all, :, :], cmin, cmax).astype(np.float32)

            else:
                # Extended leads: repeat last hindcast lead bias, but use OBS month mean for the real months.
                months_real = t_days[idx_all].astype("datetime64[M]")
                uniq_months = np.unique(months_real)

                for mM in uniq_months:
                    idx_m = idx_all[np.where(months_real == mM)[0]]
                    if idx_m.size < 1:
                        continue
                    mes_real = int(str(mM)[5:7])
                    obs_mean_m = self._obs_month_mean[mes_real]
                    bias_m = (obs_mean_m - hind_mean_m).astype(np.float32)
                    corr_np[idx_m, :, :] = corr_np[idx_m, :, :] + bias_m[None, :, :]

                    if self.cfg.clip_with_abs_limit:
                        clim_blk = clim_obs_all[idx_m, :, :].astype(np.float32)
                        cmin = (clim_blk - delta_clip).astype(np.float32)
                        cmax = (clim_blk + delta_clip).astype(np.float32)
                        corr_np[idx_m, :, :] = np.clip(corr_np[idx_m, :, :], cmin, cmax).astype(np.float32)

        corr = xr.DataArray(
            corr_np,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": fore_d_ref["time"].values,
                "latitude": fore_d_ref["latitude"].values,
                "longitude": fore_d_ref["longitude"].values,
            },
            name=self.cfg.var_name,
            attrs=fore_d_ref.attrs.copy(),
        )

        ds_out = corr.to_dataset(name=self.cfg.var_name)
        ds_out[self.cfg.var_name].attrs.setdefault("units", "degC")
        return ds_out

    # ------------------------------------------------------------------
    # Naming / traversal
    # ------------------------------------------------------------------
    @staticmethod
    def _init_stamp_from_filename(path: Path) -> str:
        m = re.search(r"(19|20)\d{8}", path.name)
        if not m:
            # fallback: accept YYYYMMDDHH (10 digits) too
            m = re.search(r"(19|20)\d{10}", path.name)
        if not m:
            raise ValueError(f"Could not find init_stamp in filename: {path.name}")
        return m.group(0)

    @staticmethod
    def _build_corr_outname_from_input(input_name: str, init_stamp: str) -> str:
        out = re.sub(r"_M\d{3}_", "_M100_", input_name)
        if "_M100_" not in out:
            out = out.replace(f"_{init_stamp}", f"_M100_{init_stamp}")
        if not out.endswith(".nc"):
            out = out + ".nc"
        return out

    def _forecast_files(self, in_dir: Path) -> List[Path]:
        files = sorted(in_dir.glob("*.nc"))
        return [fp for fp in files if "M000" in fp.name]

    def run(self) -> None:
        # DOY selection behavior:
        # - If forecast_root itself is a DOY folder (.../2024/001) -> process only that DOY
        # - If forecast_root is a parent folder (.../2024) -> pick the most recent DOY by numeric value (max)
        fr = self.forecast_root
        if fr.is_dir() and fr.name.isdigit() and len(fr.name) == 3:
            doy_dirs = [fr]
        else:
            doy_dirs_all = [p for p in fr.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 3]
            if not doy_dirs_all:
                raise RuntimeError(f"No DOY subfolders found under: {fr}")
            doy_dirs_all.sort(key=lambda p: int(p.name))
            doy_dirs = [doy_dirs_all[-1]]

        # --- LOG header (similar to precip module) ---
        output_base = self.out_root

        # Count forecast files across selected DOYs (usually 1 DOY)
        total_forecast_files = 0
        for doy_dir in doy_dirs:
            in_dir = doy_dir
            if self.cfg.subfolder:
                in_dir = in_dir / self.cfg.subfolder
                if not in_dir.exists():
                    continue
            total_forecast_files += len(self._forecast_files(in_dir))

        print(f"Total raw forecast files: {total_forecast_files}")
        print(f"Output base: {output_base}")
        print(f"Climatology mode: daily (lookup max_doy={self._max_doy})")

        # --- Main loop ---
        for doy_dir in doy_dirs:
            doy = doy_dir.name

            in_dir = doy_dir
            if self.cfg.subfolder:
                in_dir = in_dir / self.cfg.subfolder
                if not in_dir.exists():
                    continue

            out_dir = self.out_root
            ensure_dir(out_dir)

            weights_file = None
            if self.cfg.save_regrid_weights:
                weights_file = out_dir / self.cfg.regrid_cache_subdir

            hind = self._load_hindcast(doy, weights_file=weights_file)

            forecast_files = self._forecast_files(in_dir)
            if not forecast_files:
                continue

            for fp in forecast_files:
                init_stamp = self._init_stamp_from_filename(fp)
                out_name = self._build_corr_outname_from_input(fp.name, init_stamp)
                out_path = out_dir / out_name

                if self.cfg.skip_existing and out_path.exists():
                    continue

                print("\n=== Correcting (DAILY) ===")
                print("Input: ", str(fp))
                print("DOY:   ", doy)
                print("Output:", str(out_path))

                fore_daily, t_init, _ = self._preprocess_forecast_daily(fp, hind, weights_file=weights_file)
                ds_corr = self._correct_daily(fore_daily, hind, t_init=t_init)

                ds_corr, enc = self._format_corrected_output(ds_corr)
                ds_corr.to_netcdf(out_path, encoding=enc)

                print("[OK] Saved")

        print("\nDone.")