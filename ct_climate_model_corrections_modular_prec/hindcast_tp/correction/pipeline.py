from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, List, Optional

import re
import numpy as np
import xarray as xr

from .config import CorrectionConfig
from ..utils import extract_year_from_path

try:
    import xesmf as xe
except Exception as _e:  # pragma: no cover
    xe = None


class ForecastCorrectionPipeline:
    """
    Daily correction pipeline.

    Assumptions (aligned with the daily reference script you provided):
      - Forecast is 6/6h accumulated precipitation (sum -> daily total).
      - Observed climatology is daily (time >= 365/366), in mm/day (or meters if --to-mm).
      - Hindcast is monthly total per lead (mm/month) with:
          dims: lead, lat, lon
          coord: month (1..12) per lead

    Output structure:
      <out_root>/<var_name>/<year>/<DOY>/<standardized_name>.nc  (member=M100, init from forecast)
    """

    def __init__(self, cfg: CorrectionConfig):
        self.cfg = cfg
        try:
            self.out_year = extract_year_from_path(self.cfg.forecast_root)
        except Exception:
            self.out_year = extract_year_from_path(self.cfg.hindcast_root)

        self.out_base = self.cfg.out_root.expanduser().resolve()
        self.out_base.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.out_base / getattr(self.cfg, "regrid_cache_subdir", "cache")
        if getattr(self.cfg, "save_regrid_weights", True):
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load observed daily climatology once (fast lookup by DOY)
        self.clim_lookup, self.max_doy, self.lat_ref, self.lon_ref = self._load_obs_climatology_daily_fast(
            self.cfg.clim_file, self.cfg.var_name, self.cfg.to_mm
        )

        self._ny = int(self.lat_ref.size)
        self._nx = int(self.lon_ref.size)

        # Month cache (in-memory) for month_sum and idx_map
        self._month_cache_sum: Dict[int, np.ndarray] = {}
        self._month_cache_idx: Dict[int, np.ndarray] = {}

        # XESMF regridder (lazy)
        self._regridder = None

    # -------------------------------------------------------------------------
    # Coordinates helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _std_latlon(ds: xr.Dataset) -> xr.Dataset:
        # standardize to lat/lon coords
        rename = {}
        if "latitude" in ds.coords and "lat" not in ds.coords:
            rename["latitude"] = "lat"
        if "longitude" in ds.coords and "lon" not in ds.coords:
            rename["longitude"] = "lon"
        if rename:
            ds = ds.rename(rename)
        if "lat" not in ds.coords or "lon" not in ds.coords:
            raise ValueError(f"Missing lat/lon coords. Coords={list(ds.coords)}")
        return ds

    @staticmethod
    def _sort_latlon(ds: xr.Dataset) -> xr.Dataset:
        if "lat" in ds.coords:
            ds = ds.sortby("lat")
        if "lon" in ds.coords:
            ds = ds.sortby("lon")
        return ds

    @staticmethod
    def _adjust_lon(ds: xr.Dataset, lon_ref: xr.DataArray) -> xr.Dataset:
        ds = ForecastCorrectionPipeline._std_latlon(ds)
        if "lon" not in ds.coords:
            return ds

        ds_lon_max = float(ds["lon"].max())
        ref_lon_min = float(lon_ref.min())

        if ds_lon_max > 180 and ref_lon_min < 0:
            lon_adj = ((ds["lon"] + 180) % 360) - 180
            ds = ds.assign_coords(lon=lon_adj).sortby("lon")
        return ds

    # -------------------------------------------------------------------------
    # Observed climatology daily (one read)
    # -------------------------------------------------------------------------
    def _load_obs_climatology_daily_fast(
        self, clim_file: Path, var_name: str, to_mm: bool
    ) -> Tuple[np.ndarray, int, xr.DataArray, xr.DataArray]:
        ds = xr.open_dataset(clim_file)
        ds = self._std_latlon(ds)
        ds = self._sort_latlon(ds)

        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in climatology file: {clim_file}")

        da = ds[var_name]
        if "time" not in da.dims:
            raise ValueError("Observed climatology must have 'time' dimension (daily climatology).")

        if to_mm:
            da = da * 1000.0

        da = da.transpose("time", "lat", "lon")

        n = int(da.sizes["time"])
        if n < 365:
            raise ValueError(f"Daily climatology expected >=365 steps, got time={n} in {clim_file}")

        # SINGLE READ
        da = da.astype("float32").load()

        doys = da["time"].dt.dayofyear.values.astype(np.int32)
        max_doy = int(doys.max())
        ny = int(da.sizes["lat"])
        nx = int(da.sizes["lon"])

        clim_lookup = np.zeros((max_doy + 1, ny, nx), dtype=np.float32)
        filled = np.zeros((max_doy + 1,), dtype=bool)

        vals = da.values  # (time, y, x)
        for i in range(vals.shape[0]):
            d = int(doys[i])
            if 1 <= d <= max_doy and (not filled[d]):
                clim_lookup[d, :, :] = vals[i, :, :]
                filled[d] = True

        nfilled = int(filled.sum()) - int(filled[0])
        if nfilled < 365:
            raise ValueError(f"Daily climatology insufficient: filled {nfilled} DOYs (expected >=365)")

        return clim_lookup, max_doy, ds["lat"], ds["lon"]

    # -------------------------------------------------------------------------
    # Month cache helpers (based on 2001 calendar)
    # -------------------------------------------------------------------------
    @staticmethod
    def _list_doys_in_month_2001(month: int) -> List[int]:
        start = np.datetime64(f"2001-{month:02d}-01")
        end = np.datetime64("2002-01-01") if month == 12 else np.datetime64(f"2001-{month+1:02d}-01")
        ndays = int((end - start).astype("timedelta64[D]").astype(np.int64))
        doy0 = int(((start.astype("datetime64[D]") - np.datetime64("2001-01-01")) / np.timedelta64(1, "D")) + 1)
        return [doy0 + i for i in range(ndays)]

    @staticmethod
    def _doy_of_2001_from_day(tday: np.datetime64) -> int:
        s = str(tday.astype("datetime64[D]"))  # YYYY-MM-DD
        mm = int(s[5:7])
        dd = int(s[8:10])

        # climatologia é 365 (não bissexto): colapsa 29/fev para 28/fev
        if mm == 2 and dd == 29:
            dd = 28

        d2001 = np.datetime64(f"2001-{mm:02d}-{dd:02d}").astype("datetime64[D]")
        return int(((d2001 - np.datetime64("2001-01-01")) / np.timedelta64(1, "D")) + 1)

    def _get_month_sum_and_idxmap(self, month: int) -> Tuple[np.ndarray, np.ndarray]:
        if month in self._month_cache_sum:
            return self._month_cache_sum[month], self._month_cache_idx[month]

        doys_m = np.array(self._list_doys_in_month_2001(month), dtype=np.int32)
        doys_m = doys_m[doys_m <= self.max_doy]

        if doys_m.size < 1:
            month_sum = np.zeros((self._ny, self._nx), dtype=np.float32)
            idx_map = np.full((self.max_doy + 1,), -1, dtype=np.int32)
        else:
            clim_days = self.clim_lookup[doys_m, :, :]  # (nd, y, x)
            month_sum = np.sum(clim_days, axis=0).astype(np.float32)

            idx_map = np.full((self.max_doy + 1,), -1, dtype=np.int32)
            for i, d in enumerate(doys_m):
                idx_map[int(d)] = i

        self._month_cache_sum[month] = month_sum
        self._month_cache_idx[month] = idx_map
        return month_sum, idx_map

    # -------------------------------------------------------------------------
    # Regrid (XESMF bilinear; fallback to interp if xesmf is unavailable)
    # -------------------------------------------------------------------------
    def _build_regridder(self, ds_in: xr.Dataset) -> None:
        if xe is None:
            self._regridder = None
            return

        ds_in = self._std_latlon(ds_in)
        ds_in = self._sort_latlon(ds_in)

        grid_in = xr.Dataset(
            {"lat": (("lat",), ds_in["lat"].values),
             "lon": (("lon",), ds_in["lon"].values)}
        )
        grid_out = xr.Dataset(
            {"lat": (("lat",), self.lat_ref.values),
             "lon": (("lon",), self.lon_ref.values)}
        )

        if getattr(self.cfg, "save_regrid_weights", True):
            wfile = self.cache_dir / "regrid_weights.nc"
            reuse = wfile.exists()
            self._regridder = xe.Regridder(
                grid_in, grid_out, "bilinear",
                filename=str(wfile),
                reuse_weights=reuse
            )
        else:
            self._regridder = xe.Regridder(
                grid_in, grid_out, "bilinear"
            )

    def _regrid_to_ref(self, ds: xr.Dataset) -> xr.Dataset:
        ds = self._std_latlon(ds)
        ds = self._sort_latlon(ds)
        ds = self._adjust_lon(ds, self.lon_ref)

        if xe is None:
            # fallback (slower, but avoids hard dependency)
            return ds.interp(lat=self.lat_ref, lon=self.lon_ref)

        ny_in = int(ds["lat"].size)
        nx_in = int(ds["lon"].size)
        ny_ref = int(self.lat_ref.size)
        nx_ref = int(self.lon_ref.size)

        # weights file dependente do shape da grade
        wfile = (
            self.cache_dir
            / f"regrid_weights_in{ny_in}x{nx_in}_out{ny_ref}x{nx_ref}.nc"
        )

        rebuild = False

        if self._regridder is None:
            rebuild = True
        else:
            old_shape = getattr(self._regridder, "_input_shape", None)
            if old_shape != (ny_in, nx_in):
                rebuild = True

        if rebuild:
            grid_in = xr.Dataset(
                {
                    "lat": (("lat",), ds["lat"].values),
                    "lon": (("lon",), ds["lon"].values),
                }
            )

            grid_out = xr.Dataset(
                {
                    "lat": (("lat",), self.lat_ref.values),
                    "lon": (("lon",), self.lon_ref.values),
                }
            )

            if getattr(self.cfg, "save_regrid_weights", True):
                self._regridder = xe.Regridder(
                    grid_in,
                    grid_out,
                    "bilinear",
                    filename=str(wfile),
                    reuse_weights=wfile.exists(),
                )
            else:
                self._regridder = xe.Regridder(
                    grid_in,
                    grid_out,
                    "bilinear",
                )

            # guarda shape da grade de entrada
            self._regridder._input_shape = (ny_in, nx_in)

        # Force C-contiguous for all data_vars (xesmf performance warning)
        ds = ds.copy()
        for v in list(ds.data_vars):
            if ("lat" in ds[v].dims) and ("lon" in ds[v].dims):
                ds[v].data = np.asarray(ds[v].values, order="C")

        out = self._regridder(ds)
        return self._sort_latlon(out)

    # -------------------------------------------------------------------------
    # Output name standardization (based on forecast filename)
    # -------------------------------------------------------------------------
    @staticmethod
    def _init_stamp_from_filename_strict(path: Path) -> str:
        """
        Extract YYYYMMDDHH from forecast filename.
        Example:
          cmcc_subseas_glo_total_precipitation_M000_2024010100.nc -> 2024010100
        """
        m = re.search(r"(19|20)\d{8}", path.name)
        if not m:
            raise ValueError(f"Não achei init_stamp YYYYMMDDHH no nome do arquivo: {path.name}")
        return m.group(0)

    @staticmethod
    def _build_corr_outname_from_input(input_name: str, init_stamp: str) -> str:
        """
        Standardize corrected output name:
          - If filename contains _M###_ => replace with _M100_
          - If it does not contain _M###_ => insert _M100_ before init_stamp
        Keeps the rest of the prefix the same (model-specific).
        """
        out = re.sub(r"_M\d{3}_", "_M100_", input_name)

        if "_M100_" not in out:
            out = out.replace(f"_{init_stamp}", f"_M100_{init_stamp}")

        if not out.endswith(".nc"):
            out = out + ".nc"

        return out

    # -------------------------------------------------------------------------
    # IO helpers
    # -------------------------------------------------------------------------
    def _load_hindcast(self, doy_subdir: str) -> xr.Dataset:
        files = [
            fp for fp in sorted(self.cfg.forecast_root.rglob("*.nc"))
            if "M000" in fp.name
        ]
        if not files:
            raise FileNotFoundError(f"No hindcast .nc files found in: {subdir}")

        ds = xr.open_dataset(files[0])
        ds = self._std_latlon(ds)
        ds = self._sort_latlon(ds)

        if self.cfg.var_name not in ds:
            raise KeyError(f"Variable '{self.cfg.var_name}' not found in hindcast file: {files[0]}")
        if "lead" not in ds.dims:
            raise ValueError(f"Hindcast must have 'lead' dimension. Dims: {ds.dims}")
        if "month" not in ds.coords:
            raise ValueError("Hindcast must have 'month' coordinate (month per lead).")

        if self.cfg.to_mm:
            ds[self.cfg.var_name] = ds[self.cfg.var_name] * 1000.0

        # regrid hindcast to ref grid (force C-contiguous for performance)
        ds_in = ds[[self.cfg.var_name, "month"]].copy()
        ds_in[self.cfg.var_name].data = np.ascontiguousarray(ds_in[self.cfg.var_name].data)

        ds_rg = self._regrid_to_ref(ds_in)
        return ds_rg

    # -------------------------------------------------------------------------
    # Forecast 6/6h -> daily total (mm/day)
    # -------------------------------------------------------------------------
    @staticmethod
    def _forecast_to_daily_accum_mm(da_mm: xr.DataArray) -> xr.DataArray:
        if "time" not in da_mm.dims:
            raise ValueError("Forecast variable must have 'time' dimension.")
        out = da_mm.resample(time="1D").sum(keep_attrs=True)
        out = out.astype("float32")
        out.attrs = da_mm.attrs.copy()
        out.attrs["units"] = "mm/day"
        return out

    def _preprocess_forecast_daily(self, forecast_path: Path, hind: xr.Dataset) -> xr.DataArray:
        ds = xr.open_dataset(forecast_path)
        ds = self._std_latlon(ds)
        ds = self._sort_latlon(ds)

        if self.cfg.var_name not in ds:
            raise KeyError(f"Variable '{self.cfg.var_name}' not found in forecast file: {forecast_path}")

        if "time" not in ds.coords or ds["time"].size < 1:
            raise ValueError("Forecast must have non-empty 'time' coordinate.")

        da = ds[self.cfg.var_name]
        if self.cfg.to_mm:
            da = da * 1000.0

        # Cut 6/6h steps by monthly-lead range BEFORE daily aggregation
        t_init = ds["time"].values[0].astype("datetime64[D]")
        t_6h_days = ds["time"].values.astype("datetime64[D]")
        lead0_6h = (t_6h_days.astype("datetime64[M]").astype(int) - t_init.astype("datetime64[M]").astype(int))
        lead_arr_6h = (lead0_6h + 1).astype(np.int32)

        n_hind_leads = int(hind[self.cfg.var_name].sizes["lead"])
        valid_idx_6h = np.where((lead_arr_6h >= 1) & (lead_arr_6h <= n_hind_leads))[0]
        if valid_idx_6h.size < 1:
            raise RuntimeError("No timesteps are within hindcast lead range.")

        da_slice = da.isel(time=valid_idx_6h)

        # Aggregate to daily ONLY if forecast is sub-daily (e.g., 6/6h).
        # If forecast is already daily (24h), use it as-is.
        t = da_slice["time"].values.astype("datetime64[h]")
        if t.size < 2:
            raise ValueError("Forecast has < 2 timesteps; cannot infer frequency.")

        dt = np.unique(np.diff(t).astype(int))  # hours

        if dt.size == 1 and dt[0] == 24:
            da_daily = da_slice.astype("float32")
            da_daily.attrs = da_slice.attrs.copy()
            da_daily.attrs["units"] = "mm/day"
        else:
            da_daily = self._forecast_to_daily_accum_mm(da_slice)

        # Regrid to ref
        ds_daily = da_daily.to_dataset(name=self.cfg.var_name).copy()

        # força memória C-contiguous (remove warning do xesmf)
        ds_daily[self.cfg.var_name].data = np.ascontiguousarray(ds_daily[self.cfg.var_name].data)

        ds_daily_rg = self._regrid_to_ref(ds_daily)
        out = ds_daily_rg[self.cfg.var_name].astype("float32")

        return out

    # -------------------------------------------------------------------------
    # Correction (daily)
    # -------------------------------------------------------------------------
    def _correct_daily(self, fore_d_ref: xr.DataArray, hind: xr.Dataset, t_init: np.datetime64) -> xr.Dataset:
        hind_m = hind[self.cfg.var_name].astype("float32")   # (lead, lat, lon) monthly total
        hm = hind["month"]
        n_hind_leads = int(hind_m.sizes["lead"])

        # Map each day to lead (month relative to init) and DOY (2001)
        t_days = fore_d_ref["time"].values.astype("datetime64[D]")
        lead0 = (t_days.astype("datetime64[M]").astype(int) - t_init.astype("datetime64[M]").astype(int))
        lead_arr = (lead0 + 1).astype(np.int32)

        doys = np.array([self._doy_of_2001_from_day(t) for t in t_days], dtype=np.int32)

        # Observed clim stack for all selected days (fast gather)
        if (doys > self.max_doy).any():
            raise ValueError(f"DOY out of range for climatology lookup (max={self.max_doy}).")
        clim_all = self.clim_lookup[doys, :, :]  # (T, y, x) mm/day

        corr_np = fore_d_ref.values.astype(np.float32, copy=True)  # (T, y, x)
        hind_np = hind_m.values.astype(np.float32)                 # (lead, y, x)

        # parameters (same defaults as the reference script)
        denom_min = float(getattr(self.cfg, "denom_min", 1e-3))
        alpha = float(getattr(self.cfg, "alpha", denom_min))
        p = float(getattr(self.cfg, "limit_p", 0.30))

        for L in range(1, n_hind_leads + 1):
            idx = np.where(lead_arr == L)[0]
            if idx.size < 1:
                continue

            mes_hind = int(hm.isel(lead=L - 1).values)
            month_sum, _ = self._get_month_sum_and_idxmap(mes_hind)

            doys_sel = doys[idx]
            if (doys_sel > self.max_doy).any():
                nd = float(idx.size)
                w_stack = np.full((idx.size, self._ny, self._nx), 1.0 / nd, dtype=np.float32)
            else:
                clim_sel = self.clim_lookup[doys_sel, :, :]  # (n, y, x)
                nd = float(idx.size)
                den = month_sum[None, :, :].astype(np.float32, copy=False)
                num = clim_sel.astype(np.float32, copy=False)

                # default uniform weights (fallback)
                w_stack = np.full((num.shape[0], self._ny, self._nx), 1.0 / nd, dtype=np.float32)

                # safe divide (avoids warnings / NaN)
                mask = den > 0
                np.divide(num, den, out=w_stack, where=mask)
                w_stack = w_stack.astype(np.float32, copy=False)

            hind_total_m = hind_np[L - 1, :, :]              # (y, x) mm/month
            hind_est_d = hind_total_m[None, :, :] * w_stack  # (n, y, x) mm/day

            clim_obs_d  = clim_all[idx, :, :]       # (n, y, x)
            clim_hind_d = hind_est_d                # (n, y, x)
            fore_d      = corr_np[idx, :, :]        # (n, y, x)

            # correction: multiplicative, fallback to additive, and clamp around obs clim
            factor = (clim_obs_d + alpha) / (clim_hind_d + alpha)
            corr_ratio = fore_d * factor
            corr_add   = fore_d + (clim_obs_d - clim_hind_d)

            corr_candidate = np.where(
                clim_hind_d < denom_min,
                corr_add,
                corr_ratio
            ).astype(np.float32)

            # if obs clim is 0, do not correct (keep forecast)
            corr_candidate = np.where(
                clim_obs_d == 0,
                fore_d,
                corr_candidate
            ).astype(np.float32)

            # clamp around obs clim (±p)
            cmin = (1.0 - p) * clim_obs_d
            cmax = (1.0 + p) * clim_obs_d
            corr_limited = np.clip(corr_candidate, cmin, cmax).astype(np.float32)

            corr_np[idx, :, :] = corr_limited

        corr_np = np.clip(corr_np, 0, None).astype(np.float32)

        corr = xr.DataArray(
            corr_np,
            dims=("time", "lat", "lon"),
            coords={
                "time": fore_d_ref["time"].values,
                "lat": fore_d_ref["lat"].values,
                "lon": fore_d_ref["lon"].values,
            },
            name=self.cfg.var_name,
            attrs=fore_d_ref.attrs.copy(),
        )

        return corr.to_dataset(name=self.cfg.var_name)

    # -------------------------------------------------------------------------
    # Files / output
    # -------------------------------------------------------------------------
    def _forecast_files(self) -> List[Path]:
        if self.cfg.subfolder:
            d = (self.cfg.forecast_root / self.cfg.subfolder)
            return sorted(d.glob("*.nc"))
        return sorted(self.cfg.forecast_root.rglob("*.nc"))

    def _out_path(self, forecast_path: Path, doy_subdir: str) -> Path:
        out_dir = self.out_base
        out_dir.mkdir(parents=True, exist_ok=True)

        init_stamp = self._init_stamp_from_filename_strict(forecast_path)
        out_name = self._build_corr_outname_from_input(forecast_path.name, init_stamp)

        return out_dir / out_name

    # -------------------------------------------------------------------------
    # Run
    # -------------------------------------------------------------------------
    def run(self) -> None:
        files = self._forecast_files()
        if not files:
            raise FileNotFoundError("No forecast .nc files found under forecast_root/subfolder.")

        print(f"Total raw forecast files: {len(files)}")
        print(f"Output base: {self.out_base}")
        print(f"Climatology mode: daily (lookup max_doy={self.max_doy})")

        for fp in files:
            doy = fp.parent.name
            out_fp = self._out_path(fp, doy)

            if self.cfg.skip_existing and out_fp.exists() and out_fp.stat().st_size > 0:
                print(f"\n=== SKIP existing ===\n{out_fp}")
                continue

            print(f"\n=== Correcting (DAILY) ===\nInput:  {fp}\nDOY:    {doy}\nOutput: {out_fp}")

            hind = self._load_hindcast(doy)

            # preprocess forecast -> daily + regrid
            fore_d_ref = self._preprocess_forecast_daily(fp, hind)

            # init day (for lead mapping)
            ds0 = xr.open_dataset(fp)
            t_init = ds0["time"].values[0].astype("datetime64[D]")

            out = self._correct_daily(fore_d_ref, hind, t_init)
            out.to_netcdf(out_fp)

            print("[OK] Saved")

        print("\nDone.")
