#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecast pipeline (TEMPERATURE ONLY):
- Preprocess forecasts (remove incomplete last month when applicable, monthly aggregation when applicable, regrid to reference grid)
- Apply LS-Add correction using hindcast climatology and observed climatology
- Save corrected monthly forecasts with the SAME output structure as hindcast:

  <OUT_BASE>/<VAR>/<YEAR>/<DOY>/...

No extra folders.

Temperature handling:
- Detect variable type from paths: 2m_air_temperature_min / _med / _max
- Input forecast NetCDF may contain ANY of these vars (auto-detected):
    2m_air_temperature
    2m_air_temperature_min / 2m_air_temperature_med / 2m_air_temperature_max
    t2m / 2t / 2m_temperature
- Monthly aggregation when time exists:
    min -> monthly minimum
    med -> monthly mean
    max -> monthly maximum
- Convert K -> °C only if it looks like Kelvin (mean > 100 or units indicate Kelvin)
- Rename output variable to match hindcast naming:
    2m_air_temperature_min / 2m_air_temperature_med / 2m_air_temperature_max

Correction logic (LS-Add) is kept as-is:
  corrected = forecast + (clim_obs(month_of_lead) - clim_hindcast(lead))

Climatology handling (UPDATED):
- --clim-file must point to the climatology ROOT directory, e.g. /climatology
- The script will pick the climatology file for the detected var:
    <clim-root>/<VAR>/**/*.nc

Hindcast handling (UPDATED):
- --hindcast-root must point to the hindcast BASE directory (not var/year)
- The script will find hindcast here:
    <hindcast-root>/<VAR>/<YEAR>/<DOY>/*.nc
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =========================
# HELPERS (same as hindcast scripts)
# =========================
def extract_year_from_path(path: Path) -> int:
    for part in path.parts[::-1]:
        if len(part) == 4 and part.isdigit():
            y = int(part)
            if 1900 <= y <= 2100:
                return y
    raise ValueError(f"Could not find a 4-digit year in path: {path}")


def find_first_nc_in_dir_recursive(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    if p.is_file() and p.suffix.lower() == ".nc":
        return p
    if not p.is_dir():
        raise FileNotFoundError(f"Not a directory: {p}")
    direct = sorted(p.glob("*.nc"))
    if direct:
        return direct[0]
    rec = sorted(p.rglob("*.nc"))
    if rec:
        return rec[0]
    raise FileNotFoundError(f"No .nc found under: {p}")


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class CorrectionConfig:
    forecast_root: Path
    hindcast_root: Path
    clim_file: Path
    out_root: Path

    # If None, auto-detect from paths (recommended)
    var_name: Optional[str] = None

    subfolder: Optional[str] = None
    min_steps_last_month: int = 2


# =========================
# PIPELINE
# =========================
class ForecastCorrectionPipeline:
    # Temperature-only tokens (exact)
    VAR_TOKENS = (
        "2m_air_temperature_min",
        "2m_air_temperature_max",
        "2m_air_temperature",
    )

    # Raw forecast variable name (preferred, but not required)
    RAW_TEMP_VAR = "2m_air_temperature"

    def __init__(self, cfg: CorrectionConfig):
        self.cfg = cfg

        # Year comes from forecast path (same as earlier approach)
        self.out_year = extract_year_from_path(self.cfg.forecast_root)

        # Output base (final structure applied at save time)
        self.out_root_base = self.cfg.out_root.expanduser().resolve()

        # Load reference grid (lat/lon) once
        _, self.lat_ref, self.lon_ref = self._load_ref_grid(self.cfg.clim_file)

    # -------------------------
    # Variable detection (temperature-only)
    # -------------------------
    @classmethod
    def _detect_var_from_path(cls, p: Path) -> Optional[str]:
        s = p.as_posix().lower()
        for tok in cls.VAR_TOKENS:
            if tok.lower() in s:
                return tok
        return None

    def _resolve_var_name(self, forecast_path: Path) -> str:
        # Priority: explicit CLI var -> forecast path -> hindcast_root -> clim_file
        if self.cfg.var_name:
            if self.cfg.var_name not in self.VAR_TOKENS:
                raise ValueError(f"--var-name must be one of: {', '.join(self.VAR_TOKENS)}")
            return self.cfg.var_name

        v = self._detect_var_from_path(forecast_path)
        if v:
            return v
        v = self._detect_var_from_path(self.cfg.hindcast_root)
        if v:
            return v
        v = self._detect_var_from_path(self.cfg.clim_file)
        if v:
            return v

        raise RuntimeError(
            "Could not detect temperature variable name. Expected one of: "
            f"{', '.join(self.VAR_TOKENS)} present in forecast path, hindcast_root, or clim_file. "
            "Or pass --var-name explicitly."
        )

    @staticmethod
    def _temp_kind_from_var(var_name: str) -> str:
        if var_name.endswith("_min"):
            return "min"
        if var_name == "2m_air_temperature":
            return "mean"
        if var_name.endswith("_max"):
            return "max"
        raise ValueError(f"Invalid temperature var_name: {var_name}")

    # -------------------------
    # Basic helpers
    # -------------------------
    @staticmethod
    def _standardize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
        rename = {}
        if "latitude" in ds.coords:
            rename["latitude"] = "lat"
        if "longitude" in ds.coords:
            rename["longitude"] = "lon"
        if rename:
            ds = ds.rename(rename)
        return ds

    @staticmethod
    def _adjust_lon_to_reference(ds: xr.Dataset, lon_ref: xr.DataArray) -> xr.Dataset:
        ds = ForecastCorrectionPipeline._standardize_lat_lon(ds)
        if "lon" not in ds.coords:
            return ds

        ds_lon_max = float(ds["lon"].max())
        ref_lon_min = float(lon_ref.min())

        if ds_lon_max > 180 and ref_lon_min < 0:
            lon_adj = ((ds["lon"] + 180) % 360) - 180
            ds = ds.assign_coords(lon=lon_adj).sortby("lon")

        return ds

    @staticmethod
    def _remove_incomplete_last_month(ds: xr.Dataset, min_steps_last_month: int) -> xr.Dataset:
        if "time" not in ds.dims:
            raise ValueError("Dataset does not have a 'time' dimension.")

        time_index = ds["time"].to_index()
        month_period = time_index.to_period("M")

        last_month = month_period[-1]
        mask_last = (month_period == last_month)
        n_steps_last = int(mask_last.sum())

        if n_steps_last < min_steps_last_month:
            times_ok = ds["time"].isel(time=~xr.DataArray(mask_last, dims=["time"]))
            ds = ds.sel(time=times_ok)

        return ds

    # -------------------------
    # Climatology loaders (UPDATED: clim_file is a ROOT dir)
    # -------------------------
    def _resolve_clim_file_for_var(self, var_name: str) -> Path:
        clim_root = self.cfg.clim_file
        if clim_root.is_file():
            # Backward compatibility: if user passed a file directly
            return clim_root
        clim_var_dir = clim_root / var_name
        return find_first_nc_in_dir_recursive(clim_var_dir)

    def _load_ref_grid(self, clim_root_or_file: Path) -> Tuple[None, xr.DataArray, xr.DataArray]:
        # Load ref grid from the resolved clim file for any variable:
        # - if clim_root_or_file is a file: use it
        # - if it's a dir: pick first nc under it (any var) just to get lat/lon
        if clim_root_or_file.is_dir():
            clim_file_any = find_first_nc_in_dir_recursive(clim_root_or_file)
        else:
            clim_file_any = clim_root_or_file

        ds = xr.open_dataset(clim_file_any)
        ds = self._standardize_lat_lon(ds)
        lat_ref = ds["lat"]
        lon_ref = ds["lon"]
        return None, lat_ref, lon_ref

    def _get_clim_obs(self, var_name: str) -> Dict[int, xr.DataArray]:
        clim_file = self._resolve_clim_file_for_var(var_name)

        ds = xr.open_dataset(clim_file)
        ds = self._standardize_lat_lon(ds)

        if var_name not in ds:
            # If the file has exactly one variable, accept it (robustness)
            if len(ds.data_vars) == 1:
                only = list(ds.data_vars)[0]
                da = ds[only]
            else:
                raise KeyError(f"Variable '{var_name}' not found in {clim_file}. Vars: {list(ds.data_vars)}")
        else:
            da = ds[var_name]

        if "time" not in da.dims or da.sizes["time"] != 12:
            raise ValueError(f"Observed climatology must have time=12 (one per month). File: {clim_file}")

        # Observed temp could be K or °C; assume K if mean > 100
        try:
            mval = float(da.mean().values)
            if mval > 100.0:
                da = da - 273.15
        except Exception:
            pass

        clim_obs: Dict[int, xr.DataArray] = {}
        for t in da.time:
            month = int(t.dt.month.values)
            clim_obs[month] = da.sel(time=t).drop_vars("time")

        return clim_obs

    # -------------------------
    # Hindcast loader (UPDATED: hindcast_root is BASE, build var/year/doy)
    # -------------------------
    def _load_hindcast_for_subfolder(self, julian_subdir: str, var_name: str) -> xr.Dataset:
        # Expected structure:
        # <hindcast-root>/<VAR>/<YEAR>/<DOY>/*.nc
        subdir = self.cfg.hindcast_root / var_name / str(self.out_year) / julian_subdir
        files = sorted(subdir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No hindcast files found in: {subdir}")

        ds = xr.open_dataset(files[0])
        ds = self._standardize_lat_lon(ds)

        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in: {files[0]}")

        if "lead" not in ds.dims:
            raise ValueError(f"Hindcast must have a 'lead' dimension. Dims: {ds.dims}")

        if "month" not in ds.coords:
            raise ValueError("Hindcast must have a 'month' coordinate associated with 'lead'.")

        # Hindcast temperature is assumed already °C
        return ds

    # -------------------------
    # Forecast preprocessing (temperature-only)
    # -------------------------
    def _preprocess_forecast(self, forecast_path: Path, out_var_name: str) -> xr.DataArray:
        """
        Open forecast, remove incomplete last month when time exists, monthly aggregate when time exists,
        regrid to reference grid, convert K->°C when needed, and rename to out_var_name.

        Input forecast NetCDF may contain:
          - 2m_air_temperature (raw, Kelvin)
          - or already named 2m_air_temperature_min/med/max
          - or t2m/2t/2m_temperature

        Returns:
          DataArray (time, lat, lon) named as out_var_name.
        """
        ds = xr.open_dataset(forecast_path)
        ds = self._standardize_lat_lon(ds)

        candidates = [
            out_var_name,          # if it already comes named min/med/max
            self.RAW_TEMP_VAR,     # preferred raw
            "t2m",
            "2t",
            "2m_temperature",
        ]

        in_var = None
        for cand in candidates:
            if cand in ds.data_vars:
                in_var = cand
                break

        if in_var is None:
            raise KeyError(
                f"No temperature variable found in: {forecast_path}\n"
                f"Tried: {candidates}\n"
                f"Available variables: {list(ds.data_vars)}"
            )

        # If dataset has time, apply remove-last-month + monthly aggregation
        if "time" in ds.dims:
            ds = self._remove_incomplete_last_month(ds, self.cfg.min_steps_last_month)
            if ds["time"].size == 0:
                raise ValueError("No timestamps left after removing incomplete last month.")

            kind = self._temp_kind_from_var(out_var_name)

            if kind == "min":
                ds_m = ds.resample(time="MS").min()
            elif kind == "max":
                ds_m = ds.resample(time="MS").max()
            else:  # med
                ds_m = ds.resample(time="MS").mean()
        else:
            ds_m = ds

        # Adjust longitude and regrid
        ds_m = self._adjust_lon_to_reference(ds_m, self.lon_ref)

        if "lat" not in ds_m.coords or "lon" not in ds_m.coords:
            raise ValueError(f"Dataset missing lat/lon coordinates after standardization. Coords: {list(ds_m.coords)}")

        ds_m = ds_m.interp(lat=self.lat_ref, lon=self.lon_ref)

        da = ds_m[in_var]

        # Convert K -> °C only if it looks Kelvin
        try:
            mval = float(da.mean().values)
            looks_kelvin = (mval > 100.0)
        except Exception:
            units = str(da.attrs.get("units", "")).lower()
            looks_kelvin = units in ("k", "kelvin")

        if looks_kelvin:
            da = da - 273.15

        # Enforce output naming to hindcast-style
        da.name = out_var_name
        da.attrs["units"] = "degC"

        # Keep original attributes where possible
        try:
            da.attrs.update(ds[in_var].attrs)
        except Exception:
            pass

        return da

    # -------------------------
    # Correction (DO NOT CHANGE LOGIC)
    # -------------------------
    def _ls_add_correction(self, fore: xr.DataArray, ds_h: xr.Dataset, clim_obs: Dict[int, xr.DataArray], var_name: str) -> xr.Dataset:
        """
        Apply LS-Add correction:
          corrected = forecast + (clim_obs(month_of_lead) - clim_hindcast(lead))
        """
        hind = ds_h[var_name]        # (lead, lat, lon)
        hind_months = ds_h["month"]  # (lead,)

        if not np.array_equal(fore.lat.values, hind.lat.values):
            raise ValueError("Hindcast latitude != preprocessed forecast latitude.")
        if not np.array_equal(fore.lon.values, hind.lon.values):
            raise ValueError("Hindcast longitude != preprocessed forecast longitude.")

        n_leads = hind.sizes["lead"]
        n_corr = min(n_leads, fore.sizes["time"] if "time" in fore.dims else 0)

        if n_corr == 0:
            raise ValueError("Forecast DataArray has no 'time' dimension or no time steps to correct.")

        corr = fore.copy()
        for idx in range(n_corr):
            month = int(hind_months.isel(lead=idx).values)
            if month not in clim_obs:
                raise KeyError(f"Month {month} not found in observed climatology (1..12).")

            clim_obs_m = clim_obs[month]
            clim_hind_m = hind.isel(lead=idx)
            bias = clim_obs_m - clim_hind_m

            corr[idx, :, :] = fore[idx, :, :] + bias

        corr.name = var_name
        return corr.to_dataset(name=var_name)

    # -------------------------
    # File listing & saving (hindcast-style output structure)
    # -------------------------
    def _list_forecast_files(self) -> list[Path]:
        if self.cfg.subfolder:
            target_dir = self.cfg.forecast_root / self.cfg.subfolder
            files = sorted(target_dir.glob("*.nc"))
        else:
            files = sorted(self.cfg.forecast_root.rglob("*.nc"))
        return files

    def _output_path_for_forecast(self, forecast_path: Path, var_name: str, julian_subdir: str) -> Path:
        """
        Enforce output structure (same as hindcast):
          <OUT_BASE>/<VAR>/<YEAR>/<DOY>/<original_filename>.nc
        """
        out_dir = self.out_root_base / var_name / str(self.out_year) / julian_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / forecast_path.name

    # -------------------------
    # Public entrypoint
    # -------------------------
    def run(self) -> None:
        forecast_files = self._list_forecast_files()
        if not forecast_files:
            raise FileNotFoundError("No forecast files found under the given forecast root/subfolder.")

        print(f"Total raw forecast files: {len(forecast_files)}")

        for fpath in forecast_files:
            print(f"\n=== Processing file: {fpath} ===")
            julian_subdir = fpath.parent.name  # expects DOY folder name like 001/032/335...

            # Determine output variable name (min/med/max) from paths or CLI
            var_name = self._resolve_var_name(fpath)

            # Load observed climatology for this var
            clim_obs = self._get_clim_obs(var_name)

            # 1) Preprocess forecast (monthly + regrid) and force output variable naming
            fore_m = self._preprocess_forecast(fpath, var_name)

            # 2) Load matching hindcast (AUTO: base/var/year/doy)
            ds_h = self._load_hindcast_for_subfolder(julian_subdir, var_name)

            # 3) Correct (logic unchanged)
            ds_out = self._ls_add_correction(fore_m, ds_h, clim_obs, var_name)

            # 4) Save (hindcast structure)
            out_path = self._output_path_for_forecast(fpath, var_name, julian_subdir)
            ds_out.to_netcdf(out_path)
            print(f"[OK] Saved corrected forecast: {out_path}")

        print("\nDone.")


# =========================
# CLI
# =========================
def parse_args() -> CorrectionConfig:
    p = argparse.ArgumentParser(
        description="TEMPERATURE ONLY: preprocess forecasts (monthly + regrid) and apply LS-Add correction using hindcast and observed climatology."
    )
    p.add_argument("--forecast-root", type=Path, required=True, help="Root directory containing forecast NetCDF files.")
    p.add_argument("--hindcast-root", type=Path, required=True,
                   help="Hindcast BASE directory. Script will use <hindcast-root>/<VAR>/<YEAR>/<DOY>/*.nc")
    p.add_argument("--clim-file", type=Path, required=True,
                   help="Climatology ROOT directory (recommended) containing subfolders per var (e.g. /climatology/<VAR>/**/*.nc). "
                        "You may also pass a single .nc file (legacy).")
    p.add_argument("--out-root", type=Path, required=True, help="Output base directory for corrected forecasts.")
    p.add_argument(
        "--var-name",
        type=str,
        default=None,
        help="Output variable name (hindcast style). If omitted, auto-detect from paths: "
             "2m_air_temperature_min / 2m_air_temperature / 2m_air_temperature_max",
    )
    p.add_argument("--subfolder", type=str, default=None, help="Process only this subfolder (e.g., 335).")
    p.add_argument("--min-steps-last-month", type=int, default=2,
                   help="Remove the last month if it has fewer than this number of timestamps.")
    a = p.parse_args()

    return CorrectionConfig(
        forecast_root=a.forecast_root.expanduser().resolve(),
        hindcast_root=a.hindcast_root.expanduser().resolve(),
        clim_file=a.clim_file.expanduser().resolve(),
        out_root=a.out_root.expanduser().resolve(),
        var_name=a.var_name,
        subfolder=a.subfolder,
        min_steps_last_month=int(a.min_steps_last_month),
    )


def main():
    cfg = parse_args()
    pipeline = ForecastCorrectionPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
