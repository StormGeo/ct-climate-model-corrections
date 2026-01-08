#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Forecast pipeline:
- Preprocess raw forecasts (remove incomplete last month, monthly accumulation, regrid to reference grid)
- Apply LS-Add correction using hindcast climatology and observed climatology
- Save corrected monthly forecasts with standardized output structure

OUTPUT RULE (same line as your hindcast scripts):
After --out-root, the script MUST ONLY create:
  <OUT_BASE>/<VAR>/<YEAR>/<DOY>/...

No extra folders.

Notes:
- The same NetCDF file (--clim-file) is used as:
  (1) observed monthly climatology (time=12)
  (2) reference grid (lat/lon) for regridding the raw forecasts
- Hindcast is assumed to already be on the final grid.
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
# HELPERS (same idea as your code 1)
# =========================
def extract_year_from_path(path: Path) -> int:
    for part in path.parts[::-1]:
        if len(part) == 4 and part.isdigit():
            y = int(part)
            if 1900 <= y <= 2100:
                return y
    raise ValueError(f"Could not find a 4-digit year in path: {path}")


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class CorrectionConfig:
    forecast_root: Path
    hindcast_root: Path
    clim_file: Path
    out_root: Path

    var_name: str = "total_precipitation"
    to_mm: bool = False
    subfolder: Optional[str] = None
    min_steps_last_month: int = 2


# =========================
# PIPELINE
# =========================
class ForecastCorrectionPipeline:
    def __init__(self, cfg: CorrectionConfig):
        self.cfg = cfg

        # Detect YEAR (prefer from forecast_root; fallback to hindcast_root)
        try:
            self.out_year = extract_year_from_path(self.cfg.forecast_root)
        except Exception:
            self.out_year = extract_year_from_path(self.cfg.hindcast_root)

        # FINAL RULE: ONLY <OUT_BASE>/<VAR>/<YEAR>/...
        self.out_root = (self.cfg.out_root / self.cfg.var_name / str(self.out_year)).expanduser().resolve()
        self.out_root.mkdir(parents=True, exist_ok=True)

        # Load observed climatology (time=12) and reference grid from the same file
        self.clim_obs, self.lat_ref, self.lon_ref = self._load_climatology_and_ref_grid(
            self.cfg.clim_file, self.cfg.var_name, self.cfg.to_mm
        )

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
        """
        Align longitude convention with reference grid when needed.
        Example: ds lon 0..360 and reference lon -180..180 -> convert ds to -180..180 and sort.
        """
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
        """
        Remove last month if it has fewer than min_steps_last_month timestamps.
        """
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
    # I/O loaders
    # -------------------------
    def _load_climatology_and_ref_grid(
        self, clim_file: Path, var_name: str, to_mm: bool
    ) -> Tuple[Dict[int, xr.DataArray], xr.DataArray, xr.DataArray]:
        """
        Read observed monthly climatology (time=12) and reference grid (lat/lon) from the same file.

        Returns:
          - clim_obs: dict[month(1..12)] -> DataArray(lat, lon)
          - lat_ref, lon_ref: reference grid coords
        """
        ds = xr.open_dataset(clim_file)
        ds = self._standardize_lat_lon(ds)

        if var_name not in ds:
            raise KeyError(f"Variable '{var_name}' not found in {clim_file}")

        da = ds[var_name]
        if "time" not in da.dims or da.sizes["time"] != 12:
            raise ValueError("Observed climatology must have time=12 (one per month).")

        if to_mm:
            da = da * 1000.0

        lat_ref = ds["lat"]
        lon_ref = ds["lon"]

        clim_obs: Dict[int, xr.DataArray] = {}
        for t in da.time:
            month = int(t.dt.month.values)
            clim_obs[month] = da.sel(time=t).drop_vars("time")

        return clim_obs, lat_ref, lon_ref

    def _load_hindcast_for_subfolder(self, julian_subdir: str) -> xr.Dataset:
        """
        Open hindcast dataset from hindcast_root/<julian_subdir>/*.nc.
        Hindcast is expected to have:
          - var_name
          - dim 'lead'
          - coord 'month' linked to 'lead'
        """
        subdir = self.cfg.hindcast_root / julian_subdir
        files = sorted(subdir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"No hindcast files found in: {subdir}")

        ds = xr.open_dataset(files[0])
        ds = self._standardize_lat_lon(ds)

        if self.cfg.var_name not in ds:
            raise KeyError(f"Variable '{self.cfg.var_name}' not found in: {files[0]}")

        if "lead" not in ds.dims:
            raise ValueError(f"Hindcast must have a 'lead' dimension. Dims: {ds.dims}")

        if "month" not in ds.coords:
            raise ValueError("Hindcast must have a 'month' coordinate associated with 'lead'.")

        if self.cfg.to_mm:
            ds[self.cfg.var_name] = ds[self.cfg.var_name] * 1000.0

        return ds

    # -------------------------
    # Forecast preprocessing
    # -------------------------
    def _preprocess_forecast(self, forecast_path: Path) -> xr.DataArray:
        """
        Open raw forecast, remove incomplete last month, aggregate to monthly totals (MS),
        and regrid to the reference grid.

        Returns:
          DataArray (time, lat, lon)
        """
        ds = xr.open_dataset(forecast_path)
        ds = self._standardize_lat_lon(ds)

        if self.cfg.var_name not in ds:
            raise KeyError(f"Variable '{self.cfg.var_name}' not found in: {forecast_path}")

        ds = self._remove_incomplete_last_month(ds, self.cfg.min_steps_last_month)
        if ds["time"].size == 0:
            raise ValueError("No timestamps left after removing incomplete last month.")

        # Monthly accumulation
        ds_m = ds.resample(time="MS").sum()

        # Adjust longitude convention and regrid
        ds_m = self._adjust_lon_to_reference(ds_m, self.lon_ref)
        ds_m = ds_m.interp(lat=self.lat_ref, lon=self.lon_ref)

        da = ds_m[self.cfg.var_name]
        if self.cfg.to_mm:
            da = da * 1000.0

        # Keep original variable attributes where possible
        da.attrs.update(ds[self.cfg.var_name].attrs if self.cfg.var_name in ds else {})
        return da

    # -------------------------
    # Correction
    # -------------------------
    def _ls_add_correction(self, fore: xr.DataArray, ds_h: xr.Dataset) -> xr.Dataset:
        """
        Apply LS-Add correction:
          corrected = forecast + (clim_obs(month_of_lead) - clim_hindcast(lead))
        """
        hind = ds_h[self.cfg.var_name]       # (lead, lat, lon)
        hind_months = ds_h["month"]          # (lead,)

        # Grid check (expect exact match after preprocessing)
        if not np.array_equal(fore.lat.values, hind.lat.values):
            raise ValueError("Hindcast latitude != preprocessed forecast latitude.")
        if not np.array_equal(fore.lon.values, hind.lon.values):
            raise ValueError("Hindcast longitude != preprocessed forecast longitude.")

        n_leads = hind.sizes["lead"]
        n_corr = min(n_leads, fore.sizes["time"])

        corr = fore.copy()
        for idx in range(n_corr):
            month = int(hind_months.isel(lead=idx).values)
            if month not in self.clim_obs:
                raise KeyError(f"Month {month} not found in observed climatology (1..12).")

            clim_obs_m = self.clim_obs[month]
            clim_hind_m = hind.isel(lead=idx)
            bias = clim_obs_m - clim_hind_m

            corr[idx, :, :] = fore[idx, :, :] + bias

        corr = corr.clip(min=0)
        corr.name = self.cfg.var_name
        return corr.to_dataset(name=self.cfg.var_name)

    # -------------------------
    # File listing & saving
    # -------------------------
    def _list_forecast_files(self) -> list[Path]:
        if self.cfg.subfolder:
            target_dir = self.cfg.forecast_root / self.cfg.subfolder
            files = sorted(target_dir.glob("*.nc"))
        else:
            files = sorted(self.cfg.forecast_root.rglob("*.nc"))
        return files

    def _output_path_for_forecast(self, forecast_path: Path, julian_subdir: str) -> Path:
        """
        Enforce output structure:
          <OUT_BASE>/<VAR>/<YEAR>/<DOY>/<original_filename>.nc
        """
        out_dir = self.out_root / julian_subdir
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
        print("Outputs will be under:")
        print(f"  {self.out_root}")

        for fpath in forecast_files:
            print(f"\n=== Processing file: {fpath} ===")
            julian_subdir = fpath.parent.name  # expects DOY folder name like 001/032/335...

            # 1) Preprocess forecast (monthly + regrid)
            fore_m = self._preprocess_forecast(fpath)

            # 2) Load matching hindcast
            ds_h = self._load_hindcast_for_subfolder(julian_subdir)

            # 3) Correct
            ds_out = self._ls_add_correction(fore_m, ds_h)

            # 4) Save (enforced structure)
            out_path = self._output_path_for_forecast(fpath, julian_subdir)
            ds_out.to_netcdf(out_path)
            print(f"[OK] Saved corrected forecast: {out_path}")

        print("\nDone.")


# =========================
# CLI
# =========================
def parse_args() -> CorrectionConfig:
    p = argparse.ArgumentParser(
        description="Preprocess raw forecasts (monthly + regrid) and apply LS-Add correction using hindcast and observed climatology."
    )
    p.add_argument("--forecast-root", type=Path, required=True, help="Root directory containing raw forecast NetCDF files.")
    p.add_argument("--hindcast-root", type=Path, required=True, help="Root directory containing processed hindcast subfolders (e.g., 335/).")
    p.add_argument("--clim-file", type=Path, required=True, help="NetCDF file containing observed climatology (time=12) and reference grid.")
    p.add_argument("--out-root", type=Path, required=True, help="Output base directory for corrected forecasts.")
    p.add_argument("--var-name", type=str, default="total_precipitation", help="Variable name inside NetCDF files.")
    p.add_argument("--to-mm", action="store_true", help="Convert from meters to millimeters (multiply by 1000).")
    p.add_argument("--subfolder", type=str, default=None, help="Process only this subfolder (e.g., 335).")
    p.add_argument("--min-steps-last-month", type=int, default=2, help="Remove the last month if it has fewer than this number of timestamps.")
    a = p.parse_args()

    return CorrectionConfig(
        forecast_root=a.forecast_root.expanduser().resolve(),
        hindcast_root=a.hindcast_root.expanduser().resolve(),
        clim_file=a.clim_file.expanduser().resolve(),
        out_root=a.out_root.expanduser().resolve(),
        var_name=a.var_name,
        to_mm=bool(a.to_mm),
        subfolder=a.subfolder,
        min_steps_last_month=int(a.min_steps_last_month),
    )


def main():
    cfg = parse_args()
    pipeline = ForecastCorrectionPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
