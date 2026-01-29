#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single script (download + processing) for ECMWF System 51 hindcast monthly fields (TEMPERATURE ONLY).

IMPORTANT OUTPUT RULE (final):
After --out-grib and --out-nc, the script MUST ONLY create:
  <OUT_BASE>/<VAR>/<YEAR>/<DOY>/...

No extra folders (no "temperatura", no "ECMWF_2025", no group/model folders).

- Variable is detected from --doy-root path (must contain one of: 2m_air_temperature_min/med/max).
- Year is detected from --doy-root path (a 4-digit folder).
- DOY subfolders under --doy-root identify which months to run.
- NetCDF filename pattern:
    ecmwf_subseas_glo_<var>_AVG_<YYYYMMDDHH>.nc
"""

import argparse
import calendar
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, List

import cdsapi
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# CONSTANTS (operational defaults)
# =========================
DEFAULT_DATASET = "seasonal-monthly-single-levels"
DEFAULT_ORIGINATING_CENTRE = "ecmwf"
DEFAULT_SYSTEM = "51"

# Temperature-only defaults (keep logic; remove precip defaults)
DEFAULT_VARIABLE = ["2m_temperature"]
DEFAULT_PRODUCT_TYPE = ["monthly_mean"]
DEFAULT_DAY = ["01"]
DEFAULT_TIME = ["00:00"]
DEFAULT_LEADTIME_MONTH = ["1", "2", "3", "4", "5", "6"]
DEFAULT_DATA_FORMAT = "grib"
DEFAULT_AREA = [90, -180, -90, 180]
DEFAULT_YEARS = [str(y) for y in range(1993, 2017)]

DEFAULT_REGRID_METHOD = "bilinear"
DEFAULT_REGRID_PERIODIC = True
DEFAULT_REUSE_WEIGHTS = True

OUT_PREFIX = "ecmwf_subseas_glo"
OUT_STAT_TOKEN = "AVG"


# =========================
# CONFIG
# =========================
@dataclass(frozen=True)
class HindcastConfig:
    dataset: str = DEFAULT_DATASET
    originating_centre: str = DEFAULT_ORIGINATING_CENTRE
    system: str = DEFAULT_SYSTEM

    variable: list[str] = None
    product_type: list[str] = None
    day: list[str] = None
    time: list[str] = None
    leadtime_month: list[str] = None
    data_format: str = DEFAULT_DATA_FORMAT
    area: list[float] = None
    years: list[str] = None

    regrid_method: str = DEFAULT_REGRID_METHOD
    regrid_periodic: bool = DEFAULT_REGRID_PERIODIC
    reuse_weights: bool = DEFAULT_REUSE_WEIGHTS

    def __post_init__(self):
        object.__setattr__(self, "variable", DEFAULT_VARIABLE if self.variable is None else self.variable)
        object.__setattr__(self, "product_type", DEFAULT_PRODUCT_TYPE if self.product_type is None else self.product_type)
        object.__setattr__(self, "day", DEFAULT_DAY if self.day is None else self.day)
        object.__setattr__(self, "time", DEFAULT_TIME if self.time is None else self.time)
        object.__setattr__(self, "leadtime_month", DEFAULT_LEADTIME_MONTH if self.leadtime_month is None else self.leadtime_month)
        object.__setattr__(self, "area", DEFAULT_AREA if self.area is None else self.area)
        object.__setattr__(self, "years", DEFAULT_YEARS if self.years is None else self.years)


# =========================
# PIPELINE CLASS
# =========================
class HindcastPipeline:
    # Temperature-only detection hints
    VAR_HINTS = {
        "t2m_min": ["2m_air_temperature_min", "t2m_min", "2m_temperature_min", "temperature_min", "temperatura_min", "tmin"],
        "t2m_med": ["2m_air_temperature", "t2m_med", "2m_temperature_mean", "temperature_mean", "temperatura_media", "tmean", "tmed"],
        "t2m_max": ["2m_air_temperature_max", "t2m_max", "2m_temperature_max", "temperature_max", "temperatura_max", "tmax"],
    }

    CDS_VARIABLE = {
        "t2m_min": "2m_temperature",
        "t2m_med": "2m_temperature",
        "t2m_max": "2m_temperature",
    }

    CDS_PRODUCT_TYPE = {
        "t2m_min": ["monthly_minimum"],
        "t2m_med": ["monthly_mean"],
        "t2m_max": ["monthly_maximum"],
    }

    def __init__(self, config: HindcastConfig, out_grib_root: Path, out_nc_root: Path, enable_regrid: bool, ref_grid: Optional[Path]):
        self.cfg = config
        self.out_grib_root = out_grib_root.expanduser().resolve()
        self.out_nc_root = out_nc_root.expanduser().resolve()
        self.enable_regrid = enable_regrid
        self.ref_grid = ref_grid.expanduser().resolve() if ref_grid else None

        self.out_grib_root.mkdir(parents=True, exist_ok=True)
        self.out_nc_root.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # DOY -> month helpers
    # -------------------------
    @staticmethod
    def month_from_doy(doy: int, year_dummy: int = 2001) -> int:
        if doy < 1 or doy > 366:
            raise ValueError(f"DOY out of range: {doy}")
        cum = 0
        for m in range(1, 13):
            cum += calendar.monthrange(year_dummy, m)[1]
            if doy <= cum:
                return m
        raise ValueError(f"Unable to map DOY to month: {doy}")

    @staticmethod
    def list_doy_subfolders(doy_root: Path) -> List[int]:
        if not doy_root.is_dir():
            raise NotADirectoryError(f"DOY root is not a directory: {doy_root}")
        doys: List[int] = []
        for p in doy_root.iterdir():
            if p.is_dir() and len(p.name) == 3 and p.name.isdigit():
                v = int(p.name)
                if 1 <= v <= 366:
                    doys.append(v)
        return sorted(doys)

    @classmethod
    def detect_var_from_path_only(cls, path: Path) -> Set[str]:
        s = path.as_posix().lower()
        hits: Set[str] = set()
        for key, hints in cls.VAR_HINTS.items():
            if any(h.lower() in s for h in hints):
                hits.add(key)
        return hits

    @staticmethod
    def extract_year_from_path(path: Path) -> int:
        for part in path.parts[::-1]:
            if len(part) == 4 and part.isdigit():
                y = int(part)
                if 1900 <= y <= 2100:
                    return y
        raise ValueError(f"Could not find a 4-digit year in doy-root path: {path}")

    # -------------------------
    # Path / naming helpers
    # -------------------------
    @staticmethod
    def julian_day_for_month(month: int, year_dummy: int = 2001) -> int:
        return sum(calendar.monthrange(year_dummy, m)[1] for m in range(1, month)) + 1

    def out_folder_for_month(self, month_str: str) -> str:
        doy = self.julian_day_for_month(int(month_str), year_dummy=2001)
        return f"{doy:03d}"

    @staticmethod
    def build_grib_outfile_t2m(kind: str, month_str: str, folder_path: Path) -> Path:
        return folder_path / f"ecmwf_sys51_{kind}_hindcast_monthly_init{month_str}_1993-2016_lead1-6_global.grib"

    @staticmethod
    def output_variable_name(var_key: str) -> str:
        if var_key == "t2m_min":
            return "2m_air_temperature_min"
        if var_key == "t2m_med":
            return "2m_air_temperature"
        if var_key == "t2m_max":
            return "2m_air_temperature_max"
        raise ValueError(f"Unknown var_key: {var_key}")

    @staticmethod
    def build_operational_nc_filename(var_name: str, init_stamp: str) -> str:
        return f"ecmwf_subseas_glo_{var_name}_AVG_{init_stamp}.nc"

    def build_nc_outfile_operational(self, nc_dir: Path, var_key: str, year: int, month_str: str) -> Path:
        init_stamp = f"{year:04d}{int(month_str):02d}0100"
        var_name = self.output_variable_name(var_key)
        return nc_dir / self.build_operational_nc_filename(var_name, init_stamp)

    # -------------------------
    # Time / coord helpers
    # -------------------------
    @staticmethod
    def seconds_in_month_from_ym(ym: np.datetime64) -> float:
        start = ym.astype("datetime64[ns]")
        end = (ym + np.timedelta64(1, "M")).astype("datetime64[ns]")
        return float((end - start) / np.timedelta64(1, "s"))

    @staticmethod
    def normalize_coords_latlon(ds: xr.Dataset) -> xr.Dataset:
        if "lat" in ds.coords and "latitude" not in ds.coords:
            ds = ds.rename({"lat": "latitude"})
        if "lon" in ds.coords and "longitude" not in ds.coords:
            ds = ds.rename({"lon": "longitude"})
        if "latitude" not in ds.coords or "longitude" not in ds.coords:
            raise RuntimeError(f"Dataset is missing latitude/longitude coordinates. Coords: {list(ds.coords)}")
        return ds

    @staticmethod
    def load_reference_grid(path: Path) -> xr.Dataset:
        ds_ref = xr.open_dataset(path)
        if "lat" in ds_ref.coords and "latitude" not in ds_ref.coords:
            ds_ref = ds_ref.rename({"lat": "latitude"})
        if "lon" in ds_ref.coords and "longitude" not in ds_ref.coords:
            ds_ref = ds_ref.rename({"lon": "longitude"})
        if "latitude" not in ds_ref.coords or "longitude" not in ds_ref.coords:
            raise ValueError("Reference grid file must have latitude/longitude coordinates (or lat/lon).")
        return xr.Dataset({"lat": ds_ref["latitude"], "lon": ds_ref["longitude"]})

    # -------------------------
    # Download (unchanged logic; temp-only usage)
    # -------------------------
    def download_grib(self, month_str: str, out_file: Path, variables_override: Optional[List[str]] = None, product_type_override: Optional[List[str]] = None) -> None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        request = {
            "originating_centre": self.cfg.originating_centre,
            "system": self.cfg.system,
            "variable": self.cfg.variable if variables_override is None else variables_override,
            "product_type": self.cfg.product_type if product_type_override is None else product_type_override,
            "year": self.cfg.years,
            "month": [month_str],
            "day": self.cfg.day,
            "time": self.cfg.time,
            "leadtime_month": self.cfg.leadtime_month,
            "data_format": self.cfg.data_format,
            "area": self.cfg.area,
        }
        print(f"\n=== DOWNLOAD month={month_str} ===")
        print(f"Variables: {request['variable']}")
        print(f"Product type: {request['product_type']}")
        print(f"Output: {out_file}")
        c = cdsapi.Client()
        c.retrieve(self.cfg.dataset, request, str(out_file))
        print("[OK] Download completed")

    # -------------------------
    # Processing (TEMP)
    # -------------------------
    def process_t2m_hindcast_grib(self, in_grib: Path, kind: str) -> xr.Dataset:
        ds = xr.open_dataset(in_grib, engine="cfgrib")
        ds = self.normalize_coords_latlon(ds)

        if "valid_time" not in ds:
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

        tname = None
        for cand in ("t2m", "2t"):
            if cand in ds.data_vars:
                tname = cand
                break
        if tname is None:
            if len(ds.data_vars) == 1:
                tname = list(ds.data_vars)[0]
            else:
                raise RuntimeError(f"Expected temperature variable in GRIB. Found: {list(ds.data_vars)}")

        t_k = ds[tname]
        vt = ds["valid_time"]

        vt_np = vt.values.astype("datetime64[ns]")
        target_np = vt_np - np.timedelta64(1, "D")
        target_ym = target_np.astype("datetime64[M]")

        init_ym = ds["init_time"].values.astype("datetime64[M]")
        lead = (target_ym.astype("int64") - init_ym.astype("int64")[:, None]) + 1
        lead_da = xr.DataArray(lead, dims=vt.dims, coords=vt.coords, name="lead")
        mask = (lead_da >= 1) & (lead_da <= 6)

        t_k = t_k.where(mask)
        t_k = t_k.assign_coords(lead=lead_da)

        t_c = t_k - 273.15

        suffix_map = {"t2m_min": "min", "t2m_med": "med", "t2m_max": "max"}
        out_name = f"2m_air_temperature_{suffix_map[kind]}"

        t_c.name = out_name
        t_c.attrs["units"] = "degC"

        lead_fields = []
        for k in range(1, 7):
            sel = t_c.where(t_c["lead"] == k, drop=True)
            if "step_raw" in sel.dims:
                sel = sel.mean("step_raw", skipna=True)
            sel = sel.reset_coords(drop=True)
            sel = sel.expand_dims({"lead": [k]})
            lead_fields.append(sel)

        t_lead = xr.concat(lead_fields, dim="lead", coords="minimal", compat="override")
        t_clim = t_lead.mean(dim=["member", "init_time"], skipna=True)

        out = xr.Dataset({out_name: t_clim})
        out = out.assign_coords(month=("lead", np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)))
        out[out_name] = out[out_name].transpose("lead", "latitude", "longitude")
        return out

    # -------------------------detect_var_from_path_only
    # Optional regrid
    # -------------------------
    def regrid_dataset(self, out: xr.Dataset, ds_grid: xr.Dataset, weights_file: Optional[Path]) -> xr.Dataset:
        import xesmf as xe

        src_grid = xr.Dataset({"lat": out["latitude"], "lon": out["longitude"]})

        kwargs = dict(method=self.cfg.regrid_method, periodic=self.cfg.regrid_periodic, reuse_weights=False)

        if weights_file is not None:
            weights_file.parent.mkdir(parents=True, exist_ok=True)
            if weights_file.exists():
                kwargs["reuse_weights"] = True
            kwargs["filename"] = str(weights_file)

        regridder = xe.Regridder(src_grid, ds_grid, **kwargs)
        out_regrid = regridder(out)

        if "lat" in out_regrid.coords and "latitude" not in out_regrid.coords:
            out_regrid = out_regrid.rename({"lat": "latitude"})
        if "lon" in out_regrid.coords and "longitude" not in out_regrid.coords:
            out_regrid = out_regrid.rename({"lon": "longitude"})

        for vn in out_regrid.data_vars:
            out_regrid[vn] = out_regrid[vn].transpose("lead", "latitude", "longitude")
        return out_regrid

    # -------------------------detect_var_from_path_only
    # Run (temperature-only)
    # -------------------------
    def run_var(self, month_str: str, var_key: str, out_year: int) -> Path:
        valid_months = {f"{m:02d}" for m in range(1, 13)}
        month_str = month_str.strip()
        if month_str not in valid_months:
            raise ValueError("--month must be in the range 01..12")
        if var_key not in self.CDS_VARIABLE:
            raise ValueError(f"Unknown var_key: {var_key}")

        subdir = self.out_folder_for_month(month_str)
        grib_dir = self.out_grib_root / subdir
        nc_dir = self.out_nc_root / subdir
        grib_dir.mkdir(parents=True, exist_ok=True)
        nc_dir.mkdir(parents=True, exist_ok=True)

        cds_var = self.CDS_VARIABLE[var_key]
        cds_ptype = self.CDS_PRODUCT_TYPE[var_key]

        grib_file = self.build_grib_outfile_t2m(var_key, month_str, grib_dir)
        self.download_grib(month_str, grib_file, [cds_var], cds_ptype)

        print("\n=== PROCESSING ===")
        print(f"Input GRIB: {grib_file}")
        out = self.process_t2m_hindcast_grib(grib_file, kind=var_key)

        nc_file = self.build_nc_outfile_operational(nc_dir, var_key, out_year, month_str)

        if self.enable_regrid:
            if self.ref_grid is None:
                raise ValueError("With --regrid you must provide --ref-grid /path/to/grid.nc")
            if not self.ref_grid.exists():
                raise FileNotFoundError(f"Reference grid file does not exist: {self.ref_grid}")

            ds_grid = self.load_reference_grid(self.ref_grid)

            weights_file = None
            if self.cfg.reuse_weights:
                weights_dir = self.out_nc_root / "_weights"
                weights_file = weights_dir / f"weights_{self.cfg.regrid_method}_periodic{int(self.cfg.regrid_periodic)}.nc"

            print(f"[INFO] Regridding enabled: method={self.cfg.regrid_method} periodic={self.cfg.regrid_periodic}")
            out = self.regrid_dataset(out, ds_grid, weights_file)

        print("\n=== SAVING ===")
        print(f"Output NC: {nc_file}")
        out.to_netcdf(nc_file)
        print("[OK] Done")
        return nc_file


# =========================
# CLI
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Download + process ECMWF System 51 hindcast monthly TEMPERATURE fields.")
    p.add_argument("--doy-root", required=True, help="Directory containing DOY subfolders like 001, 015, 215")
    p.add_argument("--out-grib", required=True, help="Base directory for GRIB outputs")
    p.add_argument("--out-nc", required=True, help="Base directory for NetCDF outputs")
    p.add_argument("--regrid", action="store_true", help="Enable regridding using xesmf")
    p.add_argument("--ref-grid", default=None, help="NetCDF reference grid file (lat/lon or latitude/longitude)")
    return p.parse_args()


def main():
    args = parse_args()

    ref_grid = Path(args.ref_grid) if args.ref_grid else None
    doy_root = Path(args.doy_root).expanduser().resolve()

    cfg_tmp = HindcastConfig()
    tmp = HindcastPipeline(
        config=cfg_tmp,
        out_grib_root=Path(args.out_grib),
        out_nc_root=Path(args.out_nc),
        enable_regrid=bool(args.regrid),
        ref_grid=ref_grid,
    )

    hits = tmp.detect_var_from_path_only(doy_root)
    if not hits:
        raise RuntimeError(
            f"Could not detect temperature variable from doy-root path: {doy_root}. "
            f"Expected path to include one of: 2m_air_temperature_min/med/max."
        )
    if len(hits) != 1:
        raise RuntimeError(f"Ambiguous doy-root path; detected {sorted(hits)} from: {doy_root}")

    var_key = next(iter(hits))
    out_year = tmp.extract_year_from_path(doy_root)
    var_folder = tmp.output_variable_name(var_key)

    # FINAL RULE: ONLY <OUT_BASE>/<VAR>/<YEAR>/...
    out_grib_root = Path(args.out_grib) / var_folder / str(out_year)
    out_nc_root = Path(args.out_nc) / var_folder / str(out_year)

    cfg = HindcastConfig()
    pipeline = HindcastPipeline(
        config=cfg,
        out_grib_root=out_grib_root,
        out_nc_root=out_nc_root,
        enable_regrid=bool(args.regrid),
        ref_grid=ref_grid,
    )

    print(f"Detected variable from path: {var_key}")
    print(f"Detected year from path: {out_year}")
    print(f"Outputs will be under:")
    print(f"  GRIB: {out_grib_root}")
    print(f"  NC:   {out_nc_root}")

    doys = pipeline.list_doy_subfolders(doy_root)
    if not doys:
        raise RuntimeError(f"No DOY subfolders found under: {doy_root}")

    months_needed: Set[int] = set()
    for doy in doys:
        m = pipeline.month_from_doy(doy, year_dummy=2001)
        months_needed.add(m)

    months_sorted = sorted(months_needed)
    print(f"Detected DOYs: {len(doys)}. Months to download/process: {months_sorted}")

    for m in months_sorted:
        pipeline.run_var(f"{m:02d}", var_key, out_year)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
