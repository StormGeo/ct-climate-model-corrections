#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
import warnings
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import xesmf as xe

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class PipelineConfig:
    default_var: str = "total_precipitation"
    hindcast_leads_expected: int = 9
    hindcast_time_mode: str = "00z"          # "00z" | "first" | "mean"
    adjust_lon_360_to_180: bool = True

    # NetCDF writing
    netcdf_engine: str = "netcdf4"
    zlib: bool = False

    # Cache month weights (month_sum + idx_map) on disk for reuse
    cache_month_weights_to_disk: bool = True

    # Option 2: allow extra months beyond hindcast (repeat last monthly total, but real-month weights)
    extend_months: int = 1  # ALWAYS 1 now (10 months window possible)

    # NEW: hard limit output to N daily steps (always)
    max_days_output: int = 280


# =============================================================================
# Helpers
# =============================================================================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_coords_latlon(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ds = ds.rename({"lon": "longitude"})
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise RuntimeError(f"Missing latitude/longitude coords. Coords={list(ds.coords)}")
    return ds


def sort_latlon(ds: xr.Dataset) -> xr.Dataset:
    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")
    if "longitude" in ds.coords:
        ds = ds.sortby("longitude")
    return ds


def extract_doy_from_path(path: Path) -> str:
    for part in path.parts[::-1]:
        if len(part) == 3 and part.isdigit():
            d = int(part)
            if 1 <= d <= 366:
                return f"{d:03d}"
    raise ValueError(f"Não consegui achar DOY (001..366) no caminho: {path}")


def extract_year_from_path_optional(path: Path) -> Optional[int]:
    for part in path.parts[::-1]:
        if len(part) == 4 and part.isdigit():
            y = int(part)
            if 1900 <= y <= 2100:
                return y
    return None


def extract_lead_from_filename(name: str) -> Optional[int]:
    m = re.search(r"[Mm](\d{3})", name)
    if m:
        return int(m.group(1))
    m = re.search(r"lead(\d+)", name, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def init_stamp_from_filename_strict(path: Path) -> str:
    """
    Extrai YYYYMMDDHH diretamente do nome do arquivo de forecast.
    Exemplo:
      cfs_glo_total_precipitation_M000_2025010106.nc -> 2025010106
    """
    m = re.search(r"(19|20)\d{8}", path.name)
    if not m:
        raise ValueError(f"Não achei init_stamp YYYYMMDDHH no nome do arquivo: {path.name}")
    return m.group(0)


def month_from_doy_and_lead(doy_str: str, lead: int) -> int:
    start_doy = int(doy_str)
    start_date = np.datetime64("2001-01-01") + np.timedelta64(start_doy - 1, "D")
    start_month = int(str(start_date)[5:7])
    return int(((start_month - 1 + (lead - 1)) % 12) + 1)


def list_doys_in_month_2001(month: int) -> List[int]:
    start = np.datetime64(f"2001-{month:02d}-01")
    end = np.datetime64("2002-01-01") if month == 12 else np.datetime64(f"2001-{month+1:02d}-01")
    ndays = int((end - start).astype("timedelta64[D]").astype(np.int64))
    doy0 = int(((start.astype("datetime64[D]") - np.datetime64("2001-01-01")) / np.timedelta64(1, "D")) + 1)
    return [doy0 + i for i in range(ndays)]


def doy_of_2001_from_yyyy_mm_dd(tday: np.datetime64) -> int:
    s = str(tday.astype("datetime64[D]"))  # YYYY-MM-DD
    mm = int(s[5:7]); dd = int(s[8:10])
    d2001 = np.datetime64(f"2001-{mm:02d}-{dd:02d}").astype("datetime64[D]")
    return int(((d2001 - np.datetime64("2001-01-01")) / np.timedelta64(1, "D")) + 1)


def choose_var_name(ds: xr.Dataset, preferred: str, user_var: Optional[str]) -> str:
    if user_var is not None:
        if user_var not in ds.data_vars:
            raise KeyError(f"Variável --var='{user_var}' não encontrada. Vars: {list(ds.data_vars)}")
        return user_var
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return list(ds.data_vars)[0]
    raise KeyError(
        f"Variável '{preferred}' não encontrada e dataset tem múltiplas vars: {list(ds.data_vars)}. "
        f"Use --var para especificar."
    )


def load_ref_latlon_from_refgrid(ref_grid_file: Path) -> Tuple[xr.DataArray, xr.DataArray]:
    ds_ref = xr.open_dataset(ref_grid_file)
    ds_ref = normalize_coords_latlon(ds_ref)
    ds_ref = sort_latlon(ds_ref)
    return ds_ref["latitude"], ds_ref["longitude"]


def build_forecast_outname(var_name: str, init_stamp: str) -> str:
    return f"cfs_glo_{var_name}_M100_{init_stamp}.nc"


# =============================================================================
# Pipeline
# =============================================================================

class CFSPipelineAuto:
    def _find_hindcast_dir_for_doy(self) -> Path:
        var_root = self.hindcast_root / self.var_name
        if var_root.is_dir():
            year_dirs = [d for d in var_root.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 4]
            year_dirs = sorted(year_dirs, key=lambda p: int(p.name), reverse=True)
            for ydir in year_dirs:
                cand = ydir / self.doy
                if cand.is_dir():
                    return cand

            for child in sorted(var_root.iterdir()):
                if child.is_dir():
                    cand = child / self.doy
                    if cand.is_dir():
                        return cand

            candidates = [p for p in var_root.rglob(self.doy) if p.is_dir()]
            if candidates:
                return sorted(candidates)[0]

        cand_root = self.hindcast_root / self.doy
        if cand_root.is_dir():
            return cand_root

        candidates = [p for p in self.hindcast_root.rglob(self.doy) if p.is_dir()]
        if candidates:
            return sorted(candidates)[0]

        raise FileNotFoundError(f"Pasta de hindcast não encontrada para DOY {self.doy} em {self.hindcast_root}")

    def __init__(
        self,
        cfg: PipelineConfig,
        forecast_input: Path,
        hindcast_root: Path,
        clim_obs_file: Path,
        ref_grid_file: Path,
        out_hindcast_base: Path,
        out_corr_base: Path,
        year_override: Optional[int],
        var_override: Optional[str],
        debug: bool = False,
    ):
        self.cfg = cfg
        self.forecast_input = forecast_input.expanduser().resolve()
        self.hindcast_root = hindcast_root.expanduser().resolve()
        self.clim_obs_file = clim_obs_file.expanduser().resolve()
        self.ref_grid_file = ref_grid_file.expanduser().resolve()
        self.out_hindcast_base = out_hindcast_base.expanduser().resolve()
        self.out_corr_base = out_corr_base.expanduser().resolve()
        self.debug = bool(debug)
        self.var_override = var_override

        self.doy = extract_doy_from_path(self.forecast_input)

        inferred_year = extract_year_from_path_optional(self.forecast_input)
        self.year = int(year_override) if year_override is not None else (inferred_year if inferred_year is not None else 0)

        if self.forecast_input.is_dir():
            self.forecast_dir = self.forecast_input
            files = sorted(self.forecast_dir.glob("*_M000_*00.nc"))
            if not files:
                raise FileNotFoundError(
                    f"Nenhum forecast M000 00Z encontrado em: {self.forecast_dir} "
                    f"(padrão esperado: *_M000_YYYYMMDD00.nc)"
                )
            self.forecast_files = [files[-1]]
        else:
            self.forecast_dir = self.forecast_input.parent
            self.forecast_files = [self.forecast_input]

        if not self.forecast_files:
            raise FileNotFoundError(f"Nenhum forecast .nc encontrado em: {self.forecast_dir}")

        if not self.ref_grid_file.exists():
            raise FileNotFoundError(f"Ref-grid não existe: {self.ref_grid_file}")

        t0 = time.time()
        self._clim_lookup, self._max_doy, self.var_name = self._load_obs_climatology_daily_fast()
        self._ny = self._clim_lookup.shape[1]
        self._nx = self._clim_lookup.shape[2]
        if self.debug:
            print("[T] build clim_lookup (FAST one-read):", round(time.time() - t0, 2), "s")

        self.lat_ref, self.lon_ref = load_ref_latlon_from_refgrid(self.ref_grid_file)

        self.hindcast_dir = self._find_hindcast_dir_for_doy()
        if not self.hindcast_dir.is_dir():
            raise NotADirectoryError(f"Pasta de hindcast não encontrada: {self.hindcast_dir}")

        self.out_hindcast_root = self.out_hindcast_base / self.var_name / f"{self.year:04d}" / self.doy
        self.out_corr_root = self.out_corr_base / self.var_name / f"{self.year:04d}" / self.doy
        ensure_dir(self.out_hindcast_root)
        ensure_dir(self.out_corr_root)

        self._regridder = None

        self._month_cache_sum: Dict[int, np.ndarray] = {}
        self._month_cache_idx: Dict[int, np.ndarray] = {}

        if self.debug:
            print("=== AUTO DETECT ===")
            print("forecast_input :", self.forecast_input)
            print("year(out)      :", f"{self.year:04d}")
            print("doy            :", self.doy)
            print("hindcast_dir   :", self.hindcast_dir)
            print("forecast_dir   :", self.forecast_dir)
            print("forecast_files :", len(self.forecast_files))
            print("var_name       :", self.var_name)
            print("out_hindcast   :", self.out_hindcast_root)
            print("out_corr       :", self.out_corr_root)
            print("extend_months  :", int(self.cfg.extend_months))
            print("max_days_output:", int(self.cfg.max_days_output))

    # -------------------------------------------------------------------------
    # Obs daily climatology FAST (one read)
    # -------------------------------------------------------------------------
    def _load_obs_climatology_daily_fast(self) -> Tuple[np.ndarray, int, str]:
        if not self.clim_obs_file.exists():
            raise FileNotFoundError(f"Climatologia observada não existe: {self.clim_obs_file}")

        ds = xr.open_dataset(self.clim_obs_file)
        ds = normalize_coords_latlon(ds)
        ds = sort_latlon(ds)

        v = choose_var_name(ds, self.cfg.default_var, self.var_override)
        da = ds[v]

        if "time" not in da.dims:
            raise ValueError("Climatologia observada diária precisa ter dimensão 'time'.")

        da = da.transpose("time", "latitude", "longitude")

        n = int(da.sizes["time"])
        if n < 365:
            raise ValueError(f"Climatologia diária esperada >=365 passos, veio time={n}.")

        da = da.astype("float32").load()

        doys = da["time"].dt.dayofyear.values.astype(np.int32)
        if doys.size != da.sizes["time"]:
            raise ValueError("Falha ao obter dayofyear da climatologia.")

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
            raise ValueError(f"Climatologia diária insuficiente: preencheu {nfilled} DOYs (esperado >=365).")

        return clim_lookup, max_doy, v

    # -------------------------------------------------------------------------
    # Month cache (on-demand + optional disk persistence)
    # -------------------------------------------------------------------------
    def _month_cache_paths(self, month: int) -> Tuple[Path, Path]:
        cache_dir = self.out_hindcast_root / "month_cache"
        ensure_dir(cache_dir)
        f_sum = cache_dir / f"month{month:02d}_sum.npy"
        f_idx = cache_dir / f"month{month:02d}_idx.npy"
        return f_sum, f_idx

    def _get_month_sum_and_idxmap(self, month: int) -> Tuple[np.ndarray, np.ndarray]:
        if month in self._month_cache_sum:
            return self._month_cache_sum[month], self._month_cache_idx[month]

        if self.cfg.cache_month_weights_to_disk:
            f_sum, f_idx = self._month_cache_paths(month)
            if f_sum.exists() and f_idx.exists():
                month_sum = np.load(f_sum).astype(np.float32, copy=False)
                idx_map = np.load(f_idx).astype(np.int32, copy=False)
                self._month_cache_sum[month] = month_sum
                self._month_cache_idx[month] = idx_map
                return month_sum, idx_map

        doys_m = np.array(list_doys_in_month_2001(month), dtype=np.int32)
        doys_m = doys_m[doys_m <= self._max_doy]
        if doys_m.size < 1:
            month_sum = np.zeros((self._ny, self._nx), dtype=np.float32)
            idx_map = np.full((self._max_doy + 1,), -1, dtype=np.int32)
        else:
            clim_days = self._clim_lookup[doys_m, :, :]
            month_sum = np.sum(clim_days, axis=0).astype(np.float32)

            idx_map = np.full((self._max_doy + 1,), -1, dtype=np.int32)
            for i, d in enumerate(doys_m):
                idx_map[int(d)] = i

        self._month_cache_sum[month] = month_sum
        self._month_cache_idx[month] = idx_map

        if self.cfg.cache_month_weights_to_disk:
            f_sum, f_idx = self._month_cache_paths(month)
            try:
                np.save(f_sum, month_sum)
                np.save(f_idx, idx_map)
            except Exception:
                pass

        return month_sum, idx_map

    # -------------------------------------------------------------------------
    # Hindcast time selection
    # -------------------------------------------------------------------------
    def _select_time_hindcast(self, da: xr.DataArray) -> xr.DataArray:
        if "time" not in da.dims:
            return da
        n = da.sizes.get("time", 0)
        if n == 1:
            return da.isel(time=0, drop=True)

        mode = self.cfg.hindcast_time_mode.lower()

        if mode == "00z":
            try:
                hours = da["time"].dt.hour
                mask = (hours == 0)
                if bool(mask.any()):
                    t00 = da["time"].where(mask, drop=True)
                    sel = da.sel(time=t00).isel(time=0, drop=True)
                    if self.debug:
                        print(f"[INFO] Hindcast 00Z selecionado: {t00.values}")
                    return sel
            except Exception:
                pass
            if self.debug:
                print("[WARN] Não consegui selecionar 00Z; usando time[0].")
            return da.isel(time=0, drop=True)

        if mode == "first":
            return da.isel(time=0, drop=True)

        if mode == "mean":
            return da.mean("time", skipna=True)

        raise ValueError("hindcast_time_mode inválido (00z|first|mean).")

    # -------------------------------------------------------------------------
    # XESMF regridder build + apply
    # -------------------------------------------------------------------------
    def _build_regridder(self, ds_in: xr.Dataset) -> None:
        ds_in = normalize_coords_latlon(ds_in)
        ds_in = sort_latlon(ds_in)

        grid_in = xr.Dataset(
            {"lat": (("lat",), ds_in["latitude"].values),
             "lon": (("lon",), ds_in["longitude"].values)}
        )
        grid_out = xr.Dataset(
            {"lat": (("lat",), self.lat_ref.values),
             "lon": (("lon",), self.lon_ref.values)}
        )

        wfile = str(self.out_hindcast_root / "regrid_weights.nc")
        reuse = Path(wfile).exists()

        if self.debug:
            print(f"[INFO] XESMF weights: {wfile} | reuse={reuse}")

        self._regridder = xe.Regridder(
            grid_in, grid_out, "bilinear",
            filename=wfile,
            reuse_weights=reuse
        )

    def regrid_to_ref(self, ds: xr.Dataset) -> xr.Dataset:
        ds = normalize_coords_latlon(ds)
        ds = sort_latlon(ds)

        if self.cfg.adjust_lon_360_to_180:
            try:
                if float(ds["longitude"].max()) > 180 and float(self.lon_ref.min()) < 0:
                    lon_adj = ((ds["longitude"] + 180) % 360) - 180
                    ds = ds.assign_coords(longitude=lon_adj)
                    ds = ds.sortby("longitude")
            except Exception:
                pass

        if self._regridder is None:
            self._build_regridder(ds)

        out = self._regridder(ds)
        if "lat" in out.coords and "latitude" not in out.coords:
            out = out.rename({"lat": "latitude"})
        if "lon" in out.coords and "longitude" not in out.coords:
            out = out.rename({"lon": "longitude"})
        return sort_latlon(out)

    # -------------------------------------------------------------------------
    # Hindcast files
    # -------------------------------------------------------------------------
    def _list_hindcast_files_and_leads(self) -> Tuple[List[int], List[Path]]:
        files = sorted(self.hindcast_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"Nenhum hindcast .nc em: {self.hindcast_dir}")

        lead_to_file: Dict[int, Path] = {}
        for f in files:
            ld = extract_lead_from_filename(f.name)
            if ld is not None:
                lead_to_file[ld] = f

        leads = [k for k in range(1, self.cfg.hindcast_leads_expected + 1) if k in lead_to_file]
        if leads:
            return leads, [lead_to_file[k] for k in leads]

        if self.debug:
            print("[WARN] Não achei lead pelo nome. Usando ordem alfabética como lead=1..N.")
        lead_files = sorted(files)[: self.cfg.hindcast_leads_expected]
        leads = list(range(1, len(lead_files) + 1))
        return leads, lead_files

    # -------------------------------------------------------------------------
    # Hindcast processed (monthly totals by lead)
    # -------------------------------------------------------------------------
    def build_hindcast_processed(self) -> xr.Dataset:
        leads, lead_files = self._list_hindcast_files_and_leads()

        if self.debug:
            print(f"\n=== HINDCAST build (DOY={self.doy}) ===")
            print("leads (available):", leads)
            for f in lead_files:
                print(" -", f.name)

        fields: List[xr.DataArray] = []
        months: List[int] = []

        for ld, f in zip(leads, lead_files):
            ds = xr.open_dataset(f)
            ds = normalize_coords_latlon(ds)
            ds = sort_latlon(ds)

            v = choose_var_name(ds, self.var_name, self.var_override)
            da = ds[v]
            da2 = self._select_time_hindcast(da)

            extra = [d for d in da2.dims if d not in ("latitude", "longitude")]
            if extra:
                da2 = da2.squeeze(extra, drop=True)

            if set(da2.dims) != {"latitude", "longitude"}:
                raise ValueError(f"Hindcast esperado 2D lat/lon após seleção. Dims={da2.dims} em {f}")

            da2 = da2.astype("float32")
            da2.name = self.var_name

            m = month_from_doy_and_lead(self.doy, int(ld))
            fields.append(da2)
            months.append(m)

        da_all = xr.concat(fields, dim="lead")
        ds_out = xr.Dataset(
            data_vars={self.var_name: (("lead", "latitude", "longitude"), da_all.values)},
            coords={
                "lead": ("lead", np.array(leads, dtype="int32")),
                "month": ("lead", np.array(months, dtype="int32")),
                "latitude": ("latitude", da_all["latitude"].values),
                "longitude": ("longitude", da_all["longitude"].values),
            },
        )

        ds_out = self.regrid_to_ref(ds_out)
        return ds_out

    def save_hindcast_processed(self, ds_h: xr.Dataset) -> Path:
        out_path = self.out_hindcast_root / f"cfs_hindcast_{self.var_name}_doy{self.doy}.nc"
        if self.debug:
            print(f"[INFO] Salvando hindcast processado: {out_path}")
        enc = {self.var_name: {"dtype": "float32", "zlib": bool(self.cfg.zlib)}}
        ds_h.to_netcdf(out_path, engine=self.cfg.netcdf_engine, encoding=enc)
        return out_path

    # -------------------------------------------------------------------------
    # Forecast 6/6h (mm) -> daily sum (mm/day)
    # -------------------------------------------------------------------------
    def forecast_to_daily_accum_mm(self, da_mm: xr.DataArray) -> xr.DataArray:
        if "time" not in da_mm.dims:
            raise ValueError("Forecast precisa ter dimensão 'time'.")
        out = da_mm.resample(time="1D").sum(keep_attrs=True)
        out = out.astype("float32")
        out.attrs = da_mm.attrs.copy()
        out.attrs["units"] = "mm/day"
        return out

    # -------------------------------------------------------------------------
    # Correct forecast (daily)
    # -------------------------------------------------------------------------
    def correct_forecast_file(self, forecast_path: Path, ds_h: xr.Dataset) -> Tuple[xr.Dataset, str]:
        ds_f = xr.open_dataset(forecast_path)
        ds_f = normalize_coords_latlon(ds_f)
        ds_f = sort_latlon(ds_f)

        v = choose_var_name(ds_f, self.var_name, self.var_override)
        init_stamp = init_stamp_from_filename_strict(forecast_path)
        fore = ds_f[v]

        if "time" not in ds_f.coords or ds_f["time"].size < 1:
            raise ValueError("Não consegui obter o tempo inicial do forecast.")
        t_init = ds_f["time"].values[0].astype("datetime64[D]")

        hind_m = ds_h[self.var_name].astype("float32")
        hm = ds_h["month"]
        n_hind_leads = int(hind_m.sizes["lead"])

        extend_months = int(self.cfg.extend_months)
        max_lead_out = n_hind_leads + extend_months

        # CUT 6/6h by lead range (before aggregation) - now up to max_lead_out
        t_6h_days = ds_f["time"].values.astype("datetime64[D]")
        lead0_6h = (t_6h_days.astype("datetime64[M]").astype(int) - t_init.astype("datetime64[M]").astype(int))
        lead_arr_6h = (lead0_6h + 1).astype(np.int32)

        valid_idx_6h = np.where((lead_arr_6h >= 1) & (lead_arr_6h <= max_lead_out))[0]
        if valid_idx_6h.size < 1:
            raise RuntimeError("Nenhum timestep 6/6h do forecast está dentro do alcance de leads de saída.")

        fore_slice = fore.isel(time=valid_idx_6h)

        if self.debug:
            print(f"\n=== CORRIGINDO (DIÁRIO) {forecast_path.name} ===")
            print("init_stamp:", init_stamp)
            print("n_hind_leads:", n_hind_leads,
                  "| extend_months:", extend_months,
                  "| max_lead_out:", max_lead_out,
                  "| kept_6h_steps:", int(valid_idx_6h.size),
                  "| total_6h_steps:", int(ds_f['time'].size))

        t0 = time.time()
        fore_d = self.forecast_to_daily_accum_mm(fore_slice)
        if self.debug:
            print("[T] after aggregation daily:", round(time.time() - t0, 2), "s")
            t0 = time.time()

        fore_d_ref = self.regrid_to_ref(fore_d.to_dataset(name=self.var_name))[self.var_name].astype("float32")
        if self.debug:
            print("[T] after regrid (xesmf):", round(time.time() - t0, 2), "s")
            t0 = time.time()

        # ---------------------------------------------------------------------
        # HARD LIMIT OUTPUT TO 280 DAYS (or cfg.max_days_output)
        # ---------------------------------------------------------------------
        MAX_DAYS = int(self.cfg.max_days_output)
        t_daily_all = fore_d_ref["time"].values.astype("datetime64[D]")
        if t_daily_all.size < 1:
            raise RuntimeError("Forecast diário ficou vazio após resample/regrid.")

        t0_day = t_daily_all[0]
        t_end = t0_day + np.timedelta64(MAX_DAYS - 1, "D")
        idx_280 = np.where((t_daily_all >= t0_day) & (t_daily_all <= t_end))[0]
        if idx_280.size < 1:
            raise RuntimeError("Limite de max_days_output removeu todos os dias (verifique o eixo time).")

        fore_d_ref = fore_d_ref.isel(time=idx_280)

        if self.debug:
            print("[DBG] limited daily days:", int(fore_d_ref["time"].size),
                  "| first:", str(fore_d_ref["time"].values[0]),
                  "| last:", str(fore_d_ref["time"].values[-1]))
        # ---------------------------------------------------------------------

        # day->lead and day->doy(2001) (AFTER limiting)
        t_days = fore_d_ref["time"].values.astype("datetime64[D]")
        lead0 = (t_days.astype("datetime64[M]").astype(int) - t_init.astype("datetime64[M]").astype(int))
        lead_arr = (lead0 + 1).astype(np.int32)
        doys = np.array([doy_of_2001_from_yyyy_mm_dd(t) for t in t_days], dtype=np.int32)

        # keep only leads within output range (1..max_lead_out)
        keep_d = np.where((lead_arr >= 1) & (lead_arr <= max_lead_out))[0]
        if keep_d.size < 1:
            raise RuntimeError("Após limite de dias, nenhum dia ficou dentro do alcance de leads de saída.")

        fore_d_ref = fore_d_ref.isel(time=keep_d)
        t_days = fore_d_ref["time"].values.astype("datetime64[D]")
        lead_arr = lead_arr[keep_d]
        doys = doys[keep_d]

        # lead_used: clamp beyond hindcast to last hindcast lead (Option 2 behavior)
        lead_used = np.minimum(lead_arr, n_hind_leads).astype(np.int32)

        if self.debug:
            print("[DBG] daily days kept after lead filter:", int(fore_d_ref["time"].size),
                  "| lead_arr min/max:", int(lead_arr.min()), int(lead_arr.max()),
                  "| lead_used min/max:", int(lead_used.min()), int(lead_used.max()))

        # clim for all selected days (FAST gather)
        clim_all = self._clim_lookup[doys, :, :]  # (T, y, x)

        if self.debug:
            print("[T] built clim_stack (FAST gather):", round(time.time() - t0, 2), "s")
            t0 = time.time()

        corr_np = fore_d_ref.values.astype(np.float32, copy=True)   # (T, y, x)
        hind_np = hind_m.values.astype(np.float32)                  # (lead, y, x)

        denom_min = 1e-3
        alpha = denom_min
        p = 0.30

        for L in range(1, n_hind_leads + 1):
            idx_all = np.where(lead_used == L)[0]
            if idx_all.size < 1:
                continue

            hind_total_m = hind_np[L - 1, :, :]  # (y,x) mm/mês do lead L (para L=last, repetido)

            if L < n_hind_leads:
                # Leads normais: usa mês do próprio hindcast-lead para os pesos
                mes_weights = int(hm.isel(lead=L - 1).values)

                t_cache = time.time()
                month_sum, _idx_map = self._get_month_sum_and_idxmap(mes_weights)
                if self.debug:
                    print(f"[T] month cache load/build for month={mes_weights:02d}:", round(time.time() - t_cache, 2), "s")

                doys_sel = doys[idx_all]
                if (doys_sel > self._max_doy).any():
                    nd = float(idx_all.size)
                    w_stack = np.full((idx_all.size, self._ny, self._nx), 1.0 / nd, dtype=np.float32)
                else:
                    clim_sel = self._clim_lookup[doys_sel, :, :]
                    nd = float(idx_all.size)
                    w_stack = np.where(
                        month_sum[None, :, :] > 0,
                        clim_sel / month_sum[None, :, :],
                        (1.0 / nd)
                    ).astype(np.float32)

                hind_est_d = hind_total_m[None, :, :] * w_stack

                clim_obs_d  = clim_all[idx_all, :, :]
                clim_hind_d = hind_est_d
                fore_d_blk  = corr_np[idx_all, :, :]

                factor = (clim_obs_d + alpha) / (clim_hind_d + alpha)
                corr_ratio = fore_d_blk * factor
                corr_add   = fore_d_blk + (clim_obs_d - clim_hind_d)

                corr_candidate = np.where(
                    clim_hind_d < denom_min,
                    corr_add,
                    corr_ratio
                ).astype(np.float32)

                corr_candidate = np.where(
                    clim_obs_d == 0,
                    fore_d_blk,
                    corr_candidate
                ).astype(np.float32)

                cmin = (1.0 - p) * clim_obs_d
                cmax = (1.0 + p) * clim_obs_d
                corr_limited = np.clip(corr_candidate, cmin, cmax).astype(np.float32)

                corr_np[idx_all, :, :] = corr_limited

            else:
                # Último lead: aplicar pesos por mês REAL (dos dias), mantendo o total do último lead.
                months_real = t_days[idx_all].astype("datetime64[M]")
                uniq_months = np.unique(months_real)

                for mM in uniq_months:
                    idx_m = idx_all[np.where(months_real == mM)[0]]
                    if idx_m.size < 1:
                        continue

                    mes_weights = int(str(mM)[5:7])  # 1..12

                    t_cache = time.time()
                    month_sum, _idx_map = self._get_month_sum_and_idxmap(mes_weights)
                    if self.debug:
                        print(f"[T] month cache load/build for month={mes_weights:02d}:", round(time.time() - t_cache, 2), "s")

                    doys_sel = doys[idx_m]
                    if (doys_sel > self._max_doy).any():
                        nd = float(idx_m.size)
                        w_stack = np.full((idx_m.size, self._ny, self._nx), 1.0 / nd, dtype=np.float32)
                    else:
                        clim_sel = self._clim_lookup[doys_sel, :, :]
                        nd = float(idx_m.size)
                        w_stack = np.where(
                            month_sum[None, :, :] > 0,
                            clim_sel / month_sum[None, :, :],
                            (1.0 / nd)
                        ).astype(np.float32)

                    hind_est_d = hind_total_m[None, :, :] * w_stack

                    clim_obs_d  = clim_all[idx_m, :, :]
                    clim_hind_d = hind_est_d
                    fore_d_blk  = corr_np[idx_m, :, :]

                    factor = (clim_obs_d + alpha) / (clim_hind_d + alpha)
                    corr_ratio = fore_d_blk * factor
                    corr_add   = fore_d_blk + (clim_obs_d - clim_hind_d)

                    corr_candidate = np.where(
                        clim_hind_d < denom_min,
                        corr_add,
                        corr_ratio
                    ).astype(np.float32)

                    corr_candidate = np.where(
                        clim_obs_d == 0,
                        fore_d_blk,
                        corr_candidate
                    ).astype(np.float32)

                    cmin = (1.0 - p) * clim_obs_d
                    cmax = (1.0 + p) * clim_obs_d
                    corr_limited = np.clip(corr_candidate, cmin, cmax).astype(np.float32)

                    corr_np[idx_m, :, :] = corr_limited

        corr_np = np.clip(corr_np, 0, None).astype(np.float32)

        if self.debug:
            print("[T] after correction blocks (FAST):", round(time.time() - t0, 2), "s")

        corr = xr.DataArray(
            corr_np,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": fore_d_ref["time"].values,
                "latitude": fore_d_ref["latitude"].values,
                "longitude": fore_d_ref["longitude"].values,
            },
            name=self.var_name,
            attrs=fore_d_ref.attrs.copy(),
        )

        ds_out = corr.to_dataset(name=self.var_name)
        ds_out.attrs = ds_f.attrs
        return ds_out, init_stamp

    def save_corrected_forecast(self, ds_corr: xr.Dataset, init_stamp: str) -> Path:
        out_name = build_forecast_outname(self.var_name, init_stamp)
        out_path = self.out_corr_root / out_name
        if self.debug:
            print(f"[INFO] Salvando forecast corrigido: {out_path}")
        enc = {self.var_name: {"dtype": "float32", "zlib": bool(self.cfg.zlib)}}
        ds_corr.to_netcdf(out_path, engine=self.cfg.netcdf_engine, encoding=enc)
        return out_path

    def run(self) -> None:
        ds_h = self.build_hindcast_processed()
        self.save_hindcast_processed(ds_h)

        for fp in self.forecast_files:
            ds_corr, init_stamp = self.correct_forecast_file(fp, ds_h)
            self.save_corrected_forecast(ds_corr, init_stamp)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CFS pipeline (DIÁRIO): hindcast mensal + forecast 6/6h (mm) -> correção diária com climatologia diária + XESMF regrid. "
                    "Climatologia é lida em bloco (rápido) e month_sum é cacheado sob demanda."
    )
    p.add_argument("--forecast", required=True, help="Pasta DOY do forecast OU arquivo .nc dentro dela")
    p.add_argument("--hindcast-root", required=True, help="Diretório base do hindcast")
    p.add_argument("--clim-obs", required=True, help="Climatologia observada diária (time=365/366) mm/dia")
    p.add_argument("--ref-grid", required=True, help="Arquivo NetCDF com a grade final (use o mesmo da climatologia)")
    p.add_argument("--out-hindcast", required=True, help="Base output hindcast processado")
    p.add_argument("--out-corr", required=True, help="Base output forecast corrigido")
    p.add_argument("--year", type=int, default=None, help="Ano para usar nas pastas de saída")
    p.add_argument("--var", default=None, help="Nome da variável (default total_precipitation; ou auto se único)")
    p.add_argument("--hindcast-time-mode", choices=["00z", "first", "mean"], default="00z",
                   help="Hindcast time: 00z (default), first, mean")
    p.add_argument("--hindcast-leads-expected", type=int, default=9,
                   help="Número máximo de leads (meses) para tentar usar (default=9).")
    # NOTE: extend_months is fixed in config as 1; no CLI flag needed.
    p.add_argument("--netcdf-engine", default="netcdf4", help="Engine do to_netcdf (recomendado: netcdf4)")
    p.add_argument("--zlib", action="store_true", help="Ativa compressão (mais lento).")
    p.add_argument("--no-month-cache-disk", action="store_true",
                   help="Desativa cache em disco do month_sum/idx_map (fica só em memória).")
    p.add_argument("--debug", action="store_true", help="Logs verbosos")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PipelineConfig(
        hindcast_time_mode=args.hindcast_time_mode,
        hindcast_leads_expected=int(args.hindcast_leads_expected),
        netcdf_engine=str(args.netcdf_engine),
        zlib=bool(args.zlib),
        cache_month_weights_to_disk=(not bool(args.no_month_cache_disk)),
        extend_months=1,          # ALWAYS 1
        max_days_output=280,      # ALWAYS 280 output days
    )

    pipe = CFSPipelineAuto(
        cfg=cfg,
        forecast_input=Path(args.forecast),
        hindcast_root=Path(args.hindcast_root),
        clim_obs_file=Path(args.clim_obs),
        ref_grid_file=Path(args.ref_grid),
        out_hindcast_base=Path(args.out_hindcast),
        out_corr_base=Path(args.out_corr),
        year_override=args.year,
        var_override=args.var,
        debug=bool(args.debug),
    )
    pipe.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
