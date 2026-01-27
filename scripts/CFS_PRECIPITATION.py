#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class PipelineConfig:
    default_var: str = "total_precipitation"
    hindcast_leads_expected: int = 9  # maximum leads to look for/use
    hindcast_time_mode: str = "00z"   # "00z" | "first" | "mean"
    adjust_lon_360_to_180: bool = True


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


def month_from_doy_and_lead(doy_str: str, lead: int) -> int:
    start_doy = int(doy_str)
    start_date = np.datetime64("2001-01-01") + np.timedelta64(start_doy - 1, "D")
    start_month = int(str(start_date)[5:7])
    return int(((start_month - 1 + (lead - 1)) % 12) + 1)


def valid_time_from_month(month: int) -> np.datetime64:
    return np.datetime64(f"2001-{month:02d}-01")


def is_monthly_time_axis(time_values: np.ndarray) -> bool:
    if time_values.size < 1:
        return False
    tM = time_values.astype("datetime64[M]")
    if len(np.unique(tM)) != len(tM):
        return False
    if len(tM) == 1:
        return True
    dif = np.diff(tM.astype("int64"))
    return np.all(dif == 1)


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


def parse_init_from_time_units(units: str) -> Optional[str]:
    """
    Ex: 'hours since 1982-12-01 00:00:00' -> '1982120100'
    """
    if not units:
        return None
    m = re.search(r"since\s+(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", units)
    if not m:
        return None
    yyyy, mm, dd, hh, _, _ = m.groups()
    return f"{yyyy}{mm}{dd}{hh}"


def init_stamp_from_dataset(ds_f: xr.Dataset) -> str:
    """
    Prioridade:
    1) time.units "hours since YYYY-MM-DD HH:MM:SS"
    2) primeiro valor de time
    """
    if "time" in ds_f.coords:
        units = ds_f["time"].attrs.get("units", "")
        stamp = parse_init_from_time_units(units)
        if stamp:
            return stamp

        try:
            t0 = ds_f["time"].values[0].astype("datetime64[h]")
            s = str(t0)  # '1982-12-01T00'
            yyyy = s[0:4]
            mm = s[5:7]
            dd = s[8:10]
            hh = s[11:13]
            return f"{yyyy}{mm}{dd}{hh}"
        except Exception:
            pass

    return "0000010100"


def build_forecast_outname(var_name: str, init_stamp: str) -> str:
    return f"cfs_glo_{var_name}_M000_{init_stamp}.nc"


class CFSPipelineAuto:
    def _find_hindcast_dir_for_doy(self) -> Path:
        """
        Procura pela pasta de hindcast correspondente ao DOY, ignorando o ano.
        Preferência:
         1) <hindcast_root>/<VAR>/<YYYY>/<DOY> onde YYYY é numérico (maior YYYY primeiro)
         2) qualquer diretório dentro de <hindcast_root>/<VAR> com nome igual ao DOY (rglob)
         3) fallback antigo: <hindcast_root>/<DOY> (mantém compatibilidade se estrutura for diferente)
        """
        # Prefer var-specific structure under hindcast_root
        var_root = self.hindcast_root / self.var_name
        if var_root.is_dir():
            # 1) numeric year children, prefer newest year
            year_dirs = [
                d for d in var_root.iterdir() if d.is_dir() and d.name.isdigit() and len(d.name) == 4
            ]
            year_dirs = sorted(year_dirs, key=lambda p: int(p.name), reverse=True)
            for ydir in year_dirs:
                cand = ydir / self.doy
                if cand.is_dir():
                    return cand

            # 2) immediate children that may contain DOY
            for child in sorted(var_root.iterdir()):
                if child.is_dir():
                    cand = child / self.doy
                    if cand.is_dir():
                        return cand

            # 3) recursive search under var_root
            candidates = [p for p in var_root.rglob(self.doy) if p.is_dir()]
            if candidates:
                return sorted(candidates)[0]

        # 4) backward-compatible: direct DOY under hindcast_root
        cand_root = self.hindcast_root / self.doy
        if cand_root.is_dir():
            return cand_root

        # 5) recursive search under hindcast_root
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
            self.forecast_files = sorted(self.forecast_dir.glob("*.nc"))
        else:
            self.forecast_dir = self.forecast_input.parent
            self.forecast_files = [self.forecast_input]

        if not self.forecast_files:
            raise FileNotFoundError(f"Nenhum forecast .nc encontrado em: {self.forecast_dir}")

        if not self.ref_grid_file.exists():
            raise FileNotFoundError(f"Ref-grid não existe: {self.ref_grid_file}")

        # Load climatology first to detect the variable name (needed to locate hindcast by var)
        self.clim_obs, self.var_name = self._load_obs_climatology()
        self.lat_ref, self.lon_ref = load_ref_latlon_from_refgrid(self.ref_grid_file)

        # Hindcast: procurar APENAS pelo DOY (ignora o ano), preferindo estrutura var-specific
        self.hindcast_dir = self._find_hindcast_dir_for_doy()
        if not self.hindcast_dir.is_dir():
            raise NotADirectoryError(f"Pasta de hindcast não encontrada: {self.hindcast_dir}")

        self.out_hindcast_root = self.out_hindcast_base / self.var_name / f"{self.year:04d}" / self.doy
        self.out_corr_root = self.out_corr_base / self.var_name / f"{self.year:04d}" / self.doy
        ensure_dir(self.out_hindcast_root)
        ensure_dir(self.out_corr_root)

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

    def _load_obs_climatology(self) -> Tuple[Dict[int, xr.DataArray], str]:
        if not self.clim_obs_file.exists():
            raise FileNotFoundError(f"Climatologia observada não existe: {self.clim_obs_file}")

        ds = xr.open_dataset(self.clim_obs_file)
        ds = normalize_coords_latlon(ds)
        ds = sort_latlon(ds)

        v = choose_var_name(ds, self.cfg.default_var, self.var_override)
        da = ds[v]

        if "time" not in da.dims or da.sizes["time"] != 12:
            raise ValueError("Climatologia observada precisa ter time=12.")

        clim: Dict[int, xr.DataArray] = {}
        for t in da.time:
            m = int(t.dt.month.values)
            clim[m] = da.sel(time=t).drop_vars("time")

        missing = [m for m in range(1, 13) if m not in clim]
        if missing:
            raise ValueError(f"Climatologia observada sem meses: {missing}")

        return clim, v

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
                        try:
                            print(f"[INFO] Hindcast 00Z selecionado: {t00.values}")
                        except Exception:
                            pass
                    return sel
            except Exception:
                pass
            if self.debug:
                print("[WARN] Não consegui selecionar 00Z por dt.hour; usando time[0].")
            return da.isel(time=0, drop=True)

        if mode == "first":
            return da.isel(time=0, drop=True)

        if mode == "mean":
            return da.mean("time", skipna=True)

        raise ValueError("hindcast_time_mode inválido (00z|first|mean).")

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

        ds = ds.interp(latitude=self.lat_ref, longitude=self.lon_ref)
        ds = sort_latlon(ds)
        return ds

    def _list_hindcast_files_and_leads(self) -> Tuple[List[int], List[Path]]:
        files = sorted(self.hindcast_dir.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"Nenhum hindcast .nc em: {self.hindcast_dir}")

        lead_to_file: Dict[int, Path] = {}
        for f in files:
            ld = extract_lead_from_filename(f.name)
            if ld is not None:
                lead_to_file[ld] = f

        # Prefer explicit M001..M00N if found; use up to hindcast_leads_expected
        leads = [k for k in range(1, self.cfg.hindcast_leads_expected + 1) if k in lead_to_file]
        if leads:
            return leads, [lead_to_file[k] for k in leads]

        # Fallback: alphabetical order (up to hindcast_leads_expected)
        if self.debug:
            print("[WARN] Não achei lead pelo nome (M001/M002...). Usando ordem alfabética como lead=1..N.")
        lead_files = sorted(files)[: self.cfg.hindcast_leads_expected]
        leads = list(range(1, len(lead_files) + 1))
        return leads, lead_files

    def build_hindcast_processed(self) -> xr.Dataset:
        leads, lead_files = self._list_hindcast_files_and_leads()

        if self.debug:
            print(f"\n=== HINDCAST build (DOY={self.doy}) ===")
            print("leads (available):", leads)
            for f in lead_files:
                print(" -", f.name)

        fields: List[xr.DataArray] = []
        months: List[int] = []
        vtimes: List[np.datetime64] = []

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
            vtimes.append(valid_time_from_month(m))

        da_all = xr.concat(fields, dim="lead")
        ds_out = xr.Dataset(
            data_vars={self.var_name: (("lead", "latitude", "longitude"), da_all.values)},
            coords={
                "lead": ("lead", np.array(leads, dtype="int32")),
                "month": ("lead", np.array(months, dtype="int32")),
                "valid_time": ("lead", np.array(vtimes, dtype="datetime64[ns]")),
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
        ds_h.to_netcdf(out_path)
        return out_path

    def forecast_to_monthly_accum(self, da: xr.DataArray) -> xr.DataArray:
        if "time" not in da.dims:
            raise ValueError("Forecast precisa ter dimensão 'time'.")
        tvals = da["time"].values.astype("datetime64[ns]")
        if is_monthly_time_axis(tvals):
            return da
        return da.resample(time="MS").sum(keep_attrs=True)

    def correct_forecast_file(self, forecast_path: Path, ds_h: xr.Dataset) -> Tuple[xr.Dataset, str]:
        ds_f = xr.open_dataset(forecast_path)
        ds_f = normalize_coords_latlon(ds_f)
        ds_f = sort_latlon(ds_f)

        v = choose_var_name(ds_f, self.var_name, self.var_override)
        init_stamp = init_stamp_from_dataset(ds_f)

        fore = ds_f[v]
        fore_m = self.forecast_to_monthly_accum(fore)

        # forecast -> climatology grid
        fore_m_ref = self.regrid_to_ref(fore_m.to_dataset(name=self.var_name))[self.var_name]

        hind = ds_h[self.var_name]
        hm = ds_h["month"]

        # Use hindcast lead count as the controlling parameter
        n_corr = min(int(hind.sizes["lead"]), int(fore_m_ref.sizes["time"]))

        if n_corr < 1:
            raise RuntimeError("n_corr < 1 (hindcast lead count or forecast monthly time count is zero).")

        if self.debug:
            k = min(6, n_corr)
            hm_k = hm.isel(lead=slice(0, k)).values.astype(int).tolist()
            hm_obs = [((m % 12) + 1) for m in hm_k]
            print(f"\n=== CORRIGINDO {forecast_path.name} ===")
            print("hind_month (lead):", hm_k)
            print("mes obs usado (+1):", hm_obs)
            print("init_stamp:", init_stamp)
            print("n_hind_leads:", int(hind.sizes["lead"]), "| n_fore_months:", int(fore_m_ref.sizes["time"]), "| n_out:", n_corr)

        # IMPORTANT CHANGE:
        # Output contains ONLY corrected months (no uncorrected tail).
        corr = fore_m_ref.isel(time=slice(0, n_corr)).copy()

        for idx in range(n_corr):
            mes_hind = int(hm.isel(lead=idx).values)
            mes_obs = (mes_hind % 12) + 1

            clim_obs_m = self.clim_obs[mes_obs]
            clim_hind_m = hind.isel(lead=idx)

            vies = clim_obs_m - clim_hind_m
            corr[idx, :, :] = corr[idx, :, :] + vies

        # For precipitation, keep non-negative
        corr = corr.clip(min=0)

        corr.name = self.var_name
        ds_out = corr.to_dataset(name=self.var_name)
        ds_out.attrs = ds_f.attrs
        ds_out = ds_out.assign_coords(time=corr["time"])
        return ds_out, init_stamp

    def save_corrected_forecast(self, ds_corr: xr.Dataset, init_stamp: str) -> Path:
        out_name = build_forecast_outname(self.var_name, init_stamp)
        out_path = self.out_corr_root / out_name
        if self.debug:
            print(f"[INFO] Salvando forecast corrigido: {out_path}")
        ds_corr.to_netcdf(out_path)
        return out_path

    def run(self) -> None:
        ds_h = self.build_hindcast_processed()
        self.save_hindcast_processed(ds_h)

        for fp in self.forecast_files:
            ds_corr, init_stamp = self.correct_forecast_file(fp, ds_h)
            self.save_corrected_forecast(ds_corr, init_stamp)


def parse_args():
    p = argparse.ArgumentParser(description="CFS pipeline: hindcast (DOY) + forecast -> regrid climatologia + LS-Add.")
    p.add_argument("--forecast", required=True, help="Pasta DOY do forecast OU arquivo .nc dentro dela")
    p.add_argument("--hindcast-root", required=True, help="Diretório base do hindcast que contém subpastas 001..366 (ou estrutura por var/ano/DOY)")
    p.add_argument("--clim-obs", required=True, help="Climatologia observada mensal (time=12) (também define a grade)")
    p.add_argument("--ref-grid", required=True, help="Arquivo NetCDF com a grade final (use o mesmo da climatologia)")
    p.add_argument("--out-hindcast", required=True, help="Base output hindcast processado")
    p.add_argument("--out-corr", required=True, help="Base output forecast corrigido")
    p.add_argument("--year", type=int, default=None, help="Ano para usar nas pastas de saída (se não inferir do caminho)")
    p.add_argument("--var", default=None, help="Nome da variável (default total_precipitation; ou auto se único)")
    p.add_argument("--hindcast-time-mode", choices=["00z", "first", "mean"], default="00z",
                   help="Hindcast time: 00z (default), first, mean")
    p.add_argument("--hindcast-leads-expected", type=int, default=9,
                   help="Número máximo de leads para tentar usar (default=9). Se houver menos no hindcast, usa menos.")
    p.add_argument("--debug", action="store_true", help="Logs verbosos")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = PipelineConfig(
        hindcast_time_mode=args.hindcast_time_mode,
        hindcast_leads_expected=int(args.hindcast_leads_expected),
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
