from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Set

from .config import HindcastConfig
from .pipeline import HindcastPipeline
from .utils import extract_year_from_path, list_doy_subfolders, month_from_doy


def _is_doy_dir(p: Path) -> bool:
    return p.is_dir() and p.name.isdigit() and len(p.name) == 3


def _pick_latest_doy_dir(root: Path) -> Path:
    doy_dirs = [p for p in root.iterdir() if _is_doy_dir(p)]
    if not doy_dirs:
        raise RuntimeError(f"No DOY subfolders found under: {root}")

    # Ordena por data de modificação (mais recente por último)
    # Desempate pelo número DOY
    doy_dirs.sort(key=lambda p: (p.stat().st_mtime, int(p.name)))
    return doy_dirs[-1]


def parse_args():
    p = argparse.ArgumentParser(description="Download + process hindcast monthly precipitation (modular).")
    p.add_argument("--month", default=None, help="Initialization month override: 01..12 (optional).")
    p.add_argument("--doy-root", default=None, help="Directory containing DOY subfolders like 001, 015, 215 (optional).")
    p.add_argument("--out-grib", required=True, help="Base directory for GRIB outputs")
    p.add_argument("--out-nc", required=True, help="Base directory for NetCDF outputs")
    p.add_argument("--regrid", action="store_true", help="Enable regridding using xesmf")
    p.add_argument("--ref-grid", default=None, help="NetCDF reference grid file (lat/lon or latitude/longitude)")

    p.add_argument("--originating-centre", default="ecmwf", help="CDS originating_centre (e.g., ecmwf, lfpw)")
    p.add_argument("--system", default="51", help="CDS system id (e.g., 51, 9)")
    p.add_argument("--model-prefix", default="ecmwf_subseas_glo", help="Output filename prefix")
    p.add_argument("--input-var", default="tprate", help="Variable to read from GRIB (default: tprate)")

    return p.parse_args()


def main():
    args = parse_args()
    ref_grid = Path(args.ref_grid) if args.ref_grid else None
    var_folder = "total_precipitation"

    if args.doy_root is not None:
        out_year = extract_year_from_path(Path(args.doy_root).expanduser().resolve())
    else:
        try:
            out_year = extract_year_from_path(Path(args.out_nc).expanduser().resolve())
        except Exception:
            out_year = extract_year_from_path(Path(args.out_grib).expanduser().resolve())

    out_grib_root = Path(args.out_grib) / var_folder
    out_nc_root = Path(args.out_nc) / var_folder

    cfg = HindcastConfig(
        originating_centre=str(args.originating_centre),
        system=str(args.system),
        model_prefix=str(args.model_prefix),
        input_var=str(args.input_var),
    )

    pipeline = HindcastPipeline(cfg, out_grib_root, out_nc_root, bool(args.regrid), ref_grid)

    # Se passou --month manual
    if args.month is not None:
        pipeline.run_month(f"{int(args.month):02d}", out_year)
        return

    if args.doy_root is None:
        raise ValueError("You must provide either --month (override) or --doy-root (auto mode).")

    doy_root = Path(args.doy_root).expanduser().resolve()

    # ✅ Novo comportamento:
    # - Se já for .../2024/001 → usa só esse DOY
    # - Se for .../2024 → escolhe automaticamente o DOY mais recente
    if _is_doy_dir(doy_root):
        doys = [int(doy_root.name)]
    else:
        latest = _pick_latest_doy_dir(doy_root)
        doys = [int(latest.name)]

    months_needed: Set[int] = set()
    for doy in doys:
        m = month_from_doy(int(doy), year_dummy=2001)
        months_needed.add(m)


    months_sorted = sorted(months_needed)
    print(f"Detected DOYs: {len(doys)}. Months to download/process: {months_sorted}")

    for m in months_sorted:
        pipeline.run_month(f"{m:02d}", out_year)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
