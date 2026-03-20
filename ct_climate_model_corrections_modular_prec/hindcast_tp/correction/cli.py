from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import CorrectionConfig
from .pipeline import ForecastCorrectionPipeline


def parse_args() -> CorrectionConfig:
    p = argparse.ArgumentParser(
        description=(
            "Apply DAILY correction to raw 6/6h forecasts using monthly hindcast (by lead) + observed DAILY climatology. "
            "Forecast is aggregated to daily totals (mm/day), regridded to the climatology grid, then corrected day-by-day."
        )
    )
    p.add_argument("--forecast-root", type=Path, required=True, help="Raw forecast NetCDF root (contains DOY subfolders).")
    p.add_argument("--hindcast-root", type=Path, required=True, help="Processed hindcast root (contains DOY subfolders).")
    p.add_argument("--clim-file", type=Path, required=True, help="Observed DAILY climatology file (time>=365) + reference grid.")
    p.add_argument("--out-root", type=Path, required=True, help="Output base directory.")
    p.add_argument("--var-name", type=str, default="total_precipitation", help="Variable name to correct.")
    p.add_argument("--to-mm", action="store_true", help="Convert meters -> millimeters by multiplying by 1000.")
    p.add_argument("--subfolder", type=str, default=None, help="Process only one DOY subfolder (e.g., 335).")
    p.add_argument("--no-skip-existing", action="store_true", help="Do not skip outputs that already exist.")

    # optional tuning (keep defaults if not provided)
    p.add_argument("--limit-p", type=float, default=0.30, help="Clamp corrected values to climatology ±p (default 0.30).")
    p.add_argument("--denom-min", type=float, default=1e-3, help="Fallback threshold for multiplicative correction (default 1e-3).")

    a = p.parse_args()
    return CorrectionConfig(
        forecast_root=a.forecast_root.expanduser().resolve(),
        hindcast_root=a.hindcast_root.expanduser().resolve(),
        clim_file=a.clim_file.expanduser().resolve(),
        out_root=a.out_root.expanduser().resolve(),
        var_name=a.var_name,
        to_mm=bool(a.to_mm),
        subfolder=a.subfolder,
        skip_existing=(not a.no_skip_existing),
        denom_min=float(a.denom_min),
        alpha=float(a.denom_min),
        limit_p=float(a.limit_p),
    )


def main() -> None:
    cfg = parse_args()
    try:
        ForecastCorrectionPipeline(cfg).run()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
