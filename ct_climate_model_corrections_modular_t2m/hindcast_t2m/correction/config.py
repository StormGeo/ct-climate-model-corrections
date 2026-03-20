from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CorrectionConfig:
    """Daily temperature correction configuration.

    Directory expectations (same as precipitation module):
      - forecast_root: <...>/<VAR>/<YEAR> containing DOY subfolders (001..366)
      - hindcast_root: <...>/<VAR>/<YEAR> containing DOY subfolders (001..366) with processed hindcast
      - clim_file: observed DAILY climatology (time>=365) on the target grid
      - out_root: base output dir; output will be <out_root>/<VAR>/<YEAR>/<DOY>/...
    """

    forecast_root: Path
    hindcast_root: Path
    clim_file: Path
    out_root: Path

    var_name: str = "2m_air_temperature"  # mean | *_min | *_max
    subfolder: Optional[str] = None
    skip_existing: bool = True

    # Daily correction behavior
    max_days_output: int = 280
    extend_months: int = 1  # allow 1 month beyond hindcast leads (repeat last hindcast lead bias)

    # Clipping around daily observed climatology (absolute, degC)
    clip_with_abs_limit: bool = True
    clip_delta_c: float = 1.5
