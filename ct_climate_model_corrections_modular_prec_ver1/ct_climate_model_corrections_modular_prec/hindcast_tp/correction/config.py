from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CorrectionConfig:
    forecast_root: Path
    hindcast_root: Path
    clim_file: Path
    out_root: Path

    var_name: str = "total_precipitation"
    to_mm: bool = False
    subfolder: Optional[str] = None
    skip_existing: bool = True

    # Daily-correction parameters (defaults match the reference daily script)
    denom_min: float = 1e-3
    alpha: float = 1e-3
    limit_p: float = 0.30

    save_regrid_weights: bool = True
    regrid_cache_subdir: str = "cache"
