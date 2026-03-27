from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

DEFAULT_DATASET = "seasonal-monthly-single-levels"
DEFAULT_ORIGINATING_CENTRE = "ecmwf"
DEFAULT_SYSTEM = "51"

DEFAULT_VARIABLE = ["total_precipitation"]
DEFAULT_PRODUCT_TYPE = ["monthly_mean"]
DEFAULT_DAY = ["01"]
DEFAULT_TIME = ["00:00"]
DEFAULT_LEADTIME_MONTH = ["1", "2", "3", "4", "5", "6"]
DEFAULT_DATA_FORMAT = "grib"
DEFAULT_AREA = [90, -180, -90, 180]  # global
DEFAULT_YEARS = [str(y) for y in range(1993, 2017)]  # 1993–2016

DEFAULT_REGRID_METHOD = "bilinear"
DEFAULT_REGRID_PERIODIC = True
DEFAULT_REUSE_WEIGHTS = True
DEFAULT_SAVE_REGRID_WEIGHTS = True
DEFAULT_REGRID_CACHE_SUBDIR = "cache"

DEFAULT_MODEL_PREFIX = "ecmwf_subseas_glo"
DEFAULT_INPUT_VAR = "tprate"

@dataclass(frozen=True)
class HindcastConfig:
    dataset: str = DEFAULT_DATASET
    originating_centre: str = DEFAULT_ORIGINATING_CENTRE
    system: str = DEFAULT_SYSTEM

    variable: Optional[list[str]] = None
    product_type: Optional[list[str]] = None
    day: Optional[list[str]] = None
    time: Optional[list[str]] = None
    leadtime_month: Optional[list[str]] = None
    data_format: str = DEFAULT_DATA_FORMAT
    area: Optional[list[float]] = None
    years: Optional[list[str]] = None

    model_prefix: str = DEFAULT_MODEL_PREFIX
    input_var: str = DEFAULT_INPUT_VAR

    regrid_method: str = DEFAULT_REGRID_METHOD
    regrid_periodic: bool = DEFAULT_REGRID_PERIODIC
    reuse_weights: bool = DEFAULT_REUSE_WEIGHTS
    save_regrid_weights: bool = DEFAULT_SAVE_REGRID_WEIGHTS
    regrid_cache_subdir: str = DEFAULT_REGRID_CACHE_SUBDIR

    def __post_init__(self):
        object.__setattr__(self, "variable", DEFAULT_VARIABLE if self.variable is None else self.variable)
        object.__setattr__(self, "product_type", DEFAULT_PRODUCT_TYPE if self.product_type is None else self.product_type)
        object.__setattr__(self, "day", DEFAULT_DAY if self.day is None else self.day)
        object.__setattr__(self, "time", DEFAULT_TIME if self.time is None else self.time)
        object.__setattr__(self, "leadtime_month", DEFAULT_LEADTIME_MONTH if self.leadtime_month is None else self.leadtime_month)
        object.__setattr__(self, "area", DEFAULT_AREA if self.area is None else self.area)
        object.__setattr__(self, "years", DEFAULT_YEARS if self.years is None else self.years)
