from __future__ import annotations
from abc import ABC, abstractmethod
import xarray as xr
from ..config import HindcastConfig

class BaseProcessor(ABC):
    name: str

    @abstractmethod
    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        ...

    @abstractmethod
    def process_grib(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        ...
