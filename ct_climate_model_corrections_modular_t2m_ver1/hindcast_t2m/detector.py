from __future__ import annotations
import xarray as xr
from .config import HindcastConfig
from .processing.base import BaseProcessor
from .processing.meteofrance_sys9 import MeteoFranceSys9Processor
from .processing.ecmwf_sys51 import ECMWFSys51Processor
from .processing.time_step_like import TimeStepLikeProcessor
from .processing.ukmo_sys603 import UKMOSystem603Processor
from .processing.jma_sys3 import JMASys3Processor
from .processing.dwd_sys2 import DWDSys2Processor
from .processing.cmcc_sys35 import CMCCSys35Processor
from .processing.eccc_sys4 import ECCCSys4Processor
from .processing.ncep_sys2 import NCEPSys2Processor

_PROCESSORS: list[BaseProcessor] = [
    MeteoFranceSys9Processor(),
    ECMWFSys51Processor(),
    UKMOSystem603Processor(),
    JMASys3Processor(),
    DWDSys2Processor(),
    CMCCSys35Processor(),
    ECCCSys4Processor(),
    NCEPSys2Processor(),
    TimeStepLikeProcessor(),
]

def detect_processor(ds: xr.Dataset, cfg: HindcastConfig) -> BaseProcessor:
    for p in _PROCESSORS:
        if p.can_handle(ds, cfg):
            return p
    raise RuntimeError(
        "No processor matched dataset. "
        f"dims={dict(ds.dims)} coords={list(ds.coords)} attrs_keys={list(ds.attrs.keys())}"
    )
