from __future__ import annotations
from pathlib import Path
import cdsapi
from dotenv import load_dotenv
import os


load_dotenv("/airflow/base/credentials/.felipe_cds")

API = os.getenv("API")
KEY = os.getenv("KEY")

from .config import HindcastConfig

def download_grib(cfg: HindcastConfig, month_str: str, out_file: Path) -> None:
    print(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # REQUEST alinhado 100% com o formulário do CDS
    request = {
        "originating_centre": cfg.originating_centre,   # deve ser "jma"
        "system": cfg.system,                           # "3"
        "variable": cfg.variable,                       # ["total_precipitation"]
        "product_type": cfg.product_type,               # será sobrescrito para JMA
        "year": cfg.years,
        "month": [month_str],
        "leadtime_month": cfg.leadtime_month,
        "data_format": cfg.data_format,
        "area": cfg.area,
    }

    prefix = str(getattr(cfg, "model_prefix", "")).strip().lower()
    centre = str(getattr(cfg, "originating_centre", "")).strip().lower()
    system = str(getattr(cfg, "system", "")).strip()

    print(f"\n=== DOWNLOAD month={month_str} ===")
    print(f"Output: {out_file}")
    print("DEBUG centre/system/prefix:", centre, system, prefix)

    # JMA sys3 (CDS usa "jma", GRIB usa "rjtd")
    is_jma_sys3 = (
        centre in ("jma", "rjtd") and system == "3"
    ) or (
        prefix.startswith("jma_sys3")
    )

    if is_jma_sys3:
        # CDS vocabulary (igual ao portal)
        request["originating_centre"] = "jma"

        # EXATAMENTE o mesmo produto do GRIB manual que você mostrou
        # dataType=hcmean / type=hcmean
        request["product_type"] = ["monthly_mean"]

    # DWD subseas_glo sys2 (centre pode vir como "dwd" ou "edzw")
    is_dwd_sys2 = (
        centre in ("dwd", "edzw") and system == "2"
    ) or prefix.startswith("dwd_sys2")

    if is_dwd_sys2:
        # Forçar o produto hindcast (hcmean) e horário fixo para evitar MarsNoData
        request["originating_centre"] = "dwd"
        request["time"] = ["00:00"]
        # hcmean no GRIB == hindcast_climate_mean no CDS
        request["product_type"] = ["monthly_mean"]
        # se quiser também garantir originating centre do lado do CDS:
        # request["originating_centre"] = "dwd"

    is_cmcc_sys35 = (
        centre in ("cmcc", "cnmc") and system == "35"
    ) or prefix.startswith("cmcc_sys35")

    if is_cmcc_sys35:
        request["originating_centre"] = "cmcc"   # vocabulário CDS
        request["product_type"] = ["monthly_mean"]
        request["time"] = ["00:00"]

    is_eccc_sys4 = (
        centre in ("eccc", "cwao") and system == "4"
    ) or prefix.startswith("eccc_sys4")

    if is_eccc_sys4:
        request["originating_centre"] = "eccc"
        request["product_type"] = ["monthly_mean"]
        request["time"] = ["00:00"]

    is_ncep_sys2 = (
        centre in ("ncep", "kwbc") and system == "2"
    ) or prefix.startswith("ncep_sys2")

    if is_ncep_sys2:
        request["originating_centre"] = "ncep"
        request["time"] = ["00:00"]
        request["product_type"] = ["monthly_mean"]
    
    print("REQUEST FINAL:", request)

    c = cdsapi.Client(url=API, key=KEY, quiet=True)
    c.retrieve(cfg.dataset, request, str(out_file))

    print("[OK] Download completed")
