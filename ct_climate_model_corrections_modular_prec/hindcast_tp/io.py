from __future__ import annotations
from pathlib import Path
import xarray as xr
from .config import HindcastConfig

def open_grib(path: Path, cfg: HindcastConfig | None = None) -> xr.Dataset:
    backend_kwargs = {}

    if cfg is not None:
        centre = str(getattr(cfg, "originating_centre", "")).lower()
        system = str(getattr(cfg, "system", ""))
        prefix = str(getattr(cfg, "model_prefix", "")).lower()

        # UKMO sys603
        if centre in ("egrr", "ukmo", "uk-met-office", "metoffice", "ukmetoffice") and system == "603":
            backend_kwargs["filter_by_keys"] = {
                "centre": "egrr",
                "localDefinitionNumber": 12,
                "dataType": "fcmean",
                "typeOfLevel": "surface",
            }
            backend_kwargs["indexpath"] = ""

        # JMA subseas glo (hindcast hcmean)
        if centre == "rjtd" and system == "3":
            backend_kwargs["filter_by_keys"] = {
                "centre": "rjtd",
                "localDefinitionNumber": 16,
                "typeOfLevel": "surface",
            }
            backend_kwargs["indexpath"] = ""
            
        # DWD subseas glo (centre=edzw, system=2): GRIB vem com hcmean + fcmean -> precisa filtrar
        if centre in ("edzw", "dwd") and str(system) == "2":
            backend_kwargs["filter_by_keys"] = {
                "centre": "edzw",
                "localDefinitionNumber": 16,
                "typeOfLevel": "surface",
            }
            backend_kwargs["indexpath"] = ""

        # CMCC sys35 (centre=cnmc): GRIB vem com hcmean + fcmean -> precisa filtrar
        if centre in ("cnmc", "cmcc") and str(system) == "35":
            backend_kwargs["filter_by_keys"] = {
                "centre": "cnmc",
                "dataType": "fcmean",             # pega o hindcast por membros (o grosso do arquivo)
                "typeOfLevel": "surface",
                # "localDefinitionNumber": 16,     # só use se precisar (se continuar ambíguo)
            }
            backend_kwargs["indexpath"] = ""

        # ECCC sys4 (centre=cwao): GRIB vem com hcmean + fcmean -> precisa filtrar
        if centre in ("cwao", "eccc") and str(system) == "4":
            backend_kwargs["filter_by_keys"] = {
                "centre": "cwao",
                "dataType": "fcmean",   # usar membros do hindcast
                "typeOfLevel": "surface",
            }
            backend_kwargs["indexpath"] = ""

        # NCEP (centre=kwbc, system=2): GRIB vem com hcmean + fcmean -> filtrar
        if centre in ("kwbc", "ncep") and str(system) == "2":
            backend_kwargs["filter_by_keys"] = {
                "centre": "kwbc",
                "dataType": "fcmean",
                "type": "fcmean",
                "typeOfLevel": "surface",
            }
            backend_kwargs["indexpath"] = ""

    return xr.open_dataset(path, engine="cfgrib", backend_kwargs=backend_kwargs)
 
 
def normalize_coords_latlon(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ds = ds.rename({"lon": "longitude"})
    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise RuntimeError(f"Dataset is missing latitude/longitude coordinates. Coords: {list(ds.coords)}")
    latv = ds["latitude"].values
    if latv.size > 1 and latv[0] > latv[-1]:
        ds = ds.sortby("latitude")
    return ds

def load_reference_grid(path: Path) -> xr.Dataset:
    ds_ref = xr.open_dataset(path)
    if "lat" in ds_ref.coords and "latitude" not in ds_ref.coords:
        ds_ref = ds_ref.rename({"lat": "latitude"})
    if "lon" in ds_ref.coords and "longitude" not in ds_ref.coords:
        ds_ref = ds_ref.rename({"lon": "longitude"})
    if "latitude" not in ds_ref.coords or "longitude" not in ds_ref.coords:
        raise ValueError("Reference grid file must have latitude/longitude coordinates (or lat/lon).")
    return xr.Dataset({"lat": ds_ref["latitude"], "lon": ds_ref["longitude"]})
