from __future__ import annotations

from pathlib import Path

import xarray as xr

from .config import HindcastConfig


def _norm_centre(value: str) -> str:
    return str(value).lower().strip().replace("_", "-")


def open_grib(path: Path, cfg: HindcastConfig | None = None) -> xr.Dataset:
    backend_kwargs: dict = {}
    centre = ""
    system = ""

    if cfg is not None:
        centre = _norm_centre(getattr(cfg, "originating_centre", ""))
        system = str(getattr(cfg, "system", ""))

        # UKMO sys603
        if centre in ("egrr", "ukmo", "uk-met-office", "metoffice", "ukmetoffice") and system == "603":
            backend_kwargs["indexpath"] = ""

        # JMA subseas glo
        elif centre in ("rjtd", "jma") and system == "3":
            backend_kwargs["filter_by_keys"] = {"centre": "rjtd"}
            backend_kwargs["indexpath"] = ""

        # DWD sys2
        elif centre in ("edzw", "dwd") and system == "2":
            backend_kwargs["filter_by_keys"] = {
                "centre": "edzw",
                "localDefinitionNumber": 16,
                "typeOfLevel": "surface",
            }
            backend_kwargs["indexpath"] = ""

        # CMCC sys35
        elif centre in ("cnmc", "cmcc") and system == "35":
            backend_kwargs["indexpath"] = ""

        # ECCC sys4
        elif centre in ("cwao", "eccc") and system == "4":
            backend_kwargs["indexpath"] = ""

        # NCEP sys2
        elif centre in ("kwbc", "ncep") and system == "2":
            backend_kwargs["indexpath"] = ""

    # NCEP: abrir sem chunks, igual ao módulo de precipitação
    if centre in ("kwbc", "ncep") and system == "2":
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs=backend_kwargs,
        )
    else:
        # Demais modelos: mantém chunks como já está funcionando
        ds = xr.open_dataset(
            path,
            engine="cfgrib",
            backend_kwargs=backend_kwargs,
            chunks={"number": 1, "time": 1, "step": 1},
        )

    try:
        print("OPEN_GRIB -> vars:", list(ds.data_vars), "coords:", list(ds.coords), "dims:", dict(ds.sizes))
    except Exception:
        print("OPEN_GRIB -> opened dataset")

    return ds


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