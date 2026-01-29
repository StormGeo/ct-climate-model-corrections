import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

# =========================
# PADRÕES (empresa)
# =========================
PRECIP_TARGET_NAME = "total_precipitation"
TEMP_BASE_NAME = "2m_air_temperature"  # base, saída vira _min/_med/_max

GLOBAL_ATTRS = {
    "institution": "Climatempo - MetOps", 
    "source": "Hydra",
    "title": "climatempo - MetOps Netcdf file | from Hydra",
}

VALID_NETCDF4_ENCODINGS = {
    "zlib", "endian", "szip_pixels_per_block", "blosc_shuffle", "chunksizes",
    "least_significant_digit", "complevel", "quantize_mode", "contiguous",
    "shuffle", "szip_coding", "dtype", "_FillValue", "fletcher32",
    "compression", "significant_digits"
}

def filtrar_encoding_netcdf4(enc: dict) -> dict:
    if not isinstance(enc, dict):
        return {}
    return {k: v for k, v in enc.items() if k in VALID_NETCDF4_ENCODINGS}

# =========================
# DETECÇÃO min/med/max PELO CAMINHO
# =========================
def identificar_temp_tag_pelo_caminho(path: Path) -> str:
    s = str(path).lower()
    if any(k in s for k in ["_min", "/min", "tmin", "minimum", "minima"]):
        return "min"
    if any(k in s for k in ["_max", "/max", "tmax", "maximum", "maxima"]):
        return "max"
    if any(k in s for k in ["_med", "/med", "mean", "avg", "media", "média"]):
        return "med"
    return "med"

# =========================
# DETECÇÃO de variável
# =========================
def escolher_variavel_principal(ds: xr.Dataset) -> str:
    # Prioriza precip, depois temp base, depois primeira
    if PRECIP_TARGET_NAME in ds.data_vars:
        return PRECIP_TARGET_NAME
    if TEMP_BASE_NAME in ds.data_vars:
        return TEMP_BASE_NAME
    return list(ds.data_vars.keys())[0]

def tipo_por_variavel(vname: str) -> str:
    if vname == PRECIP_TARGET_NAME:
        return "precip"
    if vname == TEMP_BASE_NAME:
        return "temp"
    vn = vname.lower()
    if "precip" in vn or "rain" in vn or vn == "tp":
        return "precip"
    if "temp" in vn or "temperature" in vn or "t2m" in vn or "2m" in vn:
        return "temp"
    return "other"

# =========================
# PADRONIZA coords
# =========================
def padronizar_coords(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds.coords and "latitude" not in ds.coords:
        ds = ds.rename({"lat": "latitude"})
    if "lon" in ds.coords and "longitude" not in ds.coords:
        ds = ds.rename({"lon": "longitude"})

    if "latitude" in ds.coords:
        ds["latitude"].attrs.setdefault("standard_name", "latitude")
        ds["latitude"].attrs.setdefault("long_name", "latitude")
        ds["latitude"].attrs.setdefault("units", "degrees_north")
        ds["latitude"].attrs.setdefault("grads_dim", "Y")

    if "longitude" in ds.coords:
        ds["longitude"].attrs.setdefault("standard_name", "longitude")
        ds["longitude"].attrs.setdefault("long_name", "longitude")
        ds["longitude"].attrs.setdefault("units", "degrees_east")
        ds["longitude"].attrs.setdefault("grads_dim", "x")

    return ds

# =========================
# I/O util
# =========================
def listar_arquivos(in_path: Path):
    if in_path.is_file() and in_path.suffix.lower() == ".nc":
        return [in_path]
    if in_path.is_dir():
        return sorted(in_path.rglob("*.nc"))
    raise FileNotFoundError(f"Entrada inválida: {in_path}")

# =========================
# PROCESSAMENTO
# =========================
def processar_arquivo(in_nc: Path, out_path: Path):
    ds = xr.open_dataset(in_nc)
    ds = padronizar_coords(ds)

    if "time" not in ds.dims and "time" not in ds.coords:
        raise ValueError(f"Arquivo sem dimensão/coord 'time': {in_nc}")

    ds["time"] = pd.to_datetime(ds["time"].values)

    v_in = escolher_variavel_principal(ds)
    da_in = ds[v_in]
    vtipo = tipo_por_variavel(v_in)

    # agregação + renome
    if vtipo == "precip":
        v_out = PRECIP_TARGET_NAME
        da_m = da_in.resample(time="MS").sum()
        da_m.attrs = dict(da_in.attrs)
    elif vtipo == "temp":
        tag = identificar_temp_tag_pelo_caminho(in_nc)
        v_out = f"{TEMP_BASE_NAME}_{tag}"
        da_m = da_in.resample(time="MS").mean()
        da_m.attrs = dict(da_in.attrs)
        if not str(da_m.attrs.get("units", "")).strip():
            da_m.attrs["units"] = "C"
    else:
        v_out = v_in
        da_m = da_in.resample(time="MS").sum()
        da_m.attrs = dict(da_in.attrs)

    da_m = da_m.rename(v_out)

    ds_out = da_m.to_dataset()
    ds_out = padronizar_coords(ds_out)

    # attrs globais padrão
    ds_out.attrs.update(GLOBAL_ATTRS)
    now_tag = datetime.utcnow().strftime("%Y%m%d%H")
    ds_out.attrs.setdefault("description", f"netcdf file created by Hydra in {now_tag}")
    ds_out.attrs.setdefault("history", f"Created in {now_tag}")

    # time attrs (NÃO colocar calendar/units aqui para não conflitar com encoding)
    ds_out["time"].attrs.setdefault("standard_name", "time")
    ds_out["time"].attrs.setdefault("axis", "T")

    # encoding: filtra e garante defaults para a variável
    enc_in = {}
    try:
        enc_in = dict(getattr(da_in, "encoding", {}))
    except Exception:
        enc_in = {}

    encoding = {v_out: filtrar_encoding_netcdf4(enc_in)}
    encoding[v_out].setdefault("_FillValue", np.float32(-9.99e8))
    encoding[v_out].setdefault("least_significant_digit", 2)
    encoding[v_out].setdefault("dtype", "float32")

    # time encoding (hours since t0) — calendar/units ficam SÓ aqui
    t0 = pd.to_datetime(ds_out["time"].values[0]).to_pydatetime()
    ds_out["time"].encoding["units"] = f"hours since {t0:%Y-%m-%d %H:%M:%S}"
    ds_out["time"].encoding["calendar"] = "standard"
    ds_out["time"].encoding["dtype"] = "float32"

    # remover attrs conflitantes do time (se vierem do arquivo original)
    for k in ["calendar", "units", "dtype"]:
        ds_out["time"].attrs.pop(k, None)

    # -------------------------
    # SAÍDA: usuário controla o path
    # -------------------------
    # Se --out terminar com .nc => é o arquivo final.
    # Caso contrário => é diretório e salva como <var>.nc dentro dele.
    if out_path.suffix.lower() == ".nc":
        out_nc = out_path
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        out_nc = out_path / f"{v_out}.nc"

    out_nc.parent.mkdir(parents=True, exist_ok=True)

    ds_out.to_netcdf(out_nc, unlimited_dims=["time"], encoding=encoding)

    return out_nc

# =========================
# MAIN
# =========================
def main():
    p = argparse.ArgumentParser(
        description="Agrega mensal (precip=soma; temp=média) e padroniza nomes/estrutura NetCDF."
    )
    p.add_argument("--in", dest="in_path", default=".", help="Arquivo .nc ou diretório (default: .)")
    p.add_argument(
        "--out",
        dest="out_path",
        default="./out",
        help="Path de saída (arquivo .nc final OU diretório onde salvar)",
    )
    args = p.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()

    arquivos = listar_arquivos(in_path)
    print(f"Total de arquivos encontrados: {len(arquivos)}")

    # Se for processar múltiplos arquivos, --out precisa ser diretório
    if len(arquivos) > 1 and out_path.suffix.lower() == ".nc":
        raise ValueError(
            "Você passou um arquivo .nc em --out, mas a entrada tem múltiplos .nc. "
            "Passe um diretório em --out para salvar um por arquivo."
        )

    for f in arquivos:
        out_nc = processar_arquivo(f, out_path)
        print(f"[OK] {f} -> {out_nc}")

if __name__ == "__main__":
    main()
