from __future__ import annotations

import numpy as np
import xarray as xr

from .base import BaseProcessor
from .time_step_like import seconds_in_month_from_ym
from ..config import HindcastConfig
from ..io import normalize_coords_latlon


def _to_datetime64_ns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]")
    epoch = np.datetime64("1970-01-01T00:00:00", "ns")
    sec = arr.astype("int64")
    return epoch + sec.astype("timedelta64[s]").astype("timedelta64[ns]")


def _ym_int(dt64_ns: np.ndarray) -> np.ndarray:
    return dt64_ns.astype("datetime64[M]").astype("int64")


class CMCCSys35Processor(BaseProcessor):
    """
    CMCC (centre=cnmc, system=35)
    Hindcast monthly mean precipitation rate (tprate), layout time/step with valid_time.

    Produz:
      total_precipitation(lead, lat, lon) em mm/mês
      coords: lead=1..6, calendar_month, month
    """

    name = "cmcc_sys35"

    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        prefix = str(getattr(cfg, "model_prefix", "")).lower()
        if not prefix.startswith("cmcc_sys35"):
            return False

        centre_cfg = str(getattr(cfg, "originating_centre", "")).lower()
        sys_cfg = str(getattr(cfg, "system", "")).strip()

        centre_ds = (ds.attrs.get("GRIB_centre") or "").lower()
        sys_ds = str(ds.attrs.get("GRIB_system") or "").strip()

        # aceita vocabulário CDS ("cmcc") e GRIB ("cnmc")
        if sys_cfg == "35" and centre_cfg in ("cmcc", "cnmc"):
            return True
        if sys_ds == "35" and centre_ds == "cnmc":
            return True
        return False

    def set_target_month(self, m: int) -> None:
        self._target_month = int(m)

    def process_grib(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        ds = normalize_coords_latlon(ds)

        if cfg.input_var not in ds:
            raise RuntimeError(f"Variable '{cfg.input_var}' not found. Found: {list(ds.data_vars)}")

        for dim in ("time", "step", "latitude", "longitude"):
            if dim not in ds.dims:
                raise RuntimeError(f"CMCC expected dim '{dim}'. Got dims={ds.dims}")

        if "valid_time" not in ds:
            raise RuntimeError("CMCC expected 'valid_time' coordinate in GRIB.")

        rate = ds[cfg.input_var]  # m s-1
        vt = ds["valid_time"]

        # lead baseado no mês do valid_time relativo ao mês alvo (definido no pipeline)
        tgt_month = int(getattr(self, "_target_month", -1))
        if tgt_month < 1 or tgt_month > 12:
            raise RuntimeError("CMCC sys35: target_month não definido. Defina no pipeline antes de processar.")

        # CMCC: validityDate já vem no 1º dia do mês (não precisa +1 dia como no DWD)
        vt_month = vt.dt.month.values                      # (time, step)
        target_ym = _ym_int(_to_datetime64_ns(vt.values))   # (time, step)

        # lead 1..12 relativo ao mês alvo, depois filtramos 1..6
        lead_norm = ((vt_month - tgt_month) % 12) + 1
        keep = (lead_norm >= 1) & (lead_norm <= 6)
        if not np.any(keep):
            raise RuntimeError("CMCC sys35: nenhum lead 1..6 encontrado via valid_time/month.")

        lat = ds["latitude"].values
        lon = ds["longitude"].values

        out_sum = np.zeros((6, lat.size, lon.size), dtype=np.float64)
        out_cnt = np.zeros(6, dtype=np.int64)
        cal_month = np.full(6, -1, dtype=np.int64)

        for ti in range(ds.sizes["time"]):
            for l in range(1, 7):
                cand = np.where((lead_norm[ti, :] == l) & keep[ti, :])[0]
                if cand.size == 0:
                    continue

                chosen = None
                vals2 = None

                # Escolhe o primeiro step daquele lead que tem dado (após reduzir membros)
                for si in cand:
                    vals = np.asarray(rate.isel(time=ti, step=int(si)).values)

                    # CMCC pode vir como (number, lat, lon) ou (lat, lon)
                    if vals.ndim == 3:
                        if not np.isfinite(vals).any():
                            continue
                        vals_tmp = np.nanmean(vals, axis=0)
                    elif vals.ndim == 2:
                        vals_tmp = vals
                    else:
                        raise ValueError(f"CMCC sys35: unexpected vals.ndim={vals.ndim}, shape={vals.shape}")

                    if np.isfinite(vals_tmp).any():
                        chosen = int(si)
                        vals2 = vals_tmp
                        break

                if chosen is None:
                    continue

                ym = np.array(target_ym[ti, chosen], dtype="int64").astype("datetime64[M]")
                sec = seconds_in_month_from_ym(ym)

                out_sum[l - 1] += vals2.astype(np.float64) * float(sec) * 1000.0
                out_cnt[l - 1] += 1

                if cal_month[l - 1] < 0:
                    # Força mês-alvo por lead (tgt_month + (lead-1)) em 1..12
                    cal_month[l - 1] = ((tgt_month - 1 + (l - 1)) % 12) + 1

        out_arr = np.full_like(out_sum, np.nan, dtype=np.float64)
        for i in range(6):
            if out_cnt[i] > 0:
                out_arr[i] = out_sum[i] / float(out_cnt[i])

        da_out = xr.DataArray(
            out_arr,
            dims=("lead", "latitude", "longitude"),
            coords={
                "lead": np.arange(1, 7, dtype=np.int64),
                "latitude": lat,
                "longitude": lon,
            },
            name="total_precipitation",
            attrs={
                "units": "mm",
                "long_name": "Total precipitation (monthly accumulated)",
                "source": "CMCC sys35 (centre=cnmc) from tprate (m s-1), seconds in target month.",
            },
        )

        ds_out = xr.Dataset({"total_precipitation": da_out})
        ds_out = ds_out.assign_coords(month=("lead", cal_month))
        ds_out = ds_out.assign_coords(calendar_month=("lead", cal_month))
        ds_out["total_precipitation"] = ds_out["total_precipitation"].transpose("lead", "latitude", "longitude")
        return ds_out
