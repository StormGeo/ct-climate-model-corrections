from __future__ import annotations

import numpy as np
import xarray as xr

from .base import BaseProcessor
from .time_step_like import _pick_temp_var
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


class JMASys3Processor(BaseProcessor):
    """
    JMA sys3 (centre rjtd)
    Hindcast monthly temperature statistic (t2m), layout time/step with valid_time.

    Produz:
      <out_var_name>(lead, lat, lon) em °C
      coords: lead=1..6, calendar_month, month
    """

    name = "rjtd_sys3"

    def can_handle(self, ds: xr.Dataset, cfg: HindcastConfig) -> bool:
        # Mantém compatível com como você roda:
        #  - precip: --model-prefix jma_sys3
        #  - t2m:    --model-prefix jma_subseas_glo (se você quiser)
        prefix = str(getattr(cfg, "model_prefix", "")).lower()
        if not (prefix.startswith("jma_sys3") or prefix.startswith("jma_subseas_glo")):
            return False

        centre_cfg = str(getattr(cfg, "originating_centre", "")).lower().strip().replace("_", "-")
        centre_ds = (ds.attrs.get("GRIB_centre") or "").lower()

        if centre_cfg in ("rjtd", "jma"):
            return True
        if centre_ds == "rjtd":
            return True
        return False

    def set_target_month(self, m: int) -> None:
        self._target_month = int(m)

    def process_grib(self, ds: xr.Dataset, cfg: HindcastConfig) -> xr.Dataset:
        ds = normalize_coords_latlon(ds)

        vname = _pick_temp_var(ds, cfg.input_var)

        for dim in ("time", "step", "latitude", "longitude"):
            if dim not in ds.dims:
                raise RuntimeError(f"JMA expected dim '{dim}'. Got dims={ds.dims}")

        if "valid_time" not in ds:
            raise RuntimeError("JMA expected 'valid_time' coordinate in GRIB.")

        # Kelvin -> Celsius (lazy-friendly)
        temp_c = (ds[vname] - 273.15).astype("float32", copy=False)
        vt = ds["valid_time"]

        # === MESMA LÓGICA DO TP: lead baseado no mês do valid_time relativo ao mês alvo ===
        tgt_month = int(getattr(self, "_target_month", -1))
        if tgt_month < 1 or tgt_month > 12:
            raise RuntimeError("JMA sys3: target_month não definido. Defina no pipeline antes de processar.")

        vt_month = vt.dt.month.values  # (time, step)
        lead_norm = ((vt_month - tgt_month) % 12) + 1  # (time, step) em 1..12
        keep = (lead_norm >= 1) & (lead_norm <= 6)
        if not np.any(keep):
            raise RuntimeError("JMA sys3: nenhum lead 1..6 encontrado via valid_time/month.")

        target_ym = _ym_int(_to_datetime64_ns(vt.values))  # (time, step) int YM (p/ cal_month)

        lat = ds["latitude"].values
        lon = ds["longitude"].values

        out_sum = np.zeros((6, lat.size, lon.size), dtype=np.float64)
        out_cnt = np.zeros(6, dtype=np.int64)
        cal_month = np.full(6, -1, dtype=np.int64)

        # Para cada (time, lead), escolhe o PRIMEIRO step válido daquele lead com dados finitos.
        for ti in range(ds.sizes["time"]):
            for l in range(1, 7):
                cand = np.where((lead_norm[ti, :] == l) & keep[ti, :])[0]
                if cand.size == 0:
                    continue

                chosen = None
                for si in cand:
                    # leitura mínima para testar se tem dados (evita carregar tudo)
                    sample = temp_c.isel(time=ti, step=int(si), latitude=slice(0, 2), longitude=slice(0, 2)).values
                    # se vier com ensemble: (number/member, y, x)
                    if sample.ndim == 3:
                        sample = sample[0]
                    if np.isfinite(sample).any():
                        chosen = int(si)
                        break

                # fallback: tenta o último candidato
                if chosen is None:
                    chosen = int(cand[-1])
                    sample = temp_c.isel(time=ti, step=chosen, latitude=slice(0, 2), longitude=slice(0, 2)).values
                    if sample.ndim == 3:
                        sample = sample[0]
                    if not np.isfinite(sample).any():
                        continue

                ym = np.array(target_ym[ti, chosen], dtype="int64").astype("datetime64[M]")

                vals = np.asarray(temp_c.isel(time=ti, step=chosen).values)
                if not np.isfinite(vals).any():
                    continue

                # Ensemble: (number/member, lat, lon) -> média
                if vals.ndim == 3:
                    vals = np.nanmean(vals, axis=0)
                elif vals.ndim != 2:
                    raise ValueError(f"JMA sys3: unexpected vals.ndim={vals.ndim}, shape={vals.shape}")

                out_sum[l - 1] += vals.astype(np.float64)
                out_cnt[l - 1] += 1

                if cal_month[l - 1] < 0:
                    cal_month[l - 1] = int(str(ym).split("-")[1])

        out = np.full_like(out_sum, np.nan, dtype=np.float64)
        for i in range(6):
            if out_cnt[i] > 0:
                out[i] = out_sum[i] / float(out_cnt[i])

        da_out = xr.DataArray(
            out,
            dims=("lead", "latitude", "longitude"),
            coords={"lead": np.arange(1, 7, dtype=np.int64), "latitude": lat, "longitude": lon},
            name=str(cfg.out_var_name),
            attrs={
                "units": "degC",
                "long_name": "2m air temperature (monthly statistic)",
                "source": "JMA sys3 (centre=rjtd) monthly statistic converted to degC; lead by valid_time month (TP-like).",
            },
        )

        ds_out = xr.Dataset({str(cfg.out_var_name): da_out})
        ds_out = ds_out.assign_coords(month=("lead", cal_month))
        ds_out = ds_out.assign_coords(calendar_month=("lead", cal_month))
        ds_out[str(cfg.out_var_name)] = ds_out[str(cfg.out_var_name)].transpose("lead", "latitude", "longitude")
        return ds_out