# ct-correction-cfs-ecmwf

Pipeline para **processamento de hindcast do ECMWF System 51** e **correção de viés (LS-Add)** de previsões subseasonais de precipitação, com saída mensal acumulada em **mm**.

O repositório contém dois scripts principais:
1. Download e processamento do **hindcast**
2. Pré-processamento e **correção de forecasts** usando climatologia observada

---

## Visão geral do fluxo

### 1. Hindcast (ECMWF System 51)
- Download via CDS (`seasonal-monthly-single-levels`)
- Conversão de `tprate (m/s)` → precipitação mensal acumulada (mm)
- Leads 1–6
- Cálculo da climatologia do hindcast (média em `member` e `init_time`)
- Regridding opcional para uma grade de referência
- Saída em NetCDF

### 2. Forecast
- Leitura de forecasts brutos (NetCDF)
- Remoção do último mês incompleto
- Acumulação mensal (`resample(time="MS").sum()`)
- Ajuste de convenção de longitude (0–360 ↔ -180–180)
- Regridding para a grade de referência
- Correção de viés **LS-Add**
- Valores negativos são truncados para zero

---

## Estrutura de saída (regra obrigatória)

Após `--out-*`, **os scripts só podem criar**:


Nenhuma pasta extra é criada.

---

## Requisitos

Python 3.9+ recomendado.

Dependências principais:
- numpy
- xarray

Para hindcast:
- cdsapi
- cfgrib
- ecCodes (instalado no sistema)

Regridding opcional:
- xesmf

Exemplo:
```bash
pip install numpy xarray cdsapi cfgrib xesmf

hindcast_tp_download_process.py
python3 hindcast_tp_download_process.py \
  --month 12 \
  --out-grib /path/OUT_GRIB \
  --out-nc   /path/OUT_NC

python3 hindcast_tp_download_process.py \
  --doy-root /path/<YEAR>/ \
  --out-grib /path/OUT_GRIB \
  --out-nc   /path/OUT_NC

python3 hindcast_tp_download_process.py \
  --month 12 \
  --out-grib /path/OUT_GRIB \
  --out-nc   /path/OUT_NC \
  --regrid \
  --ref-grid /path/grid.nc

forecast_correction_pipeline.py

corrected = forecast + (clim_obs(month_of_lead) - clim_hindcast(lead))

python3 forecast_correction_pipeline.py \
  --forecast-root /path/RAW_FORECASTS/<YEAR>/ \
  --hindcast-root /path/HINDCAST_NC/total_precipitation/<YEAR>/ \
  --clim-file     /path/OBS_CLIM_AND_GRID.nc \
  --out-root      /path/OUT_CORRECTED \
  --var-name total_precipitation


---

### Como subir
No GitHub:
1. Clique em **Add a README**
2. Cole tudo acima
3. **Commit changes**

Se quiser, posso:
- Ajustar o README para **padrão StormGeo**
- Simplificar para uso operacional
- Escrever uma versão curta + técnica


