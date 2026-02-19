# CFS Temperature Daily Bias-Correction Pipeline (XESMF)

Pipeline em Python para **corrigir previsões de temperatura do CFS** em **resolução diária**, aplicando **correção aditiva (LS-Add)** baseada em **hindcast mensal** e **climatologia observada diária**, com **regradeamento bilinear (XESMF)** para uma grade de referência e **clipping absoluto** em torno da climatologia diária.

time python3 CFS_TEMPERATURE-HYBRID.PY   --forecast /home/felipe/ajuste/temperatura/2025/CFS_2025/2m_air_temperature_max/2025/335   --hindcast-root /home/felipe/ajuste/reforecast/cfs_glo   --clim-obs /home/felipe/ajuste/temperatura/climatologia   --ref-grid /home/felipe/ajuste/temperatura/climatologia/era5_glo_2m_air_temperature_max_20010101.nc   --out-hindcast /home/felipe/ajuste/temperatura/dado_temp/PASTA_HINDCAST_TEST1   --out-corr /home/felipe/ajuste/temperatura/dado_temp/PASTA_CORR_TEST1   --netcdf-engine netcdf4   --debug 2>&1 | tee run_test1_temp.log

## O que este script faz

1. **Auto-detecta**:
   - **DOY** (001–366) a partir do caminho do forecast
   - **Ano** a partir do caminho do forecast (ou via `--year`)
   - **Variável** de temperatura a partir do caminho (pasta)
2. **Processa o hindcast** (mensal por lead):
   - Lê arquivos do hindcast para o DOY correspondente
   - Seleciona o tempo do hindcast (`00z`, `first` ou `mean`)
   - Concatena por `lead` e cria coordenada `month` por lead
   - Regrida para a grade final via **XESMF bilinear**
   - Salva como NetCDF
3. **Corrige o forecast**:
   - Lê o forecast (subdiário) e agrega em **média diária**
   - Regrida para a grade de referência
   - Limita a **280 dias** (`max_days_output`)
   - Calcula o lead mensal para cada dia (por diferença de mês desde o init)
   - Aplica correção aditiva:
     - `corr = forecast + (obs_month_mean - hindcast_month)`
   - **Extend months**: permite **1 mês extra** além do hindcast (repete o último hindcast e usa média observada do mês real)
4. **Clipping absoluto (opcional)**:
   - Garante que `corr` fique em:
     - `[clim_daily - Δ, clim_daily + Δ]` por ponto de grade e por dia
   - `Δ` default: **1.5°C** (configurável)

## Variáveis suportadas

O script identifica a variável pela pasta no caminho do forecast:

- `2m_air_temperature`
- `2m_air_temperature_min`
- `2m_air_temperature_max`

> Se o NetCDF não tiver exatamente esse nome de variável, o script tenta aliases (ex.: `t2m`, `air_temperature`, etc.) ou usa a única variável presente no arquivo.

## Requisitos

- Python 3.8+
- Bibliotecas:
  - `numpy`
  - `xarray`
  - `xesmf`
  - Engine NetCDF (recomendado: `netcdf4`)

Exemplo de instalação (ajuste conforme seu ambiente):

```bash
pip install numpy xarray xesmf netcdf4
```

> **Nota:** `xesmf` pode exigir dependências como ESMF/ESMPy dependendo do seu setup.

## Entradas esperadas

### Forecast (`--forecast`)
- Pode ser:
  - Uma **pasta DOY** contendo arquivos `*_M000_YYYYMMDD00.nc` (o script pega o mais recente), **ou**
  - Um **arquivo .nc** específico

O script procura por padrão:
- `*_M000_*00.nc`

### Hindcast root (`--hindcast-root`)
Root pai onde o script vai procurar:
- `<hindcast-root>/<var_name>/<YYYY>/<DOY>/*.nc`

O script encontra automaticamente a pasta `<DOY>` buscando primeiro anos mais recentes.

### Climatologia observada diária (`--clim-obs`)
- Pode ser **arquivo .nc** ou **pasta** com `.nc`
- Precisa ter dimensão `time` e permitir obter `dayofyear` via `time.dt.dayofyear`
- O script escolhe automaticamente um arquivo compatível dentro da pasta (prioriza nomes com `daily`)

### Grade de referência (`--ref-grid`)
- NetCDF contendo `latitude` e `longitude` (ou `lat/lon`, que serão renomeados)
- Recomenda-se usar a mesma grade da climatologia observada

## Saídas

### Hindcast processado
Salvo em:
- `<out-hindcast>/<var_name>/<hindcast_year>/<DOY>/cfs_hindcast_<var_name>_doy<DOY>.nc`

Também salva (e reutiliza) pesos do XESMF em:
- `<out-hindcast>/<var_name>/<hindcast_year>/<DOY>/regrid_weights.nc`

### Forecast corrigido
Salvo em:
- `<out-corr>/<var_name>/<year>/<DOY>/cfs_glo_<var_name>_M100_<YYYYMMDDHH>.nc`

## Uso

```bash
python cfs_temp_daily_pipeline.py \
  --forecast /caminho/para/forecast/2m_air_temperature/2026/045 \
  --hindcast-root /data3/reforecast/cfs_glo \
  --clim-obs /caminho/para/climatologia_obs_daily.nc \
  --ref-grid /caminho/para/ref_grid.nc \
  --out-hindcast /saida/hindcast_proc \
  --out-corr /saida/forecast_corr \
  --year 2026 \
  --hindcast-time-mode 00z \
  --hindcast-leads-expected 9 \
  --zlib \
  --debug
```

### Desativar clipping
```bash
python cfs_temp_daily_pipeline.py ... --no-clip
```

### Ajustar delta do clipping (°C)
```bash
python cfs_temp_daily_pipeline.py ... --clip-delta-c 2.0
```

## Parâmetros (CLI)

- `--forecast` (obrigatório): pasta DOY do forecast ou arquivo `.nc`
- `--hindcast-root` (obrigatório): root do reforecast/hindcast
- `--clim-obs` (obrigatório): climatologia observada diária (arquivo ou pasta)
- `--ref-grid` (obrigatório): arquivo com grade final (`latitude/longitude`)
- `--out-hindcast` (obrigatório): base de saída do hindcast processado
- `--out-corr` (obrigatório): base de saída do forecast corrigido
- `--year` (opcional): ano para saída (se não inferir do caminho)
- `--hindcast-time-mode`: `00z` (default), `first`, `mean`
- `--hindcast-leads-expected`: número de leads (default `9`)
- `--netcdf-engine`: default `netcdf4`
- `--zlib`: ativa compressão
- `--no-clip`: desativa clipping
- `--clip-delta-c`: delta absoluto em °C (default `1.5`)
- `--debug`: logs verbosos

## Observações importantes

- Coordenadas:
  - O script normaliza `lat/lon` → `latitude/longitude`
  - Ordena `latitude` e `longitude`
  - Ajusta longitude `0..360` → `-180..180` se necessário (quando a grade alvo tem longitudes negativas)
- O forecast é convertido para diário via:
  - `resample(time="1D").mean()`
- O output diário é limitado a **280 dias** por padrão.

## Licença

Defina a licença do projeto aqui (ex.: MIT, Apache-2.0) ou remova esta seção.
