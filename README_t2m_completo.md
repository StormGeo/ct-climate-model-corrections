
# ct_climate_model_corrections_modular_t2m

Pipeline modular para **temperatura do ar a 2 metros (`t2m`)** que cobre duas etapas distintas do processamento climatológico:

1. **Geração do hindcast climatológico mensal de temperatura** a partir do CDS/C3S seasonal forecast database.
2. **Correção diária do forecast operacional bruto** usando a combinação de:
   - climatologia observada diária,
   - hindcast mensal processado por lead,
   - redistribuição intramensal do sinal climatológico.

O projeto foi desenvolvido para resolver um problema operacional comum em previsão sazonal: diferentes centros produzem campos de `t2m` com **estruturas temporais diferentes, convenções distintas de `valid_time` e definições inconsistentes de lead time**.

O código padroniza essas diferenças e gera um produto final consistente e pronto para correção e uso operacional.

---

# 1. Escopo do módulo

O repositório contém **dois subpipelines principais**.

## 1.1 `hindcast_t2m`

Responsável por:

- baixar GRIBs de hindcast do CDS
- abrir e interpretar o layout específico de cada centro/modelo
- padronizar a variável `t2m`
- mapear corretamente `lead` e `mês válido`
- opcionalmente regridar
- salvar um **NetCDF padronizado por lead**

---

## 1.2 `hindcast_t2m/correction`

Responsável por:

- ler o forecast operacional bruto (`6/6h` ou diário)
- agregar para média diária quando necessário
- regridar forecast e hindcast para a grade da climatologia observada
- construir uma climatologia diária implícita do hindcast a partir da média mensal
- aplicar correção diária
- salvar o forecast corrigido em NetCDF

---

# 2. Estrutura do projeto

```text
ct_climate_model_corrections_modular/
├── hindcast_t2m/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── detector.py
│   ├── download.py
│   ├── io.py
│   ├── pipeline.py
│   ├── regrid.py
│   ├── utils.py
│   ├── processing/
│   │   ├── base.py
│   │   ├── time_step_like.py
│   │   ├── ecmwf_sys51.py
│   │   ├── meteofrance_sys9.py
│   │   ├── ukmo_sys603.py
│   │   ├── jma_sys3.py
│   │   ├── dwd_sys2.py
│   │   ├── cmcc_sys35.py
│   │   ├── eccc_sys4.py
│   │   └── ncep_sys2.py
│   └── correction/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       └── pipeline.py
├── run_hindcast_t2m.py
└── run_correction_t2m.py
```

---

# 3. Metodologia científica e operacional

## 3.1 Problema tratado pelo pipeline

A variável de entrada no hindcast é:

- `t2m` em **Kelvin**
- disponibilizada como **monthly mean**
- com estrutura temporal dependente do centro produtor.

Diferentes centros utilizam convenções diferentes para:

- marcação do `valid_time`
- posição temporal do mês válido
- definição do lead inicial

Alguns centros:

- marcam o mês válido no **final do intervalo**
- outros no **início do intervalo**
- alguns já fornecem `forecastMonth`
- alguns sistemas começam fisicamente no que operacionalmente corresponde a **lead 2**

O pipeline implementa regras explícitas para resolver essas inconsistências.

---

## 3.2 Metodologia do hindcast mensal

### 3.2.1 Aquisição

Os dados são obtidos do dataset CDS:

`seasonal-monthly-single-levels`

Configuração padrão (`HindcastConfig`):

- variável: `2m_temperature`
- tipo de produto: `monthly_mean`
- período hindcast: `1993–2016`
- leads: `1..6`
- formato: `grib`
- domínio: global `[90, -180, -90, 180]`

Função responsável:

```python
hindcast_t2m.download.download_grib
```

---

### 3.2.2 Padronização da variável

A variável `t2m` representa **temperatura média mensal do ar a 2 metros**, fornecida diretamente em **Kelvin**.

Ao contrário da precipitação, não é necessário converter taxa para acumulado.

O pipeline apenas:

- normaliza o nome da variável
- garante consistência de unidades
- remove dimensões redundantes.

---

### 3.2.3 Determinação do mês alvo e do lead

Regra genérica:

```
target_month = month(valid_time - 1 day)
lead = month(target) - month(init) + 1
```

Processors específicos implementam regras adicionais dependendo do centro produtor.

---

### 3.2.4 Ajuste do mês solicitado ao CDS

Alguns modelos exigem solicitar **o mês anterior** ao CDS para alinhar corretamente o **lead 1**.

Modelos com ajuste:

- Météo-France sys9
- UKMO sys603
- JMA sys3
- DWD sys2
- CMCC sys35
- ECCC sys4
- NCEP sys2

---

### 3.2.5 Colapso temporal

Após processamento, o dataset final assume a forma:

```
t2m(lead, latitude, longitude)
month(lead)
calendar_month(lead)
```

Representando uma **climatologia mensal por lead**.

---

## 3.3 Metodologia da correção diária

O submódulo de correção utiliza:

- forecast operacional diário
- climatologia observada diária
- climatologia mensal do hindcast

### 3.3.1 Hipóteses

1. forecast em escala subdiária ou diária
2. climatologia observada diária disponível
3. hindcast mensal organizado por lead

---

### 3.3.2 Pré-processamento do forecast

Se o forecast for subdiário:

```
F_d = (1/N) * Σ F_t
```

onde `F_d` é a temperatura média diária.

---

### 3.3.3 Regrid espacial

Método principal:

```python
xesmf.Regridder(method="bilinear")
```

Fallback:

```python
xarray.interp
```

---

### 3.3.4 Redistribuição diária do hindcast

A climatologia diária implícita do hindcast é obtida por:

```
w(d) = C_obs(d) / C_obs_mes
H_d = H_m * w(d)
```

onde:

- `C_obs(d)` climatologia diária observada
- `H_m` climatologia mensal do hindcast

---

### 3.3.5 Correção diária

Correção aditiva:

```
F_d_corr = F_d + (C_obs_d - C_hind_d)
```

Correção multiplicativa (opcional):

```
F_d_corr = F_d * (C_obs_d / C_hind_d)
```

---

### 3.3.6 Limitação climatológica

Após correção:

```
C_min = C_obs_d - pσ
C_max = C_obs_d + pσ
```

O valor final é limitado a esse intervalo.

---

# 4. Arquitetura dos módulos

## HindcastPipeline

Responsável por:

- download
- leitura GRIB
- detecção de processor
- processamento
- regrid
- exportação NetCDF

---

## Processor detection

Implementado em:

```python
hindcast_t2m.detector.detect_processor
```

---

## CorrectionPipeline

Responsável por:

- carregar climatologia observada
- agregar forecast diário
- aplicar correção
- salvar forecast corrigido

---

# 5. Formato dos dados

## Hindcast bruto

```
time
step
number
latitude
longitude
```

Variável:

`t2m`

---

## Saída hindcast

```
t2m(lead, latitude, longitude)
```

---

# 6. Dependências

```
python >= 3.10
xarray
numpy
cfgrib
eccodes
cdsapi
netcdf4
xesmf
```

---

# 7. Resumo do método

Fluxo completo:

```
CDS monthly_mean t2m
→ interpretação temporal específica por centro
→ climatologia hindcast por lead
→ regrid espacial
→ forecast bruto → média diária
→ redistribuição diária do hindcast mensal
→ correção diária baseada na climatologia
→ NetCDF corrigido
```
