# ct_climate_model_corrections_modular_prec

Pipeline modular para **precipitação total** que cobre duas etapas distintas do processamento climatológico:

1. **Geração do hindcast climatológico mensal de precipitação** a partir do CDS/C3S seasonal forecast database.
2. **Correção diária do forecast operacional bruto** usando a combinação de:
   - climatologia observada diária,
   - hindcast mensal processado por lead,
   - redistribuição intramensal do sinal climatológico.

O projeto foi escrito para resolver um problema operacional real: diferentes centros sazonais publicam `tp` com estruturas temporais diferentes, leads deslocados e convenções distintas de `valid_time`. O código padroniza tudo isso e gera um produto final pronto para correção e uso operacional.

---

## 1. Escopo do módulo

O repositório contém **dois subpipelines**:

### 1.1 `hindcast_tp`
Responsável por:
- baixar GRIBs de hindcast do CDS,
- abrir e interpretar o layout específico de cada centro/modelo,
- converter taxa média mensal de precipitação em acumulado mensal,
- mapear corretamente `lead` e `mês válido`,
- opcionalmente regridar,
- salvar um **NetCDF padronizado por lead**.

### 1.2 `hindcast_tp/correction`
Responsável por:
- ler o forecast operacional bruto (`6/6h` ou diário),
- agregar para total diário quando necessário,
- regridar forecast e hindcast para a grade da climatologia observada,
- construir uma climatologia diária implícita do hindcast a partir do total mensal,
- aplicar a correção diária,
- salvar o forecast corrigido em NetCDF.

---

## 2. Estrutura do projeto

```text
ct_climate_model_corrections_modular/
├── hindcast_tp/
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
├── run_hindcast_tp.py
└── run_correction_tp.py
```

---

## 3. Metodologia científica e operacional

# 3.1 Problema tratado pelo pipeline

A variável de entrada no hindcast é, em geral:

- `tprate` em **m s-1**
- disponibilizada como **monthly mean**
- com estrutura temporal dependente do centro produtor.

Isso significa que o dado bruto **não é diretamente o acumulado mensal de precipitação**. Ele precisa ser convertido, e essa conversão depende de qual mês o `step` representa de fato.

Além disso, alguns centros:
- etiquetam o mês válido no **fim do período**,
- outros no **início do período**,
- outros ainda já chegam com `forecastMonth` explícito,
- e alguns começam fisicamente no que operacionalmente corresponde a **lead 2**, exigindo deslocamento do mês solicitado ao CDS.

O módulo implementa explicitamente essas regras.

---

# 3.2 Metodologia do hindcast mensal

## 3.2.1 Aquisição

Os dados são baixados do dataset CDS:

- `seasonal-monthly-single-levels`

Configuração padrão (`HindcastConfig`):

- variável: `total_precipitation`
- tipo de produto: `monthly_mean`
- período hindcast: `1993–2016`
- leads: `1..6`
- formato: `grib`
- domínio: global `[90, -180, -90, 180]`

A função responsável é:

```python
hindcast_tp.download.download_grib
```

Ela também ajusta a requisição para centros específicos (`jma`, `dwd`, `cmcc`, `eccc`, `ncep`) quando o vocabulário do CDS difere do vocabulário GRIB ou quando certos filtros são necessários para evitar ausência de dados.

---

## 3.2.2 Conversão de taxa para acumulado mensal

A metodologia central do módulo é:

Se `tprate` está em **m s-1**, então o acumulado mensal em **mm/mês** é calculado por:

\[
TP_{mes} = tprate \times S_{mes} \times 1000
\]

onde:

- \(TP_{mes}\) = precipitação total mensal em mm,
- `tprate` = taxa média mensal em m s-1,
- \(S_{mes}\) = número de segundos do mês alvo,
- fator `1000` converte metros para milímetros.

No código, isso é implementado em todos os processors especializados e, no caso genérico, em:

```python
hindcast_tp.processing.time_step_like.TimeStepLikeProcessor
```

Função auxiliar usada:

```python
seconds_in_month_from_ym(ym)
```

Esse detalhe é o núcleo físico do pipeline: o arquivo mensal do CDS representa uma **taxa média**, não um total acumulado pronto.

---

## 3.2.3 Determinação do mês alvo e do lead

O ponto mais importante do processamento é mapear corretamente cada `step` para o **mês calendário válido** e para o **lead operacional 1..6**.

### Regra genérica ECMWF-like (`time_step_like`)

Para estruturas com `time`, `step` e `valid_time`, o método genérico usa:

1. `target_month = month(valid_time - 1 day)`
2. `lead = month(target) - month(init) + 1`

Ou seja, a convenção assume que o `valid_time` marca o início do próximo intervalo e, por isso, é deslocado em **-1 dia** para identificar o mês efetivamente representado.

Isso é apropriado para layouts do tipo ECMWF-like.

### Regra Météo-France sys9

Para Météo-France sys9, o código **não aplica `-1 day`**.

Regra implementada:

1. `target_month = month(valid_time)`
2. `lead = month(target) - month(init)`

Sem o `+1` da regra genérica.

### Regra DWD sys2

Para DWD sys2, o `valid_time` vem no fim do período. O código aplica:

1. `valid_time_adjusted = valid_time + 1 day`
2. usa o mês resultante para construir o lead.

### Regra JMA sys3 / CMCC sys35 / ECCC sys4 / NCEP sys2

Nesses casos o código usa o **mês do `valid_time`** relativo ao `target_month` definido externamente pelo pipeline, com lead normalizado por:

\[
lead_{norm} = ((month(valid\_time) - target\_month) \bmod 12) + 1
\]

Depois filtra apenas `lead 1..6`.

### Regra UKMO sys603

O processor UKMO suporta **dois layouts**:

1. `indexing_time / forecastMonth`
2. `time / step / valid_time`

No layout com `forecastMonth`, o lead operacional pode vir iniciando em `0`; nesse caso o código rebasa para `1..6`.

No layout `time/step`, o código calcula o lead físico entre `init_time` e `valid_time`, encontra o menor lead positivo e rebasa para que o primeiro mês útil seja `lead=1`.

---

## 3.2.4 Ajuste do mês solicitado ao CDS

Além do mapeamento interno do GRIB, o pipeline também altera o **mês pedido ao CDS** para alguns modelos.

Motivação: há sistemas cujo primeiro campo útil do produto mensal corresponde, na prática, ao mês anterior à interpretação operacional desejada. Para alinhar o `lead 1` com o mês alvo, o pipeline solicita o mês anterior.

Implementado em `hindcast_tp.pipeline.HindcastPipeline.run_month`.

Modelos com ajuste para mês anterior:

- Météo-France sys9
- UKMO sys603
- JMA sys3
- DWD sys2
- CMCC sys35
- ECCC sys4
- NCEP sys2

Esse ajuste é operacionalmente essencial. Sem ele, o lead 1 ficaria deslocado em relação ao mês de referência usado na correção posterior.

---

## 3.2.5 Colapso temporal e climatologia hindcast

Após converter `tprate` para `total_precipitation` em mm/mês, o código colapsa as dimensões redundantes e gera um campo por lead.

Dependendo do processor, o colapso pode envolver:

- média entre `step_raw` duplicados dentro do mesmo lead,
- média entre membros de ensemble,
- média entre tempos de inicialização/hindcast years.

O resultado final salvo pelo módulo `hindcast_tp` é um dataset padronizado com forma aproximada:

```text
total_precipitation(lead, latitude, longitude)
month(lead)
calendar_month(lead)   # quando disponível
```

Em outras palavras, a saída é uma **climatologia hindcast mensal por lead**, e não a coleção crua completa de anos/membros.

Esse ponto é importante porque o módulo de correção usa exatamente esse produto resumido como referência mensal do modelo.

---

# 3.3 Metodologia da correção diária

O submódulo `hindcast_tp.correction` aplica uma correção diária ao forecast bruto usando um hindcast mensal e uma climatologia observada diária.

Essa é a parte metodológica mais importante do projeto.

## 3.3.1 Hipóteses adotadas pelo algoritmo

O algoritmo assume:

1. o forecast bruto está em passos subdiários acumulados (tipicamente **6/6h**) ou já em escala diária;
2. a climatologia observada está em **mm/dia** com dimensão `time >= 365`;
3. o hindcast mensal está em **mm/mês** e organizado como:

```text
lead, lat, lon
month(lead)
```

---

## 3.3.2 Pré-processamento do forecast operacional

Arquivo bruto de forecast:
- é aberto em NetCDF,
- tem sua grade padronizada para `lat/lon`,
- é convertido para mm se necessário (`--to-mm`),
- é cortado para o intervalo de leads existente no hindcast,
- é agregado para total diário quando a frequência é subdiária.

### Agregação diária

Se o forecast não estiver em frequência de 24h, o código faz:

\[
F_{d} = \sum_{t \in d} F_{t}
\]

isto é, soma todos os acumulados subdiários do dia para produzir **mm/dia**.

Implementado em:

```python
_forecast_to_daily_accum_mm
```

---

## 3.3.3 Regrid para a grade da climatologia observada

Tanto o hindcast quanto o forecast diário são regridados para a grade do arquivo de climatologia observada.

Método principal:
- `xesmf.Regridder(..., method="bilinear")`

Fallback, caso `xesmf` não esteja disponível:
- `xarray.interp`

O código também:
- ordena latitude e longitude,
- converte longitude de `0..360` para `-180..180` quando necessário,
- reaproveita pesos de regridding em disco.

Isso garante que todos os termos usados na correção estejam na **mesma grade espacial**.

---

## 3.3.4 Construção da climatologia diária observada em lookup rápido

A climatologia observada diária é lida uma única vez e convertida para uma estrutura de acesso rápido por dia do ano:

```text
clim_lookup[doy, y, x]
```

Essa estrutura é preenchida a partir da dimensão `time` do arquivo observado.

O código usa o ano **2001** como calendário de referência para mapear mês e DOY, porque é um ano não bissexto. Quando aparece 29 de fevereiro, ele é colapsado para 28 de fevereiro.

Isso simplifica:
- lookup rápido por DOY,
- soma climatológica mensal a partir de dias do ano,
- compatibilização entre anos operacionais e uma climatologia fixa de 365 dias.

---

## 3.3.5 Redistribuição diária do hindcast mensal

O hindcast disponível após o pipeline mensal está em **mm/mês**. Para corrigir um forecast diário, o código precisa de um equivalente diário do hindcast.

Como esse equivalente diário não existe explicitamente, o algoritmo o constrói distribuindo o total mensal do hindcast segundo o perfil diário da climatologia observada do mesmo mês.

Para um mês \(m\), define-se:

\[
C_{obs}(d)
\]

como a climatologia observada diária daquele dia,

e

\[
C_{obs}^{mes} = \sum_{d \in m} C_{obs}(d)
\]

como o total climatológico mensal observado.

Os pesos diários são então:

\[
w(d) = \frac{C_{obs}(d)}{C_{obs}^{mes}}
\]

quando o denominador é positivo.

Se o total mensal observado for zero, o algoritmo cai para pesos uniformes:

\[
w(d) = \frac{1}{N_d}
\]

onde \(N_d\) é o número de dias daquele mês/lead selecionado.

Com isso, o hindcast mensal do modelo \(H_m\) é redistribuído para diário por:

\[
H_d^{est} = H_m \cdot w(d)
\]

Esse é um ponto metodológico central do projeto: **o ciclo intramensal do hindcast é inferido a partir da forma diária observada**, preservando o total mensal do modelo.

---

## 3.3.6 Correção diária propriamente dita

Para cada dia pertencente a um lead \(L\), o algoritmo compara:

- forecast diário bruto: \(F_d\)
- climatologia observada diária: \(C_{obs,d}\)
- climatologia diária implícita do hindcast: \(C_{hind,d}\)

### Caso principal: correção multiplicativa

Quando \(C_{hind,d}\) é suficientemente grande, aplica-se:

\[
F_d^{corr} = F_d \cdot \frac{C_{obs,d} + \alpha}{C_{hind,d} + \alpha}
\]

onde:
- \(\alpha\) é um termo de regularização,
- por padrão `alpha = denom_min = 1e-3`.

### Caso de fallback: correção aditiva

Quando a climatologia diária do hindcast é muito pequena (`clim_hind_d < denom_min`), a razão multiplicativa pode explodir. Nesse caso o código usa:

\[
F_d^{corr} = F_d + (C_{obs,d} - C_{hind,d})
\]

### Caso especial: climatologia observada zero

Se a climatologia observada do dia for zero, o código **não corrige** aquele valor e mantém o forecast original:

\[
F_d^{corr} = F_d
\]

Isso evita impor uma razão mal definida em regiões/dias climatologicamente secos.

---

## 3.3.7 Limitação da correção por faixa climatológica

Após calcular o candidato corrigido, o algoritmo impõe um limite em torno da climatologia observada diária.

Com `p = limit_p` (padrão `0.30`), os limites são:

\[
C_{min} = (1-p) \cdot C_{obs,d}
\]
\[
C_{max} = (1+p) \cdot C_{obs,d}
\]

E o valor corrigido final é:

\[
F_d^{final} = clip(F_d^{corr}, C_{min}, C_{max})
\]

Na implementação padrão:
- `limit_p = 0.30`
- logo, o forecast corrigido é restringido à faixa **±30% da climatologia observada diária**.

Depois disso, o código ainda impõe:

\[
F_d^{final} \ge 0
\]

via `np.clip(corr_np, 0, None)`.

### Observação importante

Esse comportamento torna o método **fortemente climatológico**. Ou seja, ele não apenas ajusta viés médio; ele força o forecast corrigido a permanecer relativamente próximo da climatologia diária observada. Em aplicações operacionais, isso reduz explosões numéricas e evita extremos irreais, mas também pode amortecer sinal anômalo mais intenso.

---

## 3.3.8 Mapeamento dia → lead

O forecast diário é associado a um lead mensal conforme a diferença entre:

- mês de cada data do forecast,
- mês da data inicial do arquivo.

O código usa:

\[
lead = (month(time_d) - month(init)) + 1
\]

aplicado em espaço `datetime64[M]`.

Assim, todos os dias de um mesmo mês relativo à inicialização usam o mesmo `lead` do hindcast mensal.

---

## 3.3.9 Nome e convenção de saída

A saída corrigida preserva o nome-base do arquivo de forecast e padroniza o membro para `M100`.

Regra implementada:
- se houver `_M###_`, substitui por `_M100_`;
- se não houver, insere `_M100_` antes do `init_stamp`.

Isso gera nomes consistentes para o produto corrigido.

---

## 4. Arquitetura dos módulos

# 4.1 `hindcast_tp.pipeline.HindcastPipeline`

Classe principal do pipeline mensal.

Responsabilidades:
- definir o mês realmente solicitado ao CDS,
- organizar diretórios de saída,
- baixar o GRIB se ele não existir,
- abrir o dataset,
- detectar o processor correto,
- executar o processamento específico,
- regridar opcionalmente,
- salvar em NetCDF.

Método principal:

```python
run_month(month_str, out_year)
```

---

# 4.2 `hindcast_tp.detector.detect_processor`

Implementa seleção automática do processor com base em:
- `originating_centre`
- `system`
- `model_prefix`
- atributos do próprio GRIB.

Ordem atual de tentativa:
- MeteoFranceSys9Processor
- ECMWFSys51Processor
- UKMOSystem603Processor
- JMASys3Processor
- DWDSys2Processor
- CMCCSys35Processor
- ECCCSystem4Processor
- NCEPSys2Processor
- TimeStepLikeProcessor

O último funciona como fallback genérico para layouts `time/step/valid_time`.

---

# 4.3 `hindcast_tp.io`

Contém:
- `open_grib`: abre GRIB com `cfgrib` e aplica `filter_by_keys` específicos por centro quando necessário;
- `normalize_coords_latlon`: padroniza coordenadas para `latitude/longitude`;
- `load_reference_grid`: lê grid de referência para regridding.

Os filtros `cfgrib` são importantes porque alguns GRIBs contêm simultaneamente campos `hcmean` e `fcmean` ou múltiplas definições locais. Sem filtro, a leitura pode falhar ou retornar o subcubo errado.

---

# 4.4 `hindcast_tp.regrid`

Aplica regridding usando `xesmf`.

Entrada esperada:
- dataset com `latitude/longitude`
- variável `total_precipitation(lead, latitude, longitude)`

Saída:
- mesmo dataset, reamostrado para a grade do arquivo de referência.

---

# 4.5 `hindcast_tp.correction.pipeline.ForecastCorrectionPipeline`

Classe principal da correção diária.

Responsabilidades:
- carregar climatologia observada diária,
- construir cache mensal da climatologia,
- carregar hindcast mensal processado,
- agregar forecast subdiário para diário,
- regridar todos os campos para a grade observada,
- aplicar a correção lead a lead,
- salvar o forecast corrigido.

---

## 5. Formato dos dados de entrada e saída

# 5.1 Hindcast bruto do CDS

Variável esperada no GRIB:
- por padrão: `tprate`

Layouts possíveis:
- `time / step / valid_time / number / latitude / longitude`
- `indexing_time / forecastMonth / number / latitude / longitude`

---

# 5.2 Saída do pipeline `hindcast_tp`

Variável:
- `total_precipitation`

Unidade:
- `mm`

Dimensões típicas:

```text
lead, latitude, longitude
```

Coordenadas auxiliares:

```text
month(lead)
calendar_month(lead)
```

---

# 5.3 Entrada do pipeline de correção

### Forecast bruto
NetCDF contendo:
- `time`
- `lat/lon` ou `latitude/longitude`
- `total_precipitation` (ou variável equivalente indicada em `--var-name`)

### Hindcast processado
NetCDF com:
- `lead`
- `month(lead)`
- `total_precipitation(lead, lat, lon)`

### Climatologia observada diária
NetCDF com:
- `time` com 365 ou 366 dias
- grade alvo de referência
- variável diária em `mm/day` (ou `m/day` com `--to-mm`)

---

## 6. Organização de diretórios

# 6.1 Hindcast

GRIB de saída:

```text
<out-grib>/total_precipitation/<YEAR>/<DOY>/arquivo.grib
```

NetCDF de saída:

```text
<out-nc>/total_precipitation/<YEAR>/<DOY>/arquivo.nc
```

O subdiretório `DOY` representa o primeiro dia do mês de inicialização usando ano-base 2001:

- janeiro → `001`
- fevereiro → `032`
- março → `060`
- etc.

Função usada:

```python
out_folder_for_month(month_str)
```

---

# 6.2 Correção

Saída corrigida:

```text
<out-root>/<var_name>/<YEAR>/<DOY>/<arquivo_padronizado>.nc
```

---

## 7. CLI e execução

# 7.1 Geração do hindcast mensal

Script wrapper:

```bash
python run_hindcast_tp.py \
  --out-grib /path/grib \
  --out-nc /path/nc \
  --originating-centre ecmwf \
  --system 51 \
  --model-prefix ecmwf_subseas_glo \
  --input-var tprate \
  --month 01
```

### Modos de operação

#### Modo 1: mês explícito
Use `--month 01..12`.

#### Modo 2: automático por DOY
Use:

```bash
--doy-root /path/2024
```

ou diretamente um subdiretório:

```bash
--doy-root /path/2024/335
```

O CLI detecta o mês correspondente ao DOY mais recente e processa apenas aquele mês.

---

# 7.2 Correção diária do forecast

```bash
python run_correction_tp.py \
  --forecast-root /path/raw_forecast \
  --hindcast-root /path/processed_hindcast \
  --clim-file /path/obs_daily_climatology.nc \
  --out-root /path/corrected \
  --var-name total_precipitation
```

Parâmetros importantes:

- `--to-mm`: converte metros para milímetros
- `--subfolder 335`: processa apenas um DOY
- `--no-skip-existing`: reprocessa mesmo se o arquivo já existir
- `--limit-p 0.30`: largura do clamp em torno da climatologia observada
- `--denom-min 1e-3`: limiar para trocar correção multiplicativa por aditiva

---

## 8. Dependências

Bibliotecas principais:

```text
python >= 3.10
xarray
numpy
cfgrib
eccodes
cdsapi
netcdf4
xesmf
```

Dependência operacional externa:
- credenciais do CDS configuradas para `cdsapi`

---

## 9. Limitações e observações metodológicas

### 9.1 O hindcast salvo é climatológico
A saída de `hindcast_tp` já está agregada espacialmente por lead e temporalmente colapsada sobre membros/inits conforme o processor. Ela não preserva o conjunto bruto completo do hindcast.

### 9.2 A forma diária do hindcast é inferida, não observada
Na correção, a distribuição diária do hindcast dentro do mês é reconstruída a partir da climatologia observada diária. Portanto, o ciclo intramensal do modelo não é estimado diretamente do próprio hindcast mensal.

### 9.3 O clamp pode reduzir extremos
A restrição `±p` em torno da climatologia observada torna o método conservador. Isso ajuda estabilidade operacional, mas pode suavizar extremos legítimos.

### 9.4 O método é fortemente dependente da climatologia observada
Se a climatologia diária observada estiver enviesada ou mal representada, o padrão de redistribuição diária e a correção final herdarão esse problema.

---

## 10. Resumo técnico do método

Em uma frase:

> O módulo converte `tprate` mensal do CDS em `total_precipitation` mensal por lead, harmoniza convenções temporais entre centros, e depois usa uma climatologia observada diária para redistribuir o hindcast mensal no tempo e corrigir diariamente o forecast operacional em uma grade comum.

Fluxo resumido:

```text
CDS monthly_mean tprate
→ interpretação específica por centro
→ conversão para mm/mês
→ climatologia hindcast por lead
→ regrid para grade observada
→ forecast bruto 6/6h → mm/dia
→ redistribuição diária do hindcast mensal
→ correção multiplicativa/aditiva com clamp climatológico
→ NetCDF corrigido
```

---

## 11. Pontos mais importantes para manutenção

Se alguém for evoluir esse código, os pontos críticos são:

1. **regra de lead por centro** → arquivos em `hindcast_tp/processing/`
2. **filtros de leitura do GRIB** → `hindcast_tp/io.py`
3. **ajuste do mês solicitado ao CDS** → `hindcast_tp/pipeline.py`
4. **método de redistribuição diária e correção** → `hindcast_tp/correction/pipeline.py`
5. **parâmetros conservadores do clamp** → `limit_p`, `denom_min`, `alpha`

