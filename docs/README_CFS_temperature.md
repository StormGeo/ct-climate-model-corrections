README — CFS 2m Temperature Bias-Correction Pipeline (LS-Add)

CFS temperature pipeline (hindcast -> processed + forecast monthly MEAN + LS-Add correction)

Regra solicitada (dinâmica):
- O hindcast processado pode ter N leads (tipicamente 9).
- O forecast corrigido de saída deve ter APENAS os primeiros N meses (os que foram corrigidos).
- Se o forecast tiver mais meses do que o hindcast, os meses extras NÃO aparecem no arquivo final.


python3 CFS_TEMP_teste.py \
  --forecast /home/felipe/operacao_linux/2025/CFS_PREV/2m_air_temperature_min/335 \
  --hindcast-root /home/felipe/operacao_linux/reforecast/cfs_glo \
  --clim-root /home/felipe/operacao_linux/climatology \
  --out-hindcast /home/felipe/operacao_linux/PASTA_HINDCAST \
  --out-corr /home/felipe/operacao_linux/PASTA_CORR \
  --debug


1) Purpose

This script runs an end-to-end operational pipeline for CFS 2m air temperature:

2m_air_temperature_min

2m_air_temperature_med (treated as mean/med; accepts 2m_air_temperature_mean)

2m_air_temperature_max

It builds a lead-based hindcast climatology from reforecast files (per DOY), regrids everything to the observed climatology grid, converts forecast data to monthly mean, applies LS-Add bias correction, and writes corrected NetCDF output.

Key rule: the corrected output contains only months that were corrected, i.e. it is limited to:

n_out = min(n_hindcast_leads_available, n_forecast_months_available)


No “uncorrected tail” is kept in the output file.

2) Dependencies (must be installed)
Required Python packages

Python 3.10+ recommended

numpy

xarray

netCDF4 (or h5netcdf)

Optional but commonly needed

dask (if datasets are large; xarray can use it automatically)

Install (recommended: conda)
conda create -n cfs_temp python=3.10 -y
conda activate cfs_temp
conda install -c conda-forge numpy xarray netcdf4 dask -y

Install (pip alternative)
python3 -m venv venv
source venv/bin/activate
pip install numpy xarray netcdf4 dask

3) Expected Input Directory Layout
3.1 Forecast input (--forecast)

You pass either:

a DOY folder, or

a single NetCDF file inside that DOY folder.

Expected pattern:

.../<YEAR>/CFS_PREV/<VAR>/<DOY>/


Example:

/home/felipe/operacao_linux/2025/CFS_PREV/2m_air_temperature_min/335/

3.2 Hindcast input (--hindcast-root)

You pass the parent root that contains all variables.
The script constructs:

<hindcast-root>/<VAR>/<DOY>/


Example:

--hindcast-root /home/felipe/operacao_linux/reforecast/cfs_glo


Then it reads from:

/home/felipe/operacao_linux/reforecast/cfs_glo/2m_air_temperature_min/335/


Important: Do not pass the variable directory itself (do not pass .../cfs_glo/2m_air_temperature_min).

3.3 Observed climatology input (--clim-root)

You pass the climatology root which contains subfolders by variable:

<clim-root>/<VAR>/


Example:

/home/felipe/operacao_linux/climatology/2m_air_temperature_min/


The script will search for a NetCDF file under:

<clim-root>/<VAR>/**/*.nc

4) Input Data Requirements (NetCDF)
4.1 Hindcast (reforecast) files

Located in <hindcast-root>/<VAR>/<DOY>/

Multiple files per lead: typically M001 .. M009

Each file normally has:

time = 4 synoptic times (00/06/12/18)

latitude, longitude

The script will select 00Z by default.

4.2 Forecast files

Located in <forecast>/<DOY>/ or a single .nc file

Must contain:

time, latitude, longitude

Forecast can be:

already monthly

or sub-monthly (synoptic/hourly/daily) — script converts to monthly mean

4.3 Variable name

Variable is expected to be one of:

2m_air_temperature_min

2m_air_temperature_med

2m_air_temperature_max

If the dataset contains only one variable, the script will use it automatically.
If there are multiple variables and the expected one is missing, it fails.

4.4 Coordinate names

Supported coordinate naming:

latitude / longitude

or lat / lon (script renames automatically)

5) Automatic Variable Detection (min/med/max)

The script detects which temperature product to run using the forecast path string.

Examples:

Path contains 2m_air_temperature_min → uses 2m_air_temperature_min

Path contains 2m_air_temperature_max → uses 2m_air_temperature_max

Path contains 2m_air_temperature_med or 2m_air_temperature_mean → uses “med” logic

If the path does not contain any recognizable token, the script exits with an error.

6) Method Summary
6.1 Hindcast processing (DOY-based)

For each hindcast lead file (M001..M00N):

Open NetCDF

Select 00Z (or first/mean depending on --hindcast-time-mode)

Reduce to (latitude, longitude) field

Stack into lead dimension

Then it creates:

month(lead) using DOY + lead (dummy year 2001)

valid_time(lead) for reference/debug

Finally:

Regrid hindcast to the observed climatology grid (xarray interp())

6.2 Forecast monthly mean

Forecast variable is converted to monthly mean:

If time is already monthly → unchanged

Else → resample(time="MS").mean()

Then:

Regrid forecast to observed climatology grid

6.3 LS-Add correction (lead-based)

For each idx from 0 to n_out-1:

month_hind = month(lead=idx)

month_obs = month_hind + 1 (wrap 12→1)

Bias:

bias = clim_obs(month_obs) - hindcast(lead=idx)


Correct:

corrected(time=idx) = forecast_monthly(time=idx) + bias

6.4 Output limitation

Output length:

n_out = min(n_hindcast_leads, n_forecast_months)


The script writes only the first n_out months (all corrected).

7) Outputs
7.1 Hindcast processed output

Written to:

<out-hindcast>/<VAR>/<YEAR>/<DOY>/cfs_hindcast_<VAR>_doy<DOY>.nc


Example:

/home/felipe/operacao_linux/OUT_HIND_TEMP/2m_air_temperature_min/2025/335/cfs_hindcast_2m_air_temperature_min_doy335.nc

7.2 Corrected forecast output

Written to:

<out-corr>/<VAR>/<YEAR>/<DOY>/cfs_glo_<VAR>_M000_<YYYYMMDDHH>.nc


The <YYYYMMDDHH> stamp is extracted primarily from time.units (e.g., hours since 2025-12-01 00:00:00).
Fallback: first time value if units are missing.

Example:

/home/felipe/operacao_linux/OUT_CORR_TEMP/2m_air_temperature_min/2025/335/cfs_glo_2m_air_temperature_min_M000_2025120100.nc

8) How to Run
Example (minimum temperature)
python3 CFS_TEMP.py \
  --forecast /home/felipe/operacao_linux/2025/CFS_PREV/2m_air_temperature_min/335/ \
  --hindcast-root /home/felipe/operacao_linux/reforecast/cfs_glo \
  --clim-root /home/felipe/operacao_linux/climatology \
  --out-hindcast /home/felipe/operacao_linux/OUT_HIND_TEMP \
  --out-corr /home/felipe/operacao_linux/OUT_CORR_TEMP \
  --debug

Run using a single forecast file
python3 CFS_TEMP.py \
  --forecast /home/felipe/operacao_linux/2025/CFS_PREV/2m_air_temperature_min/335/<file>.nc \
  --hindcast-root /home/felipe/operacao_linux/reforecast/cfs_glo \
  --clim-root /home/felipe/operacao_linux/climatology \
  --out-hindcast /home/felipe/operacao_linux/OUT_HIND_TEMP \
  --out-corr /home/felipe/operacao_linux/OUT_CORR_TEMP

9) CLI Options
Option	Required	Default	Description
--forecast	yes	-	Forecast DOY folder or a NetCDF file (path must contain min/med/max token)
--hindcast-root	yes	-	Parent root containing hindcast variables (script appends <VAR>/<DOY>)
--clim-root	yes	-	Root containing climatology folders by variable
--out-hindcast	yes	-	Output base folder for processed hindcast
--out-corr	yes	-	Output base folder for corrected forecast
--year	no	auto	Override output year if not detectable from forecast path
--hindcast-time-mode	no	00z	How to reduce hindcast time: 00z, first, or mean
--debug	no	off	Prints detection, lead listing, month mapping and output paths
10) Common Errors / Troubleshooting
Pasta de hindcast não encontrada

Cause: --hindcast-root is wrong (often passed already inside <VAR>).

Correct:

--hindcast-root .../reforecast/cfs_glo


Not:

--hindcast-root .../reforecast/cfs_glo/2m_air_temperature_min

Variable not found in NetCDF

Confirm the variable exists in the forecast/hindcast file

If the file has multiple variables and the expected one is missing, the script stops.

Climatology file not found

Ensure this exists:

<clim-root>/<VAR>/*.nc


or any .nc inside subfolders.

Wrong / unexpected output year

If the year cannot be inferred from the forecast path, use:

--year 2025

11) Notes

Regridding uses xarray.interp() (bilinear in practice) to the climatology grid.

Longitude wrap (0..360 → -180..180) is handled automatically when needed.

Output contains only corrected months; no uncorrected data is written.