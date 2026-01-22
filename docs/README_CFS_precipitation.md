README — CFS Precipitation Bias-Correction Pipeline (LS-Add)

1) Purpose

This script runs an end-to-end operational pipeline for CFS precipitation (total_precipitation):

Reads hindcast/reforecast NetCDF files stored by DOY (001..366), typically one file per lead (M001..M00N).

Selects a single synoptic time from hindcast (00Z by default).

Builds a single processed hindcast dataset with dimension (lead, latitude, longitude).

Creates month(lead) using the DOY + lead rule (dummy year 2001), and a valid_time(lead) helper coordinate.

Regrids both hindcast and forecast to the reference grid (typically the observed climatology grid).

Converts the forecast to monthly accumulated precipitation (sums within each month if sub-monthly).

Applies LS-Add bias correction lead-by-lead:

corrected = forecast_monthly + (obs_clim(month_obs) - hindcast_lead)
month_obs = month_hind + 1 (wrap 12→1)

python3 CFS_PRECIPITATION.py \
  --forecast /home/felipe/operacao_linux/2025/CFS_PREV/total_precipitation/335 \
  --hindcast-root /home/felipe/operacao_linux/reforecast/cfs_glo/total_precipitation \
  --clim-obs /home/felipe/operacao_linux/precip_mensal.nc \
  --ref-grid /home/felipe/operacao_linux/precip_mensal.nc \
  --out-hindcast /home/felipe/operacao_linux/PASTA_HINDCAST \
  --out-corr /home/felipe/operacao_linux/PASTA_CORR \
  --debug

Writes corrected NetCDF output using the filename pattern:

cfs_glo_<var>_M000_<YYYYMMDDHH>.nc


Key rule: the output contains only months that were corrected, i.e. it is limited to:

n_out = min(n_hindcast_leads_available, n_forecast_months_available)


No uncorrected tail is written.



2) Dependencies (must be installed)
Required

Python 3.10+ recommended

numpy

xarray

netCDF4 (or h5netcdf)

Optional (recommended for performance)

dask (for large datasets)

Install (recommended: conda)
conda create -n cfs_precip python=3.10 -y
conda activate cfs_precip
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

.../<YEAR>/CFS_PREV/<DOY>/


Example:

/home/felipe/operacao_linux/2025/CFS_PREV/335/


If you pass a file, it must be inside a DOY folder:

/home/felipe/operacao_linux/2025/CFS_PREV/335/<file>.nc

3.2 Hindcast input (--hindcast-root)

You pass the directory that contains the DOY folders (001..366):

<hindcast-root>/<DOY>/


Example:

--hindcast-root /home/felipe/operacao_linux/reforecast/cfs_glo/total_precipitation


Then hindcast for DOY 335 is read from:

/home/felipe/operacao_linux/reforecast/cfs_glo/total_precipitation/335/

3.3 Observed climatology (--clim-obs)

A NetCDF file with 12 monthly values (time=12) on the target grid.

Example:

/home/felipe/operacao_linux/precip_mensal.nc

3.4 Reference grid (--ref-grid)

A NetCDF file defining the target lat/lon grid. In practice you typically pass the same file as --clim-obs.

Example:

/home/felipe/operacao_linux/precip_mensal.nc

4) Input Data Requirements (NetCDF)
4.1 Hindcast (reforecast) files

Located in:

<hindcast-root>/<DOY>/


Typical structure:

Many .nc files in that folder

Each lead file usually includes:

time = 4 synoptic times (00/06/12/18) (script selects 00Z by default)

latitude, longitude

variable: total_precipitation (or one single variable)

4.2 Forecast files

Located in the forecast DOY folder or a single file:

Must contain:

time, latitude, longitude

Can be:

already monthly, or

sub-monthly (synoptic/hourly/daily) — script converts to monthly accumulated using:

resample(time="MS").sum()

4.3 Variable name (--var)

Default is total_precipitation.
If the dataset has only one variable, it is auto-selected.
If there are multiple variables and total_precipitation is not found, you must set --var.

5) Method Summary
5.1 Hindcast processing (DOY-based)

For each hindcast lead file:

Open NetCDF

Select hindcast time using --hindcast-time-mode:

00z (default): select 00Z when possible, else fallback to first

first: always time[0]

mean: average over time

Reduce to (latitude, longitude)

Stack into lead

Then:

Create month(lead) from DOY + lead (dummy year 2001)

Regrid hindcast to the reference grid (xarray.interp())

5.2 Forecast monthly accumulation

If forecast time axis is monthly, keep it.

Else convert to monthly accumulation:

resample(time="MS").sum()


Then:

Regrid forecast to the reference grid.

5.3 LS-Add correction (lead-based)

For each lead index idx in 0..n_out-1:

month_hind = month(lead=idx)

month_obs = month_hind + 1 (wrap 12→1)

Apply:

bias = obs_clim(month_obs) - hindcast(lead=idx)
corrected(time=idx) = forecast_monthly(time=idx) + bias

5.4 Output limitation

Output length:

n_out = min(n_hindcast_leads, n_forecast_months)


Only the first n_out months are written, all corrected.

6) Outputs
6.1 Processed hindcast output

Written to:

<out-hindcast>/<var>/<year>/<doy>/cfs_hindcast_<var>_doy<doy>.nc


Example:

/home/felipe/operacao_linux/OUT_HIND/total_precipitation/2025/335/cfs_hindcast_total_precipitation_doy335.nc

6.2 Corrected forecast output

Written to:

<out-corr>/<var>/<year>/<doy>/cfs_glo_<var>_M000_<YYYYMMDDHH>.nc


The <YYYYMMDDHH> stamp is extracted from:

time.units (e.g., hours since 1982-12-01 00:00:00), else

first time value.

Example:

/home/felipe/operacao_linux/OUT_CORR/total_precipitation/2025/335/cfs_glo_total_precipitation_M000_2025120100.nc

7) How to Run
Forecast as a DOY folder
python3 CFS_PRECIP.py \
  --forecast /home/felipe/operacao_linux/2025/CFS_PREV/335/ \
  --hindcast-root /home/felipe/operacao_linux/reforecast/cfs_glo/total_precipitation \
  --clim-obs /home/felipe/operacao_linux/precip_mensal.nc \
  --ref-grid /home/felipe/operacao_linux/precip_mensal.nc \
  --out-hindcast /home/felipe/operacao_linux/OUT_HIND \
  --out-corr /home/felipe/operacao_linux/OUT_CORR \
  --debug

Forecast as a single file
python3 CFS_PRECIP.py \
  --forecast /home/felipe/operacao_linux/2025/CFS_PREV/335/<file>.nc \
  --hindcast-root /home/felipe/operacao_linux/reforecast/cfs_glo/total_precipitation \
  --clim-obs /home/felipe/operacao_linux/precip_mensal.nc \
  --ref-grid /home/felipe/operacao_linux/precip_mensal.nc \
  --out-hindcast /home/felipe/operacao_linux/OUT_HIND \
  --out-corr /home/felipe/operacao_linux/OUT_CORR

8) CLI Options
Option	Required	Default	Description
--forecast	yes	–	Forecast DOY folder or a NetCDF file inside it
--hindcast-root	yes	–	Hindcast root that contains 001..366 folders
--clim-obs	yes	–	Observed monthly climatology NetCDF (time=12)
--ref-grid	yes	–	Reference grid NetCDF (often same as --clim-obs)
--out-hindcast	yes	–	Base output folder for processed hindcast
--out-corr	yes	–	Base output folder for corrected forecast
--year	no	auto	Override output year if not detectable from forecast path
--var	no	auto	Variable name override
--hindcast-time-mode	no	00z	00z, first, or mean
--hindcast-leads-expected	no	9	Max leads to attempt to use; actual correction uses available leads
--debug	no	off	Verbose logs
9) Common Errors / Troubleshooting
Pasta de hindcast não encontrada

Cause: --hindcast-root must point to the directory containing DOY folders.

Correct:

--hindcast-root .../reforecast/cfs_glo/total_precipitation


Wrong:

--hindcast-root .../reforecast/cfs_glo/total_precipitation/335

Variable not found

If total_precipitation is not present and there are multiple variables, pass:

--var <actual_variable_name>

Climatology must be monthly (time=12)

The --clim-obs file must contain exactly 12 months on time.

10) Notes

Regridding uses xarray.interp() (bilinear in practice).

Longitude wrapping (0..360 → -180..180) is handled when needed.

Output contains only corrected months; uncorrected forecast months are not written.