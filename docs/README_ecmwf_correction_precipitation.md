FORECAST MONTHLY CORRECTION PIPELINE (LS-ADD)
===========================================

This script applies a monthly bias correction to raw forecast data using
hindcast climatology and observed climatology.

The correction method implemented is LS-Add (Linear Scaling – Additive).

----------------------------------------------------------------------
WHAT THIS SCRIPT DOES
----------------------------------------------------------------------

For each raw forecast NetCDF file, the script performs the following steps:

1) Preprocess the raw forecast
   - Removes the last month if it is incomplete
   - Aggregates data to monthly totals (monthly sum)
   - Regrids the forecast to a reference grid

2) Load hindcast climatology
   - Hindcast is loaded from a subfolder that matches the Julian day
     of the forecast file (e.g., 335/)
   - Hindcast is assumed to already be on the final grid

3) Apply LS-Add correction
   - For each lead time:
       corrected = forecast + (observed_climatology - hindcast_climatology)
   - Ensures no negative precipitation values

4) Save corrected forecast
   - Preserves the original folder structure relative to the forecast root

----------------------------------------------------------------------
PROCESSING
----------------------------------------------------------------------
python3 ECMWF_correction_precipitation.py \
  --forecast-root /home/felipe/operacao_linux/2025/ECMWF_2025 \
  --hindcast-root /home/felipe/operacao_linux/ECMWF_HINDCAST_OUT_PROCESSED \
  --clim-file /home/felipe/operacao_linux/precip_mensal.nc \
  --out-root /home/felipe/operacao_linux/2025/ECMWF_CORR

----------------------------------------------------------------------
IMPORTANT NOTES
----------------------------------------------------------------------

- The same NetCDF file provided with --clim-file is used for:
  1) Observed monthly climatology (time = 12)
  2) Reference grid (lat/lon) for regridding the raw forecast

- The hindcast must already be on the same grid as the climatology.

- Longitude conventions are automatically aligned when needed
  (e.g., 0–360 to -180–180).

----------------------------------------------------------------------
REQUIREMENTS
----------------------------------------------------------------------

Python packages:
- xarray
- numpy
- pandas (used internally by xarray for time handling)
- netCDF4 (or another xarray-compatible NetCDF backend)

----------------------------------------------------------------------
INPUT / OUTPUT STRUCTURE
----------------------------------------------------------------------

Input directories:

--forecast-root
  Root directory containing raw forecast NetCDF files.
  The directory structure is preserved in the output.

--hindcast-root
  Root directory containing hindcast climatology data.
  Expected structure:
    hindcast_root/
      001/
        hindcast_file.nc
      335/
        hindcast_file.nc
      ...

--clim-file
  NetCDF file containing:
  - Observed climatology with time dimension = 12
  - Latitude and longitude coordinates defining the reference grid

Output directory:

--out-root
  Root directory where corrected forecasts are saved.
  The internal directory structure matches --forecast-root.

----------------------------------------------------------------------
COMMAND LINE ARGUMENTS
----------------------------------------------------------------------

Required arguments:

--forecast-root PATH
  Root directory containing raw forecast NetCDF files.

--hindcast-root PATH
  Root directory containing processed hindcast climatology subfolders.

--clim-file PATH
  NetCDF file with observed climatology (time=12) and reference grid.

--out-root PATH
  Output directory for corrected forecasts.

Optional arguments:

--var-name NAME
  Variable name inside NetCDF files.
  Default: total_precipitation

--to-mm
  Convert data from meters to millimeters (multiply by 1000).

--subfolder NAME
  Process only one specific subfolder (e.g., 335).

--min-steps-last-month N
  Remove the last month if it has fewer than N time steps.
  Default: 2

----------------------------------------------------------------------
EXAMPLES
----------------------------------------------------------------------

Process all forecasts recursively:

python3 forecast_correction_pipeline.py \
  --forecast-root /home/felipe/operacao_linux/2025/ECMWF_2025 \
  --hindcast-root /home/felipe/operacao_linux/ECMWF_HINDCAST_OUT_PROCESSED \
  --clim-file /home/felipe/operacao_linux/precip_mensal.nc \
  --out-root /home/felipe/operacao_linux/2025/ECMWF_CORR


Process only one Julian-day subfolder (example: 335):

python3 forecast_correction_pipeline.py \
  --forecast-root /home/felipe/operacao_linux/2025/ECMWF_2025 \
  --hindcast-root /home/felipe/operacao_linux/ECMWF_HINDCAST_OUT_PROCESSED \
  --clim-file /home/felipe/operacao_linux/precip_mensal.nc \
  --out-root /home/felipe/operacao_linux/2025/ECMWF_CORR \
  --subfolder 335

----------------------------------------------------------------------
CORRECTION METHOD (LS-ADD)
----------------------------------------------------------------------

For each lead time:

bias = observed_climatology(month) - hindcast_climatology(lead)

corrected_forecast = forecast + bias

Negative values are clipped to zero.

----------------------------------------------------------------------
USAGE NOTES
----------------------------------------------------------------------

- This script is designed for operational workflows.
- It assumes consistent naming and directory conventions.
- Paths and defaults can be adapted as needed.

----------------------------------------------------------------------
END OF FILE
----------------------------------------------------------------------
