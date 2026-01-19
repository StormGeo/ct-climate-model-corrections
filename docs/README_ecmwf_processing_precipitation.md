ECMWF System 51 Hindcast Downloader + Processor
==============================================

Single Python script to download and process ECMWF System 51 hindcast monthly
precipitation from Copernicus CDS.

----------------------------------------------------------------------
WHAT THIS SCRIPT DOES
----------------------------------------------------------------------

- Downloads hindcast GRIB data from CDS using cdsapi
- Reads GRIB files with xarray (cfgrib engine)
- Converts tprate (m/s) to monthly accumulated precipitation (mm)
- Builds lead-time fields for lead 1..6
- Computes hindcast climatology by averaging over:
  - ensemble member dimension
  - initialization times (years)
- Optionally regrids the output to a reference grid using xesmf
- Saves the final product as NetCDF

----------------------------------------------------------------------
PROCESSING
----------------------------------------------------------------------
python3 ECMWF_Processing_precipitation.py \
  --doy-root /home/felipe/operacao_linux/temperatura/2m_air_temperature_min/2025 \
  --out-grib /home/felipe/operacao_linux/temperatura/ECMWF_HINDCAST_OUT \
  --out-nc /home/felipe/operacao_linux/temperatura/ECMWF_HINDCAST_OUT_PROCESSED \
  --regrid \
  --ref-grid /home/felipe/operacao_linux/precip_mensal.nc


----------------------------------------------------------------------
REQUIREMENTS
----------------------------------------------------------------------

Python packages:
- cdsapi
- xarray
- numpy
- cfgrib (with eccodes)
- xesmf (only required when using --regrid)

CDS API credentials:
You must have a valid CDS API key configured, usually in:
~/.cdsapirc

----------------------------------------------------------------------
INPUT / OUTPUT STRUCTURE
----------------------------------------------------------------------

The script writes outputs to two main locations:

--out-grib : directory where downloaded GRIB files are stored
--out-nc   : directory where processed NetCDF files are stored

Both outputs are organized using a Julian day-of-year (DOY) subfolder,
derived from the initialization month.

Examples:
- month = 01  -> folder 001
- month = 08  -> folder 213
- month = 12  -> folder 335

----------------------------------------------------------------------
MODES OF OPERATION
----------------------------------------------------------------------

1) EXPLICIT MONTH MODE

You explicitly define which initialization month to download and process.

Example:
python3 hindcast_tp_download_process.py \
  --month 12 \
  --out-grib /path/GRIB \
  --out-nc /path/NC \
  --regrid \
  --ref-grid /home/felipe/operacao_linux/precip_mensal.nc


2) AUTOMATIC MODE (FROM DOY SUBFOLDERS)

You provide a directory that contains subfolders named as DOY values:
001, 015, 215, 315, etc.

The script:
- scans the folder names
- converts DOY -> month using a non-leap dummy year (2001)
- downloads and processes each unique month found

Example directory structure:
/data/observed/merge_daily_as/total_precipitation/2023/
 ├── 001
 ├── 015
 ├── 215
 └── 315

Example command:
python3 hindcast_tp_download_process.py \
  --doy-root /data/observed/merge_daily_as/total_precipitation/2023 \
  --out-grib /path/GRIB \
  --out-nc /path/NC \
  --regrid \
  --ref-grid /home/felipe/operacao_linux/precip_mensal.nc

----------------------------------------------------------------------
REGRIDDING
----------------------------------------------------------------------

Regridding is enabled only when the flag --regrid is provided.

--ref-grid must point to a NetCDF file containing latitude/longitude
coordinates (lat/lon or latitude/longitude).

In the operational workflow, regridding is mandatory to ensure the
hindcast output matches the reference climatology grid.

Example:
--regrid --ref-grid /home/felipe/operacao_linux/precip_mensal.nc

If weight reuse is enabled, regridding weights are saved in:
<out-nc>/_weights/

----------------------------------------------------------------------
PHYSICAL CONVERSION DETAILS
----------------------------------------------------------------------

The script converts precipitation rate to monthly totals as:

total_precipitation_mm =
    tprate (m/s) * seconds_in_target_month * 1000

Target month definition:
- month(valid_time - 1 day)

This approach is robust for:
- varying month lengths
- leap year effects

----------------------------------------------------------------------
TROUBLESHOOTING
----------------------------------------------------------------------

- CDS download errors:
  Check CDS credentials and dataset availability.

- cfgrib errors:
  Ensure eccodes is installed and GRIB decoding works.

- xesmf errors:
  Verify xesmf installation and reference grid coordinates.

- Missing variables:
  The GRIB file must contain tprate. If CDS naming changes,
  the script may need adjustment.

----------------------------------------------------------------------
USAGE NOTES
----------------------------------------------------------------------

This is an internal operational utility script.
Paths, defaults, and flags can be adapted to match local workflows.

----------------------------------------------------------------------
END OF FILE
----------------------------------------------------------------------
