# CFS Daily Bias-Correction Pipeline (Hindcast Monthly + Forecast 6-hourly)

This script builds a **monthly hindcast dataset** (by lead/month), then uses it to **bias-correct a 6-hourly CFS forecast** into a **daily corrected forecast** on a **reference grid** using **XESMF regridding**.

It is designed to run from a **forecast DOY folder** (day-of-year folder like `001..366`) and automatically finds the matching **hindcast DOY folder**.

time python3 CFS_PRECIPITATION5-hybrid20.py   --forecast /home/felipe/ajuste/2025/CFS_2025/total_precipitation/2026/021   --hindcast-root /home/felipe/ajuste/reforecast/cfs_glo/total_precipitation   --clim-obs /home/felipe/ajuste/merge_as_total_precipitation_20010101.nc   --ref-grid /home/felipe/ajuste/merge_as_total_precipitation_20010101.nc   --out-hindcast /home/felipe/ajuste/dado_2026/PASTA_HINDCAST_TEST1   --out-corr /home/felipe/ajuste/dado_2026/PASTA_CORR_TEST1   --netcdf-engine netcdf4   --debug 2>&1 | tee run_test1.log

---

## What this pipeline does

1. **Detects DOY and Year**
   - DOY is extracted from the forecast path (a folder named `001..366` somewhere in the path).
   - Year is optionally inferred from the path (a folder like `2025`), or you can pass `--year`.

2. **Loads observed daily climatology (fast)**
   - Reads the climatology file **once** and builds a lookup array: `clim_lookup[doy, y, x]`.

3. **Builds processed hindcast (monthly totals by lead)**
   - Reads hindcast NetCDFs for the DOY (expected leads default: 9).
   - Selects a single hindcast time per file (`00z`, `first`, or `mean`).
   - Builds a dataset with dimension `lead` and a `month` coordinate (month associated with that lead).
   - Regrids to the reference grid using **XESMF bilinear**.

4. **Corrects each forecast file**
   - Reads the forecast (6-hourly accumulation in mm).
   - Cuts forecast steps to match available hindcast lead range.
   - Aggregates forecast into **daily totals** (`mm/day`).
   - Regrids forecast to the reference grid (XESMF).
   - Applies daily correction based on:
     - Observed daily climatology (`clim_obs_d`)
     - Estimated hindcast daily climatology derived from monthly totals (`clim_hind_d`)
   - Uses a **multiplicative correction**, with an **additive fallback** where hindcast daily values are too small.
   - Limits corrected values to a **±30% band around observed climatology**.
   - Clips negative values to 0.

5. **Writes outputs**
   - Saves the processed hindcast dataset.
   - Saves corrected forecast files named like:
     - `cfs_glo_<var>_M100_<YYYYMMDDHH>.nc`

---

## Requirements

Python 3 with:

- `numpy`
- `xarray`
- `xesmf`
- `netcdf4` (recommended backend for writing)

Example install:

```bash
pip install numpy xarray xesmf netcdf4

Expected input files
Forecast

A folder containing .nc files, or a single forecast file.
The DOY (001..366) must exist somewhere in the path.

Forecast file name must contain an init stamp like YYYYMMDDHH, e.g.: cfs_glo_total_precipitation_M000_2025010106.nc

Hindcast

A directory tree under --hindcast-root that includes the same DOY folder. Hindcast files are expected to be NetCDF and ideally contain a lead indicator: M001, M002, ... or lead1, lead2, ...

If lead cannot be parsed from filenames, alphabetical order is used as lead=1..N. Observed daily climatology

A NetCDF file with: time dimension representing daily climatology (365 or 366 steps) lat/lon coordinates named lat/lon or latitude/longitude variable default: total_precipitation (or use --var)

Reference grid file

NetCDF file containing the final latitude and longitude coordinates (usually same grid as climatology). Coordinate conventions The script normalizes coordinate names:

lat -> latitude
lon -> longitude

It also sorts by latitude and longitude. Optional longitude adjustment: If forecast/hindcast is 0..360 and reference grid is -180..180, it converts longitudes to -180..180.

CLI arguments

--forecast (required): Forecast DOY folder or a single .nc file inside it

--hindcast-root (required): Base hindcast directory

--clim-obs (required): Observed daily climatology NetCDF (time=365/366, mm/day)

--ref-grid (required): NetCDF containing target grid (lat/lon)

--out-hindcast (required): Output base directory for processed hindcast

--out-corr (required): Output base directory for corrected forecast

--year (optional): Year used in output path (otherwise inferred if possible)

--var (optional): Variable name (default total_precipitation, or auto if dataset has only one var)

--hindcast-time-mode (optional): 00z (default), first, mean

--hindcast-leads-expected (optional): Max number of leads to use (default 9)

--netcdf-engine (optional): NetCDF engine for writing (default netcdf4)

--zlib (optional): Enable compression (slower)

--no-month-cache-disk (optional): Disable writing month cache to disk

--debug (optional): Verbose logs
