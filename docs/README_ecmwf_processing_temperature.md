README.txt — ECMWF System 51 Hindcast Monthly Temperature (Download + Processing)

python3 ECMWF_processing_temperature.py \
  --doy-root /home/felipe/operacao_linux/temperatura/2m_air_temperature_min/2025 \
  --out-grib /home/felipe/operacao_linux/ECMWF_HINDCAST_OUT \
  --out-nc /home/felipe/operacao_linux/ECMWF_HINDCAST_OUT_PROCESSED \
  --regrid \
  --ref-grid /home/felipe/operacao_linux/climatology/2m_air_temperature_min/era5_glo_2m_air_temperature_min_20010101_Monthly.nc


1. Purpose
This single script performs end-to-end handling of ECMWF System 51 seasonal hindcast monthly fields for 2 m air temperature:
- Download from CDS (Copernicus Climate Data Store)
- Processing into an operational NetCDF output
- Optional regridding to a reference grid via xESMF

Scope: TEMPERATURE ONLY (2 m temperature), supporting monthly:
- minimum
- mean
- maximum

2. Critical Output Directory Rule (Operational Requirement)
After providing --out-grib and --out-nc, the script MUST ONLY create outputs under:

  <OUT_BASE>/<VAR>/<YEAR>/<DOY>/...

No additional folder layers are allowed (e.g., no “temperature”, no “ECMWF_2025”, no model/group subfolders).

Where:
- <VAR> is one of:
  - 2m_air_temperature_min
  - 2m_air_temperature_med
  - 2m_air_temperature_max
- <YEAR> is automatically detected from the --doy-root path (a 4-digit folder)
- <DOY> is a 3-digit folder (001..366), matching the DOY subfolders found under --doy-root

3. Data Source and Defaults
Dataset: seasonal-monthly-single-levels
Originating centre: ecmwf
System: 51
Base CDS variable: 2m_temperature
Lead times (months): 1..6
Hindcast years: 1993..2016
Initialization date/time: day=01, time=00:00
Spatial area: global [90, -180, -90, 180]
Download format: GRIB

Product type is derived from the detected statistic:
- min  -> monthly_minimum
- mean -> monthly_mean
- max  -> monthly_maximum

4. Requirements
4.1 Runtime
- Python 3.x
- Network access to CDS

4.2 CDS Credentials
A valid CDS API configuration is required (typically at ~/.cdsapirc).

4.3 Python Dependencies
Mandatory:
- cdsapi
- numpy
- xarray
- cfgrib (and ecCodes installation)

Optional (only if using --regrid):
- xesmf

Note: cfgrib requires ecCodes to be installed and discoverable in the environment.

5. Input Conventions and Auto-Detection Logic
The script requires a directory passed via --doy-root that satisfies ALL of the following:

5.1 Variable detection (from path)
The --doy-root path must contain exactly one of these tokens (or equivalent hints):
- 2m_air_temperature_min
- 2m_air_temperature_med
- 2m_air_temperature_max

If none (or more than one) is detected, execution fails.

5.2 Year detection (from path)
The --doy-root path must contain a 4-digit year folder (e.g., 2025). The script scans path parts and selects a valid year in [1900..2100].

5.3 Months selection (from DOY subfolders)
Under --doy-root, the script discovers subfolders named as 3-digit DOY values:
- 001 .. 366

These DOYs are mapped to months using a dummy year (2001) and de-duplicated. The resulting unique months determine which monthly hindcast runs are executed.

6. CLI Usage
Arguments:
--doy-root   (required) Directory containing DOY subfolders (001..366) and embedding <VAR>/<YEAR> in the path
--out-grib   (required) Base output directory for GRIB downloads
--out-nc     (required) Base output directory for NetCDF outputs
--regrid     (optional) Enable regridding via xESMF
--ref-grid   (optional) NetCDF reference grid file (required when --regrid is set)

Examples:

6.1 Without regridding
python3 script.py \
  --doy-root /data/in/2m_air_temperature_med/2025 \
  --out-grib /data/out_grib \
  --out-nc /data/out_nc

6.2 With regridding
python3 script.py \
  --doy-root /data/in/2m_air_temperature_max/2025 \
  --out-grib /data/out_grib \
  --out-nc /data/out_nc \
  --regrid \
  --ref-grid /data/grids/reference_grid.nc

7. Output Structure and File Naming
7.1 Variable folder naming
The output variable folder is derived from the detected var_key:
- t2m_min -> 2m_air_temperature_min
- t2m_med -> 2m_air_temperature_med
- t2m_max -> 2m_air_temperature_max

7.2 Output base paths (final rule enforcement)
The script enforces the following effective roots:
- GRIB root: <OUT_GRIB_BASE>/<VAR>/<YEAR>/
- NC root:   <OUT_NC_BASE>/<VAR>/<YEAR>/

Within each root, outputs are created under DOY folders only:
  .../<DOY>/...

7.3 GRIB output file
Downloaded GRIB files are written to:
  <OUT_GRIB_BASE>/<VAR>/<YEAR>/<DOY>/

File name:
  ecmwf_sys51_<kind>_hindcast_monthly_init<MM>_1993-2016_lead1-6_global.grib

Where:
- <kind> = t2m_min | t2m_med | t2m_max
- <MM>   = 01..12

7.4 NetCDF output file (Operational Naming Standard)
Processed NetCDF files are written to:
  <OUT_NC_BASE>/<VAR>/<YEAR>/<DOY>/

File name pattern:
  ecmwf_subseas_glo_<var>_AVG_<YYYYMMDDHH>.nc

Where:
- <var> is one of:
  2m_air_temperature_min | 2m_air_temperature_med | 2m_air_temperature_max
- <YYYYMMDDHH> is constructed as:
  YYYY + MM + "01" + "00"
  Example: March 2025 -> 2025030100

Example file:
  ecmwf_subseas_glo_2m_air_temperature_med_AVG_2025030100.nc

8. Processing Summary (Temperature)
The processing stage performs:
- Load GRIB with xarray (engine=cfgrib)
- Normalize latitude/longitude coordinate naming
- Rename dimensions:
  - number -> member
  - time   -> init_time
  - step   -> step_raw
- Identify temperature variable (prefers: t2m, then 2t; otherwise single data_var fallback)
- Compute lead month index (1..6) using valid_time vs init_time alignment
- Filter to leads 1..6
- Convert Kelvin to Celsius (K - 273.15)
- Aggregate over step_raw when present
- Produce lead-wise fields and compute climatology:
  mean over member and init_time
- Output dataset dimensions:
  lead, latitude, longitude
- Add coordinate “month” associated to lead (1..6)

9. Optional Regridding (xESMF)
When --regrid is enabled:
- --ref-grid is mandatory and must exist
- Reference grid must expose latitude/longitude (or lat/lon) coordinates
- Regridding method default: bilinear
- periodic default: True

Weights reuse (default enabled):
- Weights are stored under:
  <OUT_NC_BASE>/<VAR>/<YEAR>/_weights/
  weights_<method>_periodic<0|1>.nc

Operational note:
- Final NetCDF outputs remain strictly under:
  <OUT_NC_BASE>/<VAR>/<YEAR>/<DOY>/
- The _weights folder is used only for regridding weight storage within the <VAR>/<YEAR> scope.

10. Operational Checks and Common Failure Modes
- Variable not detected:
  Ensure --doy-root path includes exactly one of:
  2m_air_temperature_min / 2m_air_temperature_med / 2m_air_temperature_max

- Year not detected:
  Ensure --doy-root path includes a 4-digit year directory (e.g., 2025).

- No DOY subfolders found:
  Ensure directories like 001, 032, 215 exist directly under --doy-root.

- GRIB read errors:
  Verify ecCodes and cfgrib installation and environment compatibility.

- Regrid errors:
  Ensure xesmf is installed and --ref-grid contains valid lat/lon or latitude/longitude coordinates.

11. Notes
- This script is designed for operational execution: variable and year are inferred from the input path; run months are derived from DOY folders.
- Outputs are consistent with the operational NetCDF naming convention and the strict directory rule described above.
