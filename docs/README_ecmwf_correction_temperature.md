README.txt — Forecast Temperature Monthly Pipeline (Preprocess + LS-Add Correction)

1. Purpose
This script implements an operational forecast correction pipeline for TEMPERATURE ONLY. It performs:
- Forecast preprocessing:
  - removal of an incomplete last month (when applicable)
  - monthly aggregation (when applicable)
  - regridding to a reference grid
  - unit normalization (Kelvin → °C when needed)
  - output variable renaming to match the hindcast convention
- Bias correction using LS-Add with:
  - hindcast climatology (assumed already on the final grid)
  - observed monthly climatology (time=12)

python3 ecmwf_corr_temperature.py \
  --forecast-root /home/felipe/operacao_linux/temperatura/2m_air_temperature_min/2025 \
  --hindcast-root /home/felipe/operacao_linux/NEW_TEMP/ECMWF_HINDCAST_PROCESSED_2 \
  --clim-file /home/felipe/operacao_linux/climatology \
  --out-root /home/felipe/operacao_linux/ECMWF_CORR


The corrected forecasts are written using the SAME output structure as the hindcast products:

  <OUT_BASE>/<VAR>/<YEAR>/<DOY>/...

No extra folders are created.

2. Critical Output Directory Rule (Operational Requirement)
The script MUST ONLY create outputs under:

  <OUT_BASE>/<VAR>/<YEAR>/<DOY>/...

Where:
- <VAR> is one of:
  - 2m_air_temperature_min
  - 2m_air_temperature_med
  - 2m_air_temperature_max
- <YEAR> is automatically detected from the --forecast-root path (a 4-digit folder in the path)
- <DOY> is expected to be the parent folder name of each forecast file (e.g., 001, 032, 335)

Output file naming:
- The output file name is preserved from the input forecast file name:
  <OUT_BASE>/<VAR>/<YEAR>/<DOY>/<original_forecast_filename>.nc

3. Requirements
3.1 Runtime
- Python 3.x

3.2 Python Dependencies
Mandatory:
- numpy
- xarray

Optional (depending on NetCDF contents and environment):
- netCDF4 or h5netcdf (as xarray backends)

Note:
- This script performs regridding using xarray.interp (not xESMF). No xESMF dependency is required here.

4. Inputs
4.1 Forecast files (--forecast-root)
- Root directory containing raw forecast NetCDF files.
- If --subfolder is provided, only --forecast-root/<subfolder>/*.nc are processed.
- Otherwise, the script searches recursively under --forecast-root for *.nc.

Operational expectation:
- Forecast files are organized under DOY directories:
  .../<DOY>/<file>.nc
The DOY directory name is used as the output subfolder.

4.2 Hindcast files (--hindcast-root)
- Root directory containing processed hindcast NetCDF files, organized by DOY subfolders:
  --hindcast-root/<DOY>/*.nc
The first *.nc file found in each DOY folder is opened and used for correction.

Hindcast requirements:
- Must contain:
  - dimension: lead
  - coordinate: month (associated with lead)
- Must contain the target temperature variable (<VAR>)
- Hindcast is assumed to already be in °C and already on the final grid.

4.3 Observed climatology and reference grid (--clim-file)
A single NetCDF file is used for two purposes:
1) Observed monthly climatology:
   - Must contain the variable <VAR> with time dimension of length 12 (one per calendar month).
2) Reference grid for forecast regridding:
   - Must expose lat/lon coordinates (lat/lon or latitude/longitude).

Observed climatology unit handling:
- If the climatology mean > 100, it is assumed to be Kelvin and converted to °C (K - 273.15).

5. Variable Handling (Temperature Only)
5.1 Output variable name (hindcast naming standard)
The script enforces the hindcast-style output variable name:

- 2m_air_temperature_min
- 2m_air_temperature_med
- 2m_air_temperature_max

5.2 How the output variable is determined
Priority order:
1) --var-name (if provided; must be one of the three tokens above)
2) auto-detect from forecast file path
3) auto-detect from --hindcast-root path
4) auto-detect from --clim-file path

If none is detected and --var-name is not provided, execution fails.

5.3 Accepted input temperature variable names (forecast NetCDF)
Forecast NetCDF can contain ANY of the following; the script will auto-select the first match:
- 2m_air_temperature (preferred raw name)
- <VAR> (already in hindcast naming)
- t2m
- 2t
- 2m_temperature

If none are present, the file is rejected.

6. Forecast Preprocessing Logic
6.1 Coordinate standardization
- If coordinates are (latitude, longitude), they are renamed to (lat, lon).

6.2 Removal of incomplete last month (only when time exists)
If the forecast dataset has a time dimension:
- Identify the last calendar month present
- Count the number of time steps in that month
- If that count is < --min-steps-last-month (default 2), the entire last month is removed

This prevents generating monthly aggregates from incomplete data.

6.3 Monthly aggregation (only when time exists)
If the forecast dataset has a time dimension after the previous step, it is resampled to monthly starts (MS):
- For *_min: monthly minimum
- For *_med: monthly mean
- For *_max: monthly maximum

If the dataset has no time dimension, it is treated as already monthly/static.

6.4 Longitude adjustment to match the reference grid
If forecast longitudes are 0..360 but the reference grid uses -180..180, the script shifts and sorts longitudes accordingly.

6.5 Regridding
Forecast data are interpolated onto the reference grid using:
  ds.interp(lat=lat_ref, lon=lon_ref)

The output grid is therefore identical to the climatology grid.

6.6 Unit normalization (Kelvin → °C)
After regridding and aggregation:
- If the mean value > 100, data are treated as Kelvin and converted to °C
- If the mean check fails, the script checks the units attribute; if units are “k” or “kelvin”, convert to °C
Final output units are enforced as:
  units = "degC"

6.7 Output variable renaming
Regardless of the original input variable name, the output DataArray is renamed to <VAR>
and will be used consistently throughout correction and saving.

7. Bias Correction (LS-Add) — Logic Definition
The correction is applied per lead (up to the number of available forecast months):

  corrected = forecast + (clim_obs(month_of_lead) - clim_hindcast(lead))

Where:
- forecast is the preprocessed monthly forecast (time, lat, lon)
- clim_obs(month_of_lead) is selected from the observed climatology (month 1..12)
- clim_hindcast(lead) is selected from the hindcast climatology (lead 1..N)
- month_of_lead is taken from hindcast coordinate “month” aligned to lead

Important consistency checks:
- The script verifies that hindcast and forecast lat/lon match exactly after preprocessing.

8. Output Products
8.1 Directory structure
Outputs are written under:

  <OUT_BASE>/<VAR>/<YEAR>/<DOY>/<original_filename>.nc

Where:
- <YEAR> is extracted from --forecast-root path
- <DOY> is taken from the parent directory of the input forecast file (fpath.parent.name)

8.2 File content
The saved NetCDF contains:
- dataset variable: <VAR>
- dimensions: time (monthly), lat, lon (or potentially only lat/lon if the input is static)
- units: degC

9. CLI Usage
Arguments:
--forecast-root         (required) Root directory containing forecast NetCDF files
--hindcast-root         (required) Root directory containing processed hindcast subfolders (e.g., 335/)
--clim-file             (required) NetCDF containing observed climatology (time=12) and reference grid
--out-root              (required) Output base directory for corrected forecasts
--var-name              (optional) Force output variable name (hindcast style)
--subfolder             (optional) Process only a single subfolder under forecast-root (e.g., 335)
--min-steps-last-month  (optional) Threshold for removing the last month if incomplete (default 2)

Examples:

9.1 Process all forecast files under forecast-root
python3 script.py \
  --forecast-root /data/forecast/2m_air_temperature_med/2025 \
  --hindcast-root /data/hindcast/2m_air_temperature_med/2025 \
  --clim-file /data/climatology/clim_obs_2m_air_temperature_med.nc \
  --out-root /data/out_corrected

9.2 Process only one DOY subfolder (e.g., 335)
python3 script.py \
  --forecast-root /data/forecast/2m_air_temperature_max/2025 \
  --hindcast-root /data/hindcast/2m_air_temperature_max/2025 \
  --clim-file /data/climatology/clim_obs_2m_air_temperature_max.nc \
  --out-root /data/out_corrected \
  --subfolder 335

9.3 Force variable name (recommended for strict operations)
python3 script.py \
  --forecast-root /data/forecast/2025 \
  --hindcast-root /data/hindcast/2025 \
  --clim-file /data/climatology/clim_obs.nc \
  --out-root /data/out_corrected \
  --var-name 2m_air_temperature_min

10. Operational Checks and Common Failure Modes
- Variable detection failure:
  Provide --var-name explicitly or ensure paths include one of:
  2m_air_temperature_min / 2m_air_temperature_med / 2m_air_temperature_max

- Year extraction failure:
  Ensure --forecast-root path contains a 4-digit year directory (e.g., 2025).

- Forecast file not under DOY folder:
  The script uses fpath.parent.name as DOY. Ensure files are located under a DOY directory.

- Missing variable in forecast:
  Confirm the forecast file contains one of:
  2m_air_temperature, <VAR>, t2m, 2t, 2m_temperature

- Observed climatology format:
  Must have time dimension of length 12 for the selected <VAR>.

- Hindcast format:
  Must contain lead dimension and month coordinate, and include <VAR>.

11. Notes
- The same climatology file is used both as observed climatology and as the regridding reference grid.
- Hindcast is assumed to already be on the final grid; only the forecast is regridded.
- Output naming and directory structure are strictly enforced to match the operational hindcast layout.
