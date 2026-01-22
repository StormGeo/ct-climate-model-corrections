README — Monthly Aggregation & NetCDF Standardization (Precipitation + 2m Temperature)

python3 climatology.py \
  --in /home/felipe/operacao_linux/2025/CFS_PREV/2m_air_temperature_min/335/ \
  --out /home/felipe/operacao_linux/<folder_to_save_the_data>

1) Purpose

This script batch-processes NetCDF files and produces monthly datasets with standardized naming, metadata, and NetCDF encodings.

It supports two main product types:

Precipitation: monthly aggregation by sum

Output variable name is forced to: total_precipitation

2m Temperature: monthly aggregation by mean

Input variable is detected as temperature and output name becomes:

2m_air_temperature_min

2m_air_temperature_med

2m_air_temperature_max

The temperature suffix (min/med/max) is inferred from the file path string.

Outputs are written under:

<OUT_DIR>/<OUTPUT_VARIABLE>/<original_stem>_Monthly.nc

2) Dependencies (must be installed)
Required

Python 3.9+ (3.10+ recommended)

numpy

pandas

xarray

netCDF4 (or h5netcdf)

Optional

dask (useful for large datasets; xarray can use it automatically)

Install (recommended: conda)
conda create -n monthly_nc python=3.10 -y
conda activate monthly_nc
conda install -c conda-forge numpy pandas xarray netcdf4 dask -y

Install (pip alternative)
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas xarray netcdf4 dask

3) What the script does

For each input NetCDF file:

Opens the dataset (xarray.open_dataset)

Standardizes coordinate names:

lat → latitude

lon → longitude

Ensures time exists and converts it to pandas datetime

Detects the main variable using this priority:

total_precipitation

2m_air_temperature

the first variable in the file

Determines the variable type:

precipitation → monthly sum

temperature → monthly mean

Renames the output variable:

precipitation → total_precipitation

temperature → 2m_air_temperature_<min|med|max>

Writes a standardized NetCDF with:

Global attrs (institution/source/title + description/history timestamp)

Time encoding using hours since <first_time>

Variable encoding filtered to NetCDF4-safe keys, plus defaults:

_FillValue = -9.99e8 (float32)

least_significant_digit = 2

dtype = float32

Saves to:

<out>/<var>/<original_stem>_Monthly.nc

4) Automatic min/med/max detection (temperature)

Temperature suffix is inferred from the input path string:

min if path contains one of:

_min, /min, tmin, minimum, minima

max if path contains one of:

_max, /max, tmax, maximum, maxima

med if path contains one of:

_med, /med, mean, avg, media, média

If nothing matches, it defaults to med

5) Input requirements
5.1 Input can be

A single NetCDF file (.nc)

A directory (the script recursively searches **/*.nc)

5.2 Must contain

A valid time coordinate/dimension (otherwise it fails)

5.3 Coordinates supported

latitude/longitude or lat/lon

6) Outputs
Output directory structure

For each processed file:

<OUT_DIR>/
  <OUTPUT_VARIABLE>/
    <INPUT_STEM>_Monthly.nc


Examples:

Precipitation

out/total_precipitation/cfs_glo_total_precipitation_19821201_M001_Monthly.nc


Temperature

out/2m_air_temperature_min/cfs_glo_2m_air_temperature_min_19821201_M001_Monthly.nc

7) How to run
Process a directory (recursive)
python3 monthly_agg.py --in /path/to/input_dir --out /path/to/out_dir

Process a single file
python3 monthly_agg.py --in /path/to/file.nc --out /path/to/out_dir

Using defaults

If you omit arguments:

--in defaults to current directory .

--out defaults to ./out

Example:

python3 monthly_agg.py

8) CLI Options
Option	Required	Default	Description
--in	no	.	Input NetCDF file or directory
--out	no	./out	Output base directory
9) Notes / Caveats

The script assumes precipitation should be summed monthly, temperature should be mean monthly.

Temperature min/med/max is determined only from the path text, not from metadata inside the dataset.

The script intentionally avoids setting time.attrs["units"] and time.attrs["calendar"] to prevent conflicts; these are placed in time.encoding instead.

Encoding is filtered to NetCDF4-safe keys to avoid write errors from incompatible encodings inherited from source files.

10) Troubleshooting
Error: “Arquivo sem dimensão/coord 'time'”

Cause: input file has no time.
Fix: confirm the file includes a time coordinate/dimension.

Wrong temperature suffix (min/med/max)

Cause: path doesn’t include recognizable tokens.
Fix: ensure the file path contains _min, _max, _med, /min, /max, etc., or accept default med.

Output variable not as expected

Cause: input file has multiple variables and precipitation/temperature base names are not present, so the script picks the first variable.
Fix: ensure the file has the expected variable naming (total_precipitation or 2m_air_temperature) if you want deterministic selection.