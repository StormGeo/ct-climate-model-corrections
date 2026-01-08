# ct-correction-cfs-ecmwf

Pipeline for **ECMWF System 51 hindcast processing** and **bias correction (LS-Add)** of subseasonal precipitation forecasts, producing **monthly accumulated precipitation (mm)**.

This repository provides the **core correction workflow**.  
Detailed usage instructions are available in the README files shipped with each script.

---

## Overview

The pipeline is composed of two main stages:

### 1. Hindcast processing (ECMWF System 51)
- Download of hindcast data from the **Copernicus Climate Data Store (CDS)**
- Variable: total precipitation rate (`tprate`, m/s)
- Conversion to **monthly accumulated precipitation (mm)**
- Leads **1–6**
- Computation of hindcast climatology (mean over members and initialization dates)
- Optional regridding to a reference grid
- Output in NetCDF format

### 2. Forecast bias correction
- Reads raw subseasonal forecast NetCDF files
- Removes incomplete final months
- Aggregates data to monthly totals
- Adjusts longitude convention (0–360 ↔ −180–180)
- Optional regridding to the reference grid
- Applies **LS-Add bias correction**
- Negative values are truncated to zero

---

## Bias correction method (LS-Add)

The correction is performed independently for each **lead time** and **calendar month**, using the Linear Scaling – Additive method:

```text
corrected_forecast =
    raw_forecast +
    (observed_climatology(month_of_lead)
     − hindcast_climatology(lead))


Where:

raw_forecast: monthly accumulated forecast precipitation

observed_climatology: climatology from an observational reference dataset

hindcast_climatology: climatology computed from ECMWF System 51 hindcasts

This approach preserves forecast anomalies while correcting systematic mean bias.

## Requirements
Recommended

Python 3.9+
Core dependencies
numpy
xarray
Hindcast download and decoding
cdsapi
cfgrib
ecCodes (system-level installation required)
Optional (regridding)
xesmf

Example installation:

pip install numpy xarray cdsapi cfgrib xesmf

Note: ecCodes must be installed via Conda or the system package manager.

Output structure (strict rule)

All scripts only write files inside directories explicitly passed via --out-* arguments.

No additional folders are created automatically.

This rule is enforced to ensure compatibility with operational workflows.

Documentation scope

This README describes what the pipeline does

Detailed arguments, execution steps, and examples are documented in the script-level README files

