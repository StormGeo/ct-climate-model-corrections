ct-correction-cfs-ecmwf

Pipeline for ECMWF System 51 hindcast processing and bias correction (LS-Add) of subseasonal precipitation forecasts, producing monthly accumulated precipitation (mm).

This repository provides the core correction workflow. Detailed script usage and operational examples are documented in the individual README files shipped with each script.

What this repository does

The pipeline is organized in two main stages:

1. Hindcast processing (ECMWF System 51)

Downloads hindcast data from the Copernicus Climate Data Store (CDS)

Variable: total precipitation rate (tprate, m/s)

Converts precipitation to monthly accumulated totals (mm)

Supports leads 1–6

Computes hindcast climatology (mean over members and initialization dates)

Optional regridding to a reference grid

Outputs NetCDF files for use in forecast correction

2. Forecast bias correction

Reads raw subseasonal forecast NetCDF files

Removes incomplete final months

Aggregates data to monthly totals

Harmonizes longitude convention (0–360 ↔ −180–180)

Optional regridding to the same reference grid as climatology

Applies LS-Add bias correction

Ensures physical consistency by truncating negative precipitation values to zero

Bias correction method (LS-Add)

The correction applied is Linear Scaling – Additive (LS-Add), performed independently for each lead and calendar month:

corrected_forecast =
    raw_forecast +
    (observed_climatology(month_of_lead)
     − hindcast_climatology(lead))


Where:

raw_forecast is the monthly accumulated forecast precipitation

observed_climatology is derived from an observational reference dataset

hindcast_climatology is computed from ECMWF System 51 hindcasts

This method preserves forecast anomalies while correcting systematic mean bias.

Dependencies
Required

Python 3.9+

numpy

xarray

Hindcast download and decoding

cdsapi

cfgrib

ecCodes (must be installed at system level)

Optional (regridding)

xesmf

Example installation:

pip install numpy xarray cdsapi cfgrib xesmf


Note: ecCodes must be installed via your system package manager or Conda.

Output structure (strict rule)

Scripts only write output inside the directories explicitly passed via --out-* arguments.
No additional folders are created automatically.

This constraint is intentional to ensure safe integration in operational environments.

Documentation scope

This README describes what the pipeline does

Detailed how-to, arguments, and operational examples are documented in the README files associated with each script

If needed, this README can be:

Adapted to StormGeo operational standards

Shortened for production environments

Extended with a scientific/technical appendix describing assumptions and limitations