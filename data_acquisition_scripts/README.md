# Data Acquisition Script Copies

This folder contains direct copies of the core data acquisition modules from `src/data_collection/`.
They are provided to make it easier to bootstrap a smaller repository focused on a limited set of
sites. Each file keeps the original module name and content so that you can either drop them into a
new package or reference them while building a simplified workflow.

## Dependencies

All collectors assume the root-level `requirements.txt` has been installed (notably `requests`,
`boto3`, `s3fs`, `cdsapi`, and `pyarrow`). Before running any script, make sure you:

1. Export `PYTHONPATH` to the repository root (`export PYTHONPATH="$(pwd)"`).
2. Provide the required credentials:
	 - ERA5: populate `~/.cdsapirc` with your Copernicus API token.
	 - Optional AWS profile/keys if you mirror the NWM archives privately (public S3 reads work
		 anonymously for NOAA buckets).
3. Create the `data/raw/...` directories (or run `scripts/setup_local_storage.sh`) so outputs have a
	 consistent destination.

## Included files

- `era5.py`
- `nwm.py`
- `usgs.py`
- `land_use.py` (legacy NLCD helper, no longer part of the modeling pipeline)

## Quickstart commands

Each collector exposes a CLI—these samples reproduce the Watauga (03479000) pulls documented in the
main `README.md`:

```bash
# USGS hourly observations
python data_acquisition_scripts/usgs.py \
	--sites 03479000 \
	--start-date 2010-01-01 \
	--end-date 2023-12-31 \
	--out-dir data/raw/usgs

# NWM v2 retrospective (2010–2020)
python data_acquisition_scripts/nwm.py \
	--mode retrospective \
	--start-date 2010-01-01 \
	--end-date 2020-12-31 \
	--out-dir data/raw/nwm_v2 \
	--max-workers 6 \
	--resume

# ERA5 forcings (hourly)
python data_acquisition_scripts/era5.py \
	--sites 03479000 \
	--years 2010 2023 \
	--cadence hourly \
	--out-dir data/raw/era5

```

The former NLCD collector (`land_use.py`) is retained for archival purposes only; static land-use fractions are no longer merged into the training parquet.

When migrating to a new project, remember that these modules expect supporting configuration (for
example, `config/master_study_sites.py`) and the same Python dependencies used in this repository.
If you plan to trim the site list, update the relevant constants and helper functions after copying
these files to the new codebase.
