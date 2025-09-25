# Data Acquisition Script Copies

This folder contains direct copies of the core data acquisition modules from `src/data_collection/`.
They are provided to make it easier to bootstrap a smaller repository focused on a limited set of
sites. Each file keeps the original module name and content so that you can either drop them into a
new package or reference them while building a simplified workflow.

## Included files

- `era5.py`
- `land_use.py`
- `nwm.py`
- `usgs.py`

When migrating to a new project, remember that these modules expect supporting configuration (for
example, `config/master_study_sites.py`) and the same Python dependencies used in this repository.
If you plan to trim the site list, update the relevant constants and helper functions after copying
these files to the new codebase.

