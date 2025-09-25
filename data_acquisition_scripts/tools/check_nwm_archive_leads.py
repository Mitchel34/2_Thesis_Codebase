#!/usr/bin/env python3
"""
Probe availability of NWM short_range forecast files (with lead times) from NCEI archives
for a given date range. Attempts multiple known base URL patterns.

Outputs a CSV with per-URL existence flags and prints a per-day/cycle summary.

Usage examples:
  python3 data_acquisition_scripts/tools/check_nwm_archive_leads.py --start 2021-01-01 --end 2021-01-03 --out data/raw/nwm_v3/archive/check_2021.csv
  python3 data_acquisition_scripts/tools/check_nwm_archive_leads.py --start 2022-06-10 --end 2022-06-12 --out data/raw/nwm_v3/archive/check_2022.csv
  python3 data_acquisition_scripts/tools/check_nwm_archive_leads.py --start 2023-01-01 --end 2023-01-03 --out data/raw/nwm_v3/archive/check_2023.csv
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
import sys
from typing import Iterable, List

import pandas as pd
import requests

# Candidate NCEI THREDDS fileServer base URLs. These may change; we probe all.
BASES = [
    # Historically seen patterns; not guaranteed to be live.
    "https://www.ncei.noaa.gov/thredds/fileServer/model-nwm",
    "https://www.ncei.noaa.gov/thredds/fileServer/nwm",
    # Extra candidates for variants (if any become valid later)
    "https://www.ncei.noaa.gov/thredds/fileServer/noaa-nwm-archive",
]

@dataclass
class ProbeConfig:
    bases: List[str]
    leads: Iterable[int]
    cycles: Iterable[int]
    timeout: float = 10.0


def url_for(base: str, day: datetime, cycle_hh: int, lead: int) -> str:
    ymd = day.strftime("%Y%m%d")
    fff = f"{lead:03d}"
    return (
        f"{base.rstrip('/')}/nwm.{ymd}/short_range/"
        f"nwm.t{cycle_hh:02d}z.short_range.channel_rt.f{fff}.conus.nc"
    )


def head_ok(url: str, timeout: float = 10.0) -> bool:
    try:
        r = requests.head(url, timeout=timeout)
        return r.ok
    except requests.RequestException:
        return False


def probe(cfg: ProbeConfig, start: datetime, end: datetime) -> pd.DataFrame:
    day = start
    records = []
    while day <= end:
        for hh in cfg.cycles:
            for lead in cfg.leads:
                for base in cfg.bases:
                    u = url_for(base, day, hh, lead)
                    ok = head_ok(u, timeout=cfg.timeout)
                    records.append(
                        {
                            "date": day.strftime("%Y-%m-%d"),
                            "cycle": f"{hh:02d}Z",
                            "lead_hour": lead,
                            "exists": ok,
                            "base": base,
                            "url": u if ok else "",
                        }
                    )
        day += timedelta(days=1)
    df = pd.DataFrame.from_records(records)
    return df


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No records to summarize.")
        return
    cov = (
        df.groupby(["date", "cycle", "base"])  # per base summary
        .agg(leads_found=("exists", "sum"))
        .reset_index()
        .sort_values(["date", "cycle", "base"])
    )
    print("Per-day, per-cycle lead coverage by base (leads 1..18):")
    print(cov.to_string(index=False))

    # Also compute a max-over-bases summary to show if any base worked
    cov_any = (
        df.groupby(["date", "cycle"])  # across bases
        .agg(leads_found=("exists", "sum"))
        .reset_index()
        .sort_values(["date", "cycle"])
    )
    print("\nPer-day, per-cycle coverage across all bases:")
    print(cov_any.to_string(index=False))


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe NWM short_range archive lead availability at NCEI")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default=None, help="CSV file to write results")
    ap.add_argument("--base", action="append", default=None, help="Additional/override base URL(s); can repeat")
    ap.add_argument("--leads", default="1-18", help="Lead hours to probe (e.g., 1-18 or 1,3,6,12,18)")
    ap.add_argument("--cycles", default="0,12", help="Init cycles to probe, comma-separated hours (e.g., 0,12)")
    args = ap.parse_args(argv)

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")
    if end < start:
        print("End date must be >= start date", file=sys.stderr)
        return 2

    # Parse leads
    leads_spec = args.leads.strip()
    leads: List[int]
    if "-" in leads_spec:
        a, b = leads_spec.split("-", 1)
        leads = list(range(int(a), int(b) + 1))
    else:
        leads = [int(x) for x in leads_spec.split(",") if x]

    # Parse cycles
    cycles = [int(x) for x in args.cycles.split(",") if x]

    bases = args.base if args.base else BASES
    cfg = ProbeConfig(bases=bases, leads=leads, cycles=cycles)

    print("Probing bases:")
    for b in bases:
        print(" -", b)

    df = probe(cfg, start, end)
    print_summary(df)

    if args.out:
        # Ensure parent dirs will be created by pandas if path exists
        # If folder doesn't exist, instruct user to create it; avoid side-effects here.
        try:
            df.to_csv(args.out, index=False)
            print(f"\nWrote results to: {args.out}")
        except FileNotFoundError:
            print(f"\nOutput path not found: {args.out}. Please create parent directory and re-run.")

    # Return non-zero if nothing found
    found_any = df["exists"].any()
    return 0 if found_any else 1


if __name__ == "__main__":
    sys.exit(main())
