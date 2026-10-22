#!/usr/bin/env python3

import csv
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Series:
    """The trimmed monthly series, one value list per target, in month order."""

    dataset: str                  # Hugging Face dataset id
    sha: str                      # commit the CSV came from, or "unknown"
    rebuilt: date                 # dataset rebuild day
    months: list[str]             # YYYY-MM, oldest first
    values: dict[str, list[float]]  # target name → one value per month
    states_min: int               # states_reporting, lowest month in range
    states_max: int
    dropped: list[str]            # months removed as incomplete


def load(csv_path: Path, meta_path: Path, *, start: str, targets: list[str]) -> Series:
    for path, hint in ((csv_path, "run `python fetch_data.py` first"),
                       (meta_path, "rerun `python fetch_data.py`")):
        if not path.exists():
            raise SystemExit(f"No file at {path} — {hint}")

    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    prov = json.loads(meta_path.read_text())
    rebuilt = date.fromisoformat(prov["last_modified"][:10])

    # A month is complete only if the file was rebuilt after it ended: the dataset
    # is rebuilt daily, so its newest row is always an in-progress month.
    cut = min(rebuilt.strftime("%Y-%m"), date.today().strftime("%Y-%m"))
    series = [r for r in rows if r["month"] < cut and r["month"] >= start]
    if len(series) < 30:
        raise SystemExit(f"only {len(series)} months after START={start}; need >= 30")
    for name in targets:
        if name not in series[0]:
            raise SystemExit(f"column {name!r} missing from {csv_path}")

    states = [int(r["states_reporting"]) for r in series]
    return Series(
        dataset=prov.get("dataset"),
        sha=prov.get("sha") or "unknown",
        rebuilt=rebuilt,
        months=[r["month"] for r in series],
        values={name: [float(r[name]) for r in series] for name in targets},
        states_min=min(states),
        states_max=max(states),
        dropped=[r["month"] for r in rows if r["month"] >= cut],
    )
