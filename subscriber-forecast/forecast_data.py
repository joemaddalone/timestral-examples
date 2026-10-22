#!/usr/bin/env python3

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class Subscribers:
    """The level series, in date order, with today's snapshot at the end."""

    dates: list[str]                 # YYYY-MM-DD
    levels: list[float]              # running subscriber total, one per day
    net_total: int                   # sum of every day's net change

    @property
    def last_day(self) -> date:
        """The most recent analytics day — the day the snapshot is taken on."""
        return date.fromisoformat(self.dates[-1])

    @property
    def snapshot(self) -> float:
        """Today's total, i.e. the last level in the series."""
        return self.levels[-1]


def load(path: Path) -> Subscribers:
    if not path.exists():
        raise SystemExit(f"No data at {path} — run `python fetch_data.py` first")

    with path.open(newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["date"])
    if not rows:
        raise SystemExit(f"{path} is empty — rerun `python fetch_data.py`")

    return Subscribers(
        dates=[r["date"] for r in rows],
        levels=[float(r["level"]) for r in rows],
        net_total=sum(int(r["net"]) for r in rows),
    )
