#!/usr/bin/env python3

import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))       # `common` lives at the repo root

from common.model import predict           # after the path insert, by necessity

import forecast_data
import forecast_report

DATA = Path(os.environ.get("DATA_DIR", BASE / "data"))
CSV_PATH, META_PATH = DATA / "monthly_series.csv", DATA / "monthly_series.meta.json"

TARGET = os.environ.get("TARGET", "notices")  # notices | workers_affected | both
START = os.environ.get("START", "2015-01")          # first month to keep (YYYY-MM)
_HORIZON = os.environ.get("HORIZON", "6")          # months to forecast
if not (_HORIZON.isdigit() and int(_HORIZON) > 0):
    raise SystemExit(f"HORIZON={_HORIZON!r} is not a positive whole number")
HORIZON = int(_HORIZON)
MODEL = os.environ.get("MODEL", "mistralai/Timestral-209M")  # 5M/28M/123M/209M

KNOWN_TARGETS = ("notices", "workers_affected")


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(f"configure with env vars, not arguments: {sys.argv[1:]}")
    if TARGET != "both" and TARGET not in KNOWN_TARGETS:
        raise SystemExit(f"unknown TARGET={TARGET!r}; use notices, workers_affected or both")

    targets = list(KNOWN_TARGETS) if TARGET == "both" else [TARGET]
    series = forecast_data.load(CSV_PATH, META_PATH, start=START, targets=targets)
    forecast_report.print_provenance(series.dataset, series.sha, series.rebuilt)
    forecast_report.print_history(
        **{key: getattr(series, key) for key in (
            "months", "values", "states_min", "states_max","dropped"
        )}
    )

    # One variate → a 1-D context; both targets → one row per variate, which
    # `predict` hands to the model as a single joint forecast.
    context = (
        [series.values[name] for name in targets]
        if TARGET == "both"
        else series.values[TARGET]
    )
    forecast = predict(context, HORIZON, MODEL)

    forecast_report.print_forecast(
        model=MODEL,
        horizon=HORIZON,
        months=series.months,
        values=series.values,
        joint=TARGET == "both",
        median=forecast.median,
        low=forecast.low,
        high=forecast.high,
    )


if __name__ == "__main__":
    main()
