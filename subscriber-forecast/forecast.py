#!/usr/bin/env python3

import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent))       # `common` lives at the repo root

from common.model import predict           # after the path insert, by necessity

import forecast_data
import forecast_report

SUBS_CSV = Path(os.environ.get("DATA_DIR", BASE / "data")) / "subscribers.csv"
_HORIZON = os.environ.get("HORIZON", "14")        # days to forecast
if not (_HORIZON.isdigit() and int(_HORIZON) > 0):
    raise SystemExit(f"HORIZON={_HORIZON!r} is not a positive whole number")
HORIZON = int(_HORIZON)
MODEL = os.environ.get("MODEL", "mistralai/Timestral-28M")  # 5M/28M/123M/209M


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(f"configure with env vars, not arguments: {sys.argv[1:]}")

    data = forecast_data.load(SUBS_CSV)
    forecast_report.print_history(data.dates, data.levels, data.net_total)

    forecast = predict(data.levels, HORIZON, MODEL)

    forecast_report.print_forecast(
        model=MODEL,
        horizon=HORIZON,
        last_day=data.last_day,
        snapshot=data.snapshot,
        median=forecast.median[0],
        low=forecast.low[0],
        high=forecast.high[0],
    )


if __name__ == "__main__":
    main()
