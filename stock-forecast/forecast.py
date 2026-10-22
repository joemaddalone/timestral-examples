#!/usr/bin/env python3

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # `common` lives at the repo root

from common.model import predict           # after the path insert, by necessity

import forecast_data
import forecast_report

TICKER = os.environ.get("TICKER", "AAPL")          # any Yahoo symbol
RANGE = os.environ.get("RANGE", "5y")              # 1y, 2y, 5y, 10y, max, …
_HORIZON = os.environ.get("HORIZON", "10")        # sessions to forecast
if not (_HORIZON.isdigit() and int(_HORIZON) > 0):
    raise SystemExit(f"HORIZON={_HORIZON!r} is not a positive whole number")
HORIZON = int(_HORIZON)
MODEL = os.environ.get("MODEL", "mistralai/Timestral-28M")  # 5M/28M/123M/209M


def main() -> None:
    if len(sys.argv) > 1:
        raise SystemExit(f"configure with env vars, not arguments: {sys.argv[1:]}")

    data = forecast_data.load(TICKER, RANGE)
    forecast_report.print_history(
        **{key: getattr(data, key) for key in (
            "symbol", "name", "exchange", "currency",
            "dates", "closes", "dropped",
        )}
    )

    forecast = predict(data.closes, HORIZON, MODEL)

    forecast_report.print_forecast(
        model=MODEL,
        horizon_dates=data.next_sessions(HORIZON),
        snapshot=data.snapshot,
        median=forecast.median[0],
        low=forecast.low[0],
        high=forecast.high[0],
    )


if __name__ == "__main__":
    main()
