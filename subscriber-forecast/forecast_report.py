#!/usr/bin/env python3

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np


def print_history(dates: Sequence[str], levels: Sequence[float], net_total: int) -> None:
    print(
        f"{len(dates)} days of history · {dates[0]} → {dates[-1]} · "
        f"level {levels[0]:.0f} → {levels[-1]:.0f} (net {net_total:+d})"
    )


def print_forecast(
    *,
    model: str,
    horizon: int,
    last_day: date,
    snapshot: float,
    median: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> None:
    print(f"\n{horizon}-day forecast · {model} · CPU")
    print(f"{'date':<12}{'p10':>8}{'median':>9}{'p90':>8}{'Δ median':>11}")
    for i in range(horizon):
        day = last_day + timedelta(days=i + 1)
        print(
            f"{day.isoformat():<12}{low[i]:>8.0f}{median[i]:>9.0f}"
            f"{high[i]:>8.0f}{median[i] - snapshot:>+11.0f}"
        )

    final = last_day + timedelta(days=horizon)
    print(
        f"\nProjected on {final}: {median[-1]:.0f} subscribers "
        f"({median[-1] - snapshot:+.0f} vs today's {snapshot:.0f})"
    )
    print(f"80% range: {low[-1]:.0f} – {high[-1]:.0f}")
