#!/usr/bin/env python3

from collections.abc import Sequence
from datetime import date

import numpy as np


def print_history(
    *,
    symbol: str,
    name: str,
    exchange: str,
    currency: str,
    dates: Sequence[date],
    closes: Sequence[float],
    dropped: str | None,
) -> None:
    print(
        f"{symbol} · {name} · {exchange} · {currency} · {len(dates):,} sessions · "
        f"{dates[0].isoformat()} → {dates[-1].isoformat()} · "
        f"adjusted close {closes[0]:.2f} → {closes[-1]:.2f}"
    )
    if dropped:
        print(f"dropped at fetch, session still in progress: {dropped}")


def print_forecast(
    *,
    model: str,
    horizon_dates: Sequence[date],
    snapshot: float,
    median: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> None:
    print(f"\n{len(horizon_dates)}-session forecast · {model} · CPU · adjusted close")
    print(
        f"{'date':<12}{'dow':<6}{'p10':>9}{'median':>9}{'p90':>9}"
        f"{'Δ median':>10}{'Δ median %':>12}"
    )
    for i, day in enumerate(horizon_dates):
        print(
            f"{day.isoformat():<12}{day.strftime('%a'):<6}"
            f"{low[i]:>9.2f}{median[i]:>9.2f}{high[i]:>9.2f}"
            f"{median[i] - snapshot:>+10.2f}{(median[i] / snapshot - 1):>+11.2%}"
        )

    # The benchmark, stated plainly: holders of the stock already get this.
    print(f"\nflat line, next session = this one: {snapshot:.2f}  (0.00%, by construction)")
    print(
        f"model median in {len(horizon_dates)} sessions: "
        f"{median[-1]:.2f}  ({(median[-1] / snapshot - 1):+.2%})"
    )
    print(
        f"80% band: {low[-1]:.2f} – {high[-1]:.2f}  "
        f"({(low[-1] / snapshot - 1):+.2%} … {(high[-1] / snapshot - 1):+.2%})"
    )
    if low[-1] > snapshot:
        verdict = ("the whole band sits above the flat line — the model claims an up move; "
                   "test it against the flat line, not against being vaguely close.")
    elif high[-1] < snapshot:
        verdict = ("the whole band sits below the flat line — the model claims a down move; "
                   "test it against the flat line, not against being vaguely close.")
    else:
        verdict = ("the band straddles the flat line — no directional signal here; "
                   "read its width as the volatility the model expects.")
    print(verdict)
