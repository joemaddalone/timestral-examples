#!/usr/bin/env python3

from collections.abc import Sequence
from datetime import date
from statistics import fmean

import numpy as np


def print_provenance(dataset: str, sha: str, rebuilt: date) -> None:
    print(f"{dataset} @ {sha[:7]} · rebuilt {rebuilt.isoformat()}")


def print_history(
    *,
    months: Sequence[str],
    values: dict[str, list[float]],
    states_min: int,
    states_max: int,
    dropped: Sequence[str],
) -> None:
    summary = " · ".join(
        f"{name} mean {fmean(vals):.0f}, last {vals[-1]:.0f}" for name, vals in values.items()
    )
    print(
        f"{len(months)} months · {months[0]} → {months[-1]} · {summary} · "
        f"states_reporting {states_min}–{states_max}"
    )
    if dropped:
        print(f"dropped as incomplete: {', '.join(dropped)}")


def print_forecast(
    *,
    model: str,
    horizon: int,
    months: Sequence[str],
    values: dict[str, list[float]],
    joint: bool,
    median: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> None:
    last_year, last_month = map(int, months[-1].split("-"))
    month_index = last_year * 12 + last_month - 1          # months since year 0
    final_year, final_month = divmod(month_index + horizon, 12)

    print(
        f"\n{horizon}-month forecast · {model} · CPU"
        f"{' · joint (2 variates)' if joint else ''}"
    )

    for ti, (name, observed) in enumerate(values.items()):
        if joint:
            print(f"\n{name}:")
        print(f"{'month':<10}{'p10':>8}{'median':>9}{'p90':>8}{'Δ vs last':>11}{'last yr':>10}")
        for i in range(horizon):
            year, m = divmod(month_index + i + 1, 12)
            # same month one year earlier, for eyeballing seasonality; blank once
            # the horizon runs further than the history we have
            seasonal_at = len(observed) - 1 + i + 1 - 12
            seasonal = observed[seasonal_at] if 0 <= seasonal_at < len(observed) else float("nan")
            print(
                f"{f'{year:04d}-{m + 1:02d}':<10}{low[ti][i]:>8.0f}{median[ti][i]:>9.0f}"
                f"{high[ti][i]:>8.0f}{median[ti][i] - observed[-1]:>+11.0f}{seasonal:>10.0f}"
            )
        print(
            f"\nProjected for {final_year:04d}-{final_month + 1:02d}: {median[ti][-1]:.0f} {name} "
            f"({median[ti][-1] - observed[-1]:+.0f} vs {months[-1]}'s {observed[-1]:.0f})"
        )
        print(f"80% range: {low[ti][-1]:.0f} – {high[ti][-1]:.0f}")
