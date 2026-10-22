#!/usr/bin/env python3

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
# Yahoo answers a bare urllib request with 401; any browser-ish agent string works.
USER_AGENT = "Mozilla/5.0 (compatible; timestral-examples/1.0)"
MIN_BARS = 60          # ~3 months; below this the forecast is not worth printing


@dataclass(frozen=True)
class Series:
    """One ticker's adjusted daily closes, in session order."""

    symbol: str
    name: str
    exchange: str
    currency: str
    dates: list[date]
    closes: list[float]           # adjusted close, one per session
    dropped: str | None           # live session removed, YYYY-MM-DD

    @property
    def snapshot(self) -> float:
        """The last completed session's adjusted close."""
        return self.closes[-1]

    def next_sessions(self, count: int) -> list[date]:
        """The next `count` weekdays after the last bar.

        Holidays are not skipped — the calendar is not in the data, and a model
        stepping forward by sessions does not care which weekday a holiday falls
        on. Expect one or two labels per horizon to be a day the market was shut.
        """
        days: list[date] = []
        day = self.dates[-1]
        while len(days) < count:
            day += timedelta(days=1)
            if day.weekday() < 5:                 # Mon–Fri
                days.append(day)
        return days


def _get(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as res:
            return json.load(res)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"HTTP {exc.code} from {url.split('?')[0]} — try another TICKER or RANGE") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach Yahoo Finance ({exc.reason}) — is this machine online?") from exc


def _chart(symbol: str, span: str) -> tuple[dict, dict]:
    """GET one ticker's chart payload and return (result, meta)."""
    url = f"{CHART_URL.format(ticker=symbol)}?range={span}&interval=1d&includeAdjustedClose=true"
    chart = _get(url).get("chart") or {}
    if chart.get("error"):
        raise SystemExit(f"{symbol}: {chart['error'].get('description') or chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise SystemExit(f"no chart data for {symbol} at range {span} — check the ticker")

    result = results[0]
    meta = result["meta"]
    # Yahoo answers an unknown range with 200 and a single bar rather than an
    # error, so check the range it advertises before trusting the response.
    valid = meta.get("validRanges") or []
    if valid and span not in valid:
        raise SystemExit(
            f"RANGE={span!r} is not a range Yahoo offers for {symbol} — use one of: "
            f"{', '.join(valid)}"
        )
    return result, meta


def _session_date(stamp: int, offset: int) -> date:
    """A bar's exchange-local calendar date, from its UTC epoch."""
    return (datetime.fromtimestamp(stamp, timezone.utc) + timedelta(seconds=offset)).date()


def _session_in_progress(meta: dict) -> bool:
    """True while the exchange is inside its regular session.

    Yahoo reports the last trade's timestamp; if it falls inside the current
    session window, the newest daily bar is still moving.
    """
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    start, end, seen = regular.get("start"), regular.get("end"), meta.get("regularMarketTime")
    if start is None or end is None or seen is None:
        return False
    return start <= seen < end


def load(symbol: str, span: str) -> Series:
    """Fetch `symbol` over `span` and return its completed sessions."""
    result, meta = _chart(symbol, span)
    offset = meta.get("gmtoffset") or 0
    quote = result["indicators"]["quote"][0]
    adjusted = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose") or []

    dates, closes = [], []
    for i, stamp in enumerate(result.get("timestamp") or []):
        close = quote["close"][i]
        adj_close = adjusted[i] if i < len(adjusted) else None
        if close is None or adj_close is None:      # halted / no-trade days
            continue
        dates.append(_session_date(stamp, offset))
        closes.append(float(adj_close))
    if not dates:
        raise SystemExit(f"{symbol}: the response carried no usable bars")

    dropped = None
    if _session_in_progress(meta):
        live_day = _session_date(meta["regularMarketTime"], offset)
        if dates[-1] == live_day:
            dropped = dates[-1].isoformat()
            dates, closes = dates[:-1], closes[:-1]

    if len(dates) < MIN_BARS:
        raise SystemExit(
            f"only {len(dates)} session{'s' if len(dates) != 1 else ''} for {symbol} "
            f"at RANGE={span}; need >= {MIN_BARS} — ask for a longer RANGE"
        )

    return Series(
        symbol=meta.get("symbol") or symbol,
        name=meta.get("longName") or meta.get("shortName"),
        exchange=meta.get("fullExchangeName") or meta.get("exchangeName"),
        currency=meta.get("currency"),
        dates=dates,
        closes=closes,
        dropped=dropped,
    )
