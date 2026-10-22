# Example 4 — forecasting a stock ticker, live

The same model as examples 1–3, pointed at the most practical-looking series
there is and the least forgiving: one stock, daily bars, fetched live on every
run. No CSV, no cache, nothing written to disk.

| | |
|---|---|
| **Source** | Yahoo Finance chart endpoint (`/v8/finance/chart/AAPL`) |
| **Access** | no key, no account — one GET, ~136 KB of JSON |
| **Status** | **unofficial**: Yahoo retired its documented API in 2017, so this can break without notice |
| **Shape** | one bar per trading day, ~1,250 bars for the default 5y range, in memory only |
| **Target** | adjusted close |

Three files in this folder, one job each: `forecast_data.py` makes the network
call and shapes the bars, `forecast_report.py` formats the tables, `forecast.py`
runs the model through the shared `../common/model.py`. Unlike the layoffs
example there is no fetch step and no local file — this is the one example that
needs a network connection at forecast time.

> **Read the last section before you trust a number.** A stock price is close to
> a random walk, and that makes forecasting it *look* easier than it is.

## Setup

No new dependencies — stdlib `urllib`, plus the `timestral` install the other
examples use. The venv is shared by all three examples and lives at the repo
root, so commands here use `../.venv/bin/python`. If the root `.venv` doesn't
exist yet (or you just deleted it):

```bash
cd ..                      # repo root
uv venv --python 3.12                                  # 1. create it — installs nothing
uv pip install --python .venv/bin/python "git+https://github.com/mistralai/timestral.git"   # 2. the actual install
```

Both lines are required on a fresh venv: line 1 only creates an empty
interpreter, line 2 is what puts `numpy`/`torch`/`timestral` in it. Running only
line 1 and then `forecast.py` is exactly the `No module named 'numpy'` case.

For this folder standalone, run those two `uv` commands here instead, drop the
`../` from `cd ..`, and use `.venv/bin/python` in the check below. Confirm the
install landed before running:

```bash
../.venv/bin/python -c "import timestral, numpy, torch; print('deps ok')"
```

If that fails with `ModuleNotFoundError`, the install went to another interpreter
— see the [root README troubleshooting](../README.md#troubleshooting).

## Run it

```bash
../.venv/bin/python forecast.py
```

Or from the repo root: `.venv/bin/python stock-forecast/forecast.py`.

```
AAPL · Apple Inc. · NasdaqGS · USD · 1,254 sessions · 2021-10-22 → 2026-10-21 · adjusted close 145.03 → 343.99

10-session forecast · mistralai/Timestral-28M · CPU · adjusted close
date        dow         p10   median      p90  Δ median  Δ median %
2026-10-22  Thu      338.94   343.81   348.59     -0.18     -0.05%
2026-10-23  Fri      337.29   344.15   350.78     +0.16     +0.05%
2026-10-26  Mon      335.47   343.87   352.21     -0.12     -0.04%
…
2026-11-04  Wed      328.20   344.63   361.24     +0.64     +0.19%

flat line, next session = this one: 343.99  (0.00%, by construction)
model median in 10 sessions: 344.63  (+0.19%)
80% band: 328.20 – 361.24  (-4.59% … +5.01%)
the band straddles the flat line — no directional signal here; read its width as the volatility the model expects.
```

| Env var | Default | Meaning |
|---|---|---|
| `TICKER` | `AAPL` | any Yahoo symbol (`NVDA`, `SAP.DE`, `^GSPC` for an index) |
| `RANGE` | `5y` | any Yahoo range: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max` |
| `HORIZON` | `10` | sessions to forecast |
| `MODEL` | `mistralai/Timestral-28M` | `5M` / `28M` / `123M` / `209M` |

```bash
TICKER=NVDA HORIZON=5 ../.venv/bin/python forecast.py
RANGE=max   ../.venv/bin/python forecast.py    # the whole listed history
```

### Live, not cached

Every run is a fresh quote, so two runs minutes apart can print different
numbers — the series moved, not the code. Four consequences worth knowing:

- **Nothing is written.** No `data/` folder, no CSV, no sidecar — the bars live in
  memory for the length of the run. The flip side: if Yahoo is down, there is no
  last-good file to fall back on. The script exits with the HTTP status instead.
- **A live session is not a completed bar.** While the exchange is open the newest
  bar is still moving, so it is dropped and named in the history line
  (`dropped at fetch, session still in progress: …`) — the same rule as the
  layoffs example's in-progress month.
- **`adj_close`, not `close`.** Yahoo already split-adjusts `close` (checked
  against NVDA's 10:1 split: no cliff in either column); `adj_close` additionally
  folds in dividends. The newest bar's adjusted value equals the quoted price,
  older bars are scaled down — which is why the history line reads
  `adjusted close 145.03 → 343.99` for a stock that quoted around 148.69 five
  years ago.
- **Bad input is rejected, not papered over.** An unknown `RANGE` is checked
  against the list Yahoo advertises (it answers one with `200` and a single bar,
  which would otherwise look like a valid fetch); an unknown ticker is a 404; a
  range shorter than 60 sessions exits with a hint to ask for more.

Two more things the script is deliberately naive about:

- **Holidays are not skipped.** The next N sessions are the next N weekdays, so a
  Thanksgiving or July 4th label can appear. The model steps by session, not by
  calendar, so this affects only the printed dates.
- **Adjusted prices are not quoted prices.** Recent values match the market; the
  far end of the history is dividend-scaled. Do not compare the first row to a
  chart from 2021 without adjusting your expectations.

## Read this before you trust a number

**Price-level forecasting flatters every model.** "Tomorrow ≈ today" is a strong
forecast on a random walk: it scores a low MAE, and a model that lands near the
last close inherits that score without adding information. So the honest
question is not "is the forecast close?" but "does it beat the flat line, and
does the band mean what it says?"

Measured on this ticker, 24 non-overlapping origins covering 2025-11 → 2026-10,
10-session horizon (240 evaluated points, `Timestral-28M`):

| | Value |
|---|---|
| MAE, model median | **9.11** |
| MAE, flat line | **9.25** |
| 80% band coverage | **69%** (target 80%) |
| Model median departure from last close at 10 sessions | 0.19% mean, 0.40% max |
| Half-width of the 80% band at 10 sessions | 5.60% mean |

Read that as three findings, not one:

1. **No accuracy gain.** A 1.5% MAE edge over "repeat the last close" is inside
   the spread you get from moving the origins, so the honest statement is a tie.
2. **The band is over-confident.** 69% coverage against a nominal 80% — the same
   over-confidence measured in example 2 (67%). Widen it before using it as risk.
3. **The model is mostly predicting the flat line.** Its median moves ~0.2% at
   10 sessions while its own band spans ±5.6%. The shape is "flat with a growing
   fan", which is what a random walk looks like — and the fan is the useful part.

So: use the band as a volatility sanity check, not as a price target. For a
directional view you need information the price history does not contain
(earnings dates, guidance, sector moves) — none of which this zero-shot setup has.

Which is also why the report prints the flat line: if you report a stock forecast
without a baseline beside it, you cannot tell a working model from a clever way
of saying "about the same as today".

Sanity checks that the data is wired correctly:

- `1,254 sessions · 2021-10-22 → 2026-10-21` for the default `RANGE=5y`, ending on
  the last completed session. A different bar count means a holiday, a halt, or a
  range boundary — not a modelling problem.
- The last `Δ median %` is usually within a percent of 0. A large departure is
  worth checking against `TICKER`/`RANGE` before believing it — it is more often
  a data surprise than a forecast.
- `dropped at fetch, session still in progress: …` appears only when you run
  during market hours.
- Every `dow` in the table matches the real weekday — the one silent bug here is
  a calendar misalignment.

## Notes

- Data is fetched per run and never redistributed: the example writes nothing to
  disk. Yahoo's terms govern what you do with the numbers.
- Not investment advice, and not a trading signal — it is a worked example of
  reading a forecast honestly.
- The endpoint needs a browser-ish `User-Agent`; it answers a bare `urllib`
  request with `401`. That is in `forecast_data.py` and is the whole "API key".
- Because the source is undocumented, treat a sudden failure as Yahoo changing
  shape rather than as something the model did.
- The weights are **Modified MIT** (code is Apache-2.0) — read them before
  commercial use.
