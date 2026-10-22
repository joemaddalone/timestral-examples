# Example 3 — forecasting US WARN Act layoffs

The same model as example 1, pointed at a public dataset instead of the
dashboard: the **AP monthly WARN Act layoffs series** (how many US layoff
notices were filed each month, 1988-11 → today).

| | |
|---|---|
| **Dataset** | [`APProjects/us-layoffs-monthly-time-series-warn-act`](https://huggingface.co/datasets/APProjects/us-layoffs-monthly-time-series-warn-act) |
| **Access** | public, no token, no gating (15 KB CSV) |
| **License** | CC BY 4.0 |
| **Shape** | 455 consecutive months, one row per month, no gaps |
| **Target** | `notices` (default) or `workers_affected` |

One job per file: `fetch_data.py` downloads the dataset into `data/`,
`forecast_data.py` reads only those local files into the trimmed monthly series,
`forecast_report.py` formats the tables, and `forecast.py` runs the model
through the shared `../common/model.py` — the same `predict()` the other two
examples call, which is also what makes the joint two-variate context a one-liner.

## Setup

No new dependencies — stdlib `csv`/`urllib` for the data, plus the `timestral`
install the other examples use. The venv is shared by all three examples and
lives at the repo root, so commands here use `../.venv/bin/python`. If the root
`.venv` doesn't exist yet (or you just deleted it):

```bash
cd ..                      # repo root
uv venv --python 3.12                                  # 1. create it — installs nothing
uv pip install --python .venv/bin/python "git+https://github.com/mistralai/timestral.git"   # 2. the actual install
```

Both lines are required on a fresh venv: line 1 only creates an empty
interpreter, line 2 is what puts `numpy`/`torch`/`timestral` in it. Running only
line 1 and then `forecast.py` is exactly the `No module named 'numpy'` case.

For this folder standalone, run those two `uv` commands here instead, drop the
`../` from `cd ..`, and use `.venv/bin/python` in the check below. Either way,
confirm the install landed before running anything:

```bash
../.venv/bin/python -c "import timestral, numpy, torch; print('deps ok')"
```

An empty `uv pip list --python ../.venv/bin/python`, or a
`ModuleNotFoundError: No module named 'numpy'`, means the `uv pip install` above
went to a different interpreter — re-run it from wherever the `.venv` lives. See
the [root README troubleshooting](../README.md#troubleshooting).

## 1. Fetch the data

```bash
../.venv/bin/python fetch_data.py
```

```
wrote data/monthly_series.csv (14.9 KB)
APProjects/us-layoffs-monthly-time-series-warn-act @ f794e01 · rebuilt 2026-09-25
455 months · wrote data/monthly_series.meta.json
```

| File | Contents |
|---|---|
| `data/monthly_series.csv` | one row per month: `notices`, `workers_affected`, `states_reporting`, … |
| `data/monthly_series.meta.json` | commit sha, rebuild timestamp, byte size, fetch time |

The rebuild timestamp is not decoration: the dataset is rebuilt daily, so its
newest row is always an in-progress month, and the forecast drops every month ≥
that rebuild month. Without the sidecar it could not tell.

The cache is reused when present — `--refresh` (or `REFRESH=1`) re-downloads.
`HF_TOKEN` is optional; it only raises anonymous rate limits.

## 2. Run the forecast

```bash
../.venv/bin/python forecast.py
```

Or from the repo root:
`.venv/bin/python layoffs-forecast/forecast.py`.

| Env var | Default | Meaning |
|---|---|---|
| `TARGET` | `notices` | `notices`, `workers_affected`, or `both` (joint) |
| `START` | `2015-01` | first month to keep (`YYYY-MM`) |
| `HORIZON` | `6` | months to forecast |
| `MODEL` | `mistralai/Timestral-209M` | `5M` / `28M` / `123M` / `209M` |
| `DATA_DIR` | `./data` | where the CSV lives |

This folder defaults to the **209M** model (~837 MB, downloaded on the first
forecast into `~/.cache/huggingface/`); the other two examples default to `28M`.
Pass `MODEL=mistralai/Timestral-28M` for a ~111 MB download instead.

```bash
TARGET=workers_affected ../.venv/bin/python forecast.py
TARGET=both          ../.venv/bin/python forecast.py   # one joint model, two tables
```

Missing `data/` exits with *"No data at … — run `python fetch_data.py` first"*.
Only environment variables configure this script — passing `TARGET=both` as an
argument (instead of `TARGET=both … python …`) is rejected rather than silently
ignored.

### Both series at once (`TARGET=both`)

`workers_affected` is not independent of `notices` — same filings, subset with a
headcount. `TARGET=both` stacks them as two variates of one series
(`context` shape `(1, 2, T)`), which Timestral forecasts jointly, so each series
conditions on the other. You get both tables from one call, sharing one scenario.

Measured on 140 months, 6 rolling origins ending 2023-02 … 2025-08, 6-month
horizon (MAE, lower is better):

| Series | Univariate | Joint | Difference |
|---|---|---|---|
| `notices` | 63.9 | 65.3 | +1.3 (+2%) |
| `workers_affected` | 7,109 | 7,061 | −48 (−0.7%) |
| 80% band coverage | 94% / 92% | 92% / 92% | — |

Both differences are inside fold-to-fold spread (per-fold MAE ranges 42–78 for
notices), so the honest reading is **joint is neither better nor worse on
accuracy** — one series is barely correlated enough with the other to move the
number. Use it when you want both numbers from one coherent forecast, not because
it improves either one. For a single planning figure, `notices` univariate is
still the default.

## Read this before you trust a number

**The pre-2015 series is not a national layoff count** — it mostly counts *how
many states publish WARN notices*. In 1988 one state reported; by 2015 about 25;
today 36–42. That is why `START` defaults to `2015-01`, leaving 140 usable
months. Coverage still drifts inside the window (20–42 states), so year-over-year
levels are a compromise, not a fix.

**How it scores** (6 rolling origins, 6-month horizon):

| Target | Model MAE | Seasonal naive | Flat naive | 80% band coverage |
|---|---|---|---|---|
| `notices` | 62.9 | 78.8 | 68.9 | 94% |
| `workers_affected` | 7,138 | 6,734 | 8,765 | 89% |

At 3 months it merely ties "repeat last month" (55 vs 54) — it only clearly wins
at 6. For `workers_affected` the seasonal baseline wins at 3–6 months, so prefer
`notices`. Don't quote the 12-month row: its verdict flips with the start date.

Sanity checks that the data is wired correctly:

- `140 months · 2015-01 → 2026-08` (inclusive = exactly 140).
- `notices mean 336, last 267`.
- `dropped as incomplete: 2026-09` — exactly one month, always the newest.
- `455 months` at fetch time — the full dataset, including the partial month.
