# Timestral examples

Three runnable examples, each in its own folder with a README.

| Folder | What it forecasts | Data source |
|---|---|---|
| [`subscriber-forecast/`](subscriber-forecast/) | one channel's daily subscriber count, 14 days | CSV included in the repo — a fixed snapshot, no network |
| [`layoffs-forecast/`](layoffs-forecast/) | US WARN Act layoff notices, 6 months | public Hugging Face dataset (CC BY 4.0) |
| [`stock-forecast/`](stock-forecast/) | one ticker's daily adjusted close, 10 sessions, against the flat line | Yahoo Finance chart JSON (no key, unofficial) |

## One-time install (at the repo root)

```bash
uv venv --python 3.12 && uv pip install --python .venv/bin/python "git+https://github.com/mistralai/timestral.git"
```

First forecast downloads the model weights once into
`~/.cache/huggingface/` (~111 MB for `28M`, ~837 MB for `209M`); later runs load
them from cache and need no network (except `stock-forecast`, which always
fetches).

## Run

```bash
# subscriber: offline, reads the committed CSV
.venv/bin/python subscriber-forecast/forecast.py

# layoffs: fetch once, then forecast (TARGET defaults to notices)
.venv/bin/python layoffs-forecast/fetch_data.py
TARGET=notices .venv/bin/python layoffs-forecast/forecast.py

# the same forecast over headcount instead of filings
TARGET=workers_affected .venv/bin/python layoffs-forecast/forecast.py

# both series jointly (notices + workers_affected, one model, two tables)
TARGET=both .venv/bin/python layoffs-forecast/forecast.py

# live fetch, no local file, no key
TICKER=AAPL .venv/bin/python stock-forecast/forecast.py
```


`TARGET` only affects the layoffs forecast step (its `fetch_data.py` downloads the
whole dataset regardless); `TICKER` and `RANGE` configure the stock example's
live fetch. `HORIZON` and `MODEL` apply to all three. All configuration is
environment variables — stray command-line arguments are
rejected rather than ignored, so `python forecast.py TARGET=both` fails
instead of silently running the default.

Defaults differ per example: `HORIZON` is `14` days (subscriber), `6` months
(layoffs), `10` sessions (stock). Pass `MODEL=mistralai/Timestral-123M` to change default model.


## Model notes

Code is Apache-2.0; the weights are a separate **Modified MIT** license on
Hugging Face — read it before commercial use. Sizes: 5M / 28M / 123M / 209M
parameters (21 MB – 837 MB of weights).
