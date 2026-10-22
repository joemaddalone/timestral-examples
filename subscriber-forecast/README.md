# Example 1 — subscriber forecast

Projects the channel's daily subscriber count forward with Mistral's
[`timestral`](https://github.com/mistralai/timestral) time-series model.

One job per file, on purpose:

| File | Talks to | Writes / reads |
|---|---|---|
| `forecast_data.py` | nothing | **reads** `data/subscribers.csv` into the level series |
| `forecast_report.py` | nothing | prints the history line and the quantile table |
| `forecast.py` | nothing | runs the model, hands the numbers to the other two |
| `../common/model.py` | Hugging Face, once | the model call all three examples share: context in, p10/median/p90 out |

**The data is committed**, not fetched: `data/subscribers.csv` is a snapshot of
one channel's analytics, and it is the only input this example needs. Nothing
here touches the network except the one-time model download, so the forecast is
reproducible from a fresh clone.

That also means you can re-run the forecast as often as you like, read the CSV
without running anything, or hand it to something else entirely. `predict()` is
the one piece that is shared rather than per-example — it is identical work in
all three folders, so it lives at the repo root and each `forecast.py` adds that
root to `sys.path` before importing it.

## Setup (one time, ~5 min — mostly the model download)

The virtualenv is shared by all three examples and lives at the repo root:

```bash
cd ..                      # repo root, one level above this folder
uv venv --python 3.12                                  # 1. create it — installs nothing
uv pip install --python .venv/bin/python "git+https://github.com/mistralai/timestral.git"   # 2. the actual install
```

Both lines are required on a fresh venv (including one you just deleted): line 1
only creates an empty interpreter, line 2 is what puts
`numpy`/`torch`/`timestral` in it.

Pin Python 3.12 on purpose: it is the pairing these examples were verified with,
and the venv keeps `torch` away from your system Python.

Working on this example alone? Run those same two `uv` commands *inside* this
folder and use `.venv/bin/python` instead of `../.venv/bin/python`.

Confirm the install landed before running:

```bash
../.venv/bin/python -c "import timestral, numpy, torch; print('deps ok')"
```

If that fails with `ModuleNotFoundError`, the `uv pip install` went to another
interpreter rather than this venv — see the
[root README troubleshooting](../README.md#troubleshooting).

## The data (already in the repo)

No setup, no API, no key, no dashboard: `data/subscribers.csv` is committed and
`forecast.py` reads it directly.

`data/subscribers.csv` — one row per analytics day:

| Column | Meaning |
|---|---|
| `date` | `YYYY-MM-DD` |
| `gained` / `lost` / `net` | that day's subscribe, unsubscribe and net change |
| `level` | the running subscriber total, rebuilt backwards from the snapshot |

The dashboard behind this snapshot returns a diary of *changes*; the model wants
the *total*. Since only the latest total is known (9,840 at sync), the level is
derived backwards: `level(yesterday) = level(today) − net(today)`. That single
assumption — the snapshot is the total on the last analytics day — is what the
committed numbers encode, and the `data/subscribers.meta.json` sidecar records it
alongside the snapshot timestamp and the channel id.

Two consequences worth knowing:

- **The series is frozen.** It ends 2026-10-19 and will not grow. That is
  deliberate: a fixed input is what makes the example reproducible, and the
  model only needs history, not today.
- **The capture is not reproducible.** It came from a local dashboard that is not
  part of this repo, and no fetch script ships here — so treat the CSV as the
  source of truth. It is small enough (4.4 KB) to read.

| Env var | Default | Meaning |
|---|---|---|
| `DATA_DIR` | `./data` | folder holding `subscribers.csv` |

## Run the forecast

```bash
../.venv/bin/python forecast.py
```

First run downloads the 111 MB `Timestral-28M` weights to `~/.cache/huggingface/`;
later runs load them from cache (~1.5 s, CPU only, no network).

```bash
# 30-day horizon, bigger model
HORIZON=30 MODEL=mistralai/Timestral-209M ../.venv/bin/python forecast.py
```

| Env var | Default | Meaning |
|---|---|---|
| `HORIZON` | `14` | days to forecast |
| `MODEL` | `mistralai/Timestral-28M` | `5M` / `28M` / `123M` / `209M` |

If the CSV is missing the forecast exits with *"No data at …"* rather than
failing deeper. Restore it from the repo:
`git checkout -- data/subscribers.csv` (from this folder), or
`git checkout -- subscriber-forecast/data/subscribers.csv` from the repo root.

Running from the repo root works too:
`.venv/bin/python subscriber-forecast/forecast.py`.

## Reading the output

`p10` / `median` / `p90` are the model's quantiles per future day; `Δ median` is
the change from today's total. Read the **band**, not the median: if p10 and p90
both point the same way the signal is real, and if they straddle today you have
no signal. 14 days is the honest horizon — at 30 days the same model misses by
~1,100 subscribers on this channel.

Sanity check: the printed `level … → 9840` must match the last `level` in
`data/subscribers.csv`, and `snapshot` in `data/subscribers.meta.json` if that
sidecar is present. If those three disagree, the CSV and the sidecar are from
different captures — believe the CSV, which is what the forecast actually read.

## Notes

- Weights are **Modified MIT** (code is Apache-2.0) — read it before commercial use.
- `data/subscribers.csv` is committed, not generated — it is a fixed snapshot, so
  the forecast stays reproducible. Deleting it removes the example's input; there
  is no fetch script in this repo to rebuild it from.
- Nothing here writes to the dashboard, a database, or the app: this example only
  reads its CSV and calls the model.
- Pre-`Timestral` install or a `ModuleNotFoundError`? See
  [Troubleshooting](../README.md#troubleshooting) in the root README.
