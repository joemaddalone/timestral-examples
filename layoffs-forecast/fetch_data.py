#!/usr/bin/env python3

import argparse
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parent
DATASET = "APProjects/us-layoffs-monthly-time-series-warn-act"


def _request(url: str) -> tuple[bytes, dict]:
    headers = {}
    token = os.environ.get("HF_TOKEN")          # only needed for anonymous rate limits
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as res:
            return res.read(), dict(res.headers)
    except urllib.error.HTTPError as exc:
        raise SystemExit(
            f"HTTP {exc.code} from {url} — the dataset is public; set HF_TOKEN only if you are rate-limited"
        ) from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {url} ({exc.reason}) — is this machine online?") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the public AP WARN Act monthly layoffs series into a local CSV")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", DATASET),
                        help="Hugging Face dataset id (env DATASET)")
    parser.add_argument("--out-dir", default=os.environ.get("DATA_DIR", str(BASE / "data")),
                        help="output folder (env DATA_DIR)")
    parser.add_argument("--refresh", action="store_true",
                        default=os.environ.get("REFRESH") == "1",
                        help="re-download even if a cached copy exists (env REFRESH=1)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, meta_path = out_dir / "monthly_series.csv", out_dir / "monthly_series.meta.json"

    if csv_path.exists() and meta_path.exists() and not args.refresh:
        provenance = json.loads(meta_path.read_text())
        print(f"cached {csv_path} — pass --refresh to re-download")
    else:
        base = f"https://huggingface.co/datasets/{args.dataset}"
        body, headers = _request(f"{base}/resolve/main/data/monthly_series.csv")
        meta = json.loads(_request(f"https://huggingface.co/api/datasets/{args.dataset}")[0])
        provenance = {
            "dataset": args.dataset,
            "sha": headers.get("x-repo-commit") or meta.get("sha"),
            "last_modified": meta.get("lastModified"),
            "bytes": len(body),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        csv_path.write_text(body.decode("utf-8"))
        meta_path.write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"wrote {csv_path} ({len(body) / 1024:.1f} KB)")

    months = len(csv_path.read_text().splitlines()) - 1
    print(
        f"{args.dataset} @ {(provenance.get('sha') or 'unknown')[:7]} · "
        f"rebuilt {(provenance.get('last_modified') or 'unknown')[:10]}"
    )
    print(f"{months} months · wrote {meta_path}")


if __name__ == "__main__":
    main()
