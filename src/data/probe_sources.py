"""Bounded, explicit-network AKShare source probes with immutable local evidence."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

from src.data.preprocess import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", choices=["sina", "tencent"])
    parser.add_argument("--allow-network", action="store_true", required=True)
    args = parser.parse_args()
    if args.worker:
        import akshare as ak
        try:
            fn = ak.stock_zh_a_daily if args.worker == "sina" else ak.stock_zh_a_hist_tx
            frame = fn(symbol="sz000001", start_date="20260901", end_date="20260925", adjust="hfq")
            frame.to_parquet(args.output / (args.worker + ".parquet"), index=False)
            report = dict(source=args.worker, status="received" if len(frame) else "empty", rows=len(frame),
                          columns=list(frame.columns), preview=json.loads(frame.tail(2).to_json(orient="records", date_format="iso")))
        except Exception as exc:
            report = dict(source=args.worker, status="failed", error=repr(exc))
        write_json(args.output / (args.worker + ".json"), report)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for source in ["sina", "tencent"]:
        try:
            subprocess.run([sys.executable, "-m", "src.data.probe_sources", "--worker", source,
                            "--output", str(args.output), "--allow-network"], timeout=45, check=True)
            report = json.loads((args.output / (source + ".json")).read_text(encoding="utf-8"))
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
            report = dict(source=source, status="failed", error=str(exc))
        results.append(report)
        print(json.dumps(report, ensure_ascii=False), flush=True)
    write_json(args.output / "summary.json", dict(akshare=version("akshare"), results=results,
               symbol="sz000001", start="2026-09-01", end="2026-09-25", basis="hfq",
               note="Connectivity probe only, not a model-ready dataset or complete provider certification."))


if __name__ == "__main__":
    main()
