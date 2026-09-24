"""Print source exclusion reasons from the offline dataset audit."""

import argparse
from pathlib import Path

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2] / "data/clean/research_v3")
    args = parser.parse_args()
    audit = pd.read_csv(args.data / "source_audit.csv", dtype={"code": str})
    print(audit.loc[audit.status != "included", ["code", "source_start", "source_end", "status"]].to_string(index=False))
