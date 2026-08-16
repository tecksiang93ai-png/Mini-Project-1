"""
compare_runs.py — summarise all saved run manifests into one comparison table.

Reads every ``artifacts/run_manifest_<run_id>.json`` and prints the best model and
final test metrics per run, so multiple experiments can be compared at a glance.

    python scripts/compare_runs.py [artifacts_dir]
"""
import json
import sys
from pathlib import Path

import pandas as pd


def main(artifacts_dir: str = "./artifacts") -> int:
    files = sorted(Path(artifacts_dir).glob("run_manifest_*.json"))
    if not files:
        print(f"No run manifests found in {artifacts_dir}. Run training first.")
        return 1
    rows = []
    for f in files:
        m = json.loads(f.read_text(encoding="utf-8"))
        row = {
            "run_id": m.get("run_id"),
            "best_model": m.get("best_model"),
            "cv_mean": round(m.get("best_cv_mean", float("nan")), 4),
        }
        row.update({k: round(v, 4) for k, v in m.get("final_test_metrics", {}).items()})
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "./artifacts"))
