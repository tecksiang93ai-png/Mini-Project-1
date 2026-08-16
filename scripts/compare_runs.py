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
    """Summarise every run_manifest_*.json in ``artifacts_dir`` into one table.

    Returns 1 if the directory is missing or contains no readable manifests;
    individual malformed/unreadable manifests are skipped with a note.
    """
    base = Path(artifacts_dir)
    if not base.is_dir():
        print(f"Artifacts directory not found: {base}")
        return 1
    files = sorted(base.glob("run_manifest_*.json"))
    if not files:
        print(f"No run manifests found in {base}. Run training first.")
        return 1
    rows = []
    for f in files:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  (skipping unreadable manifest {f.name}: {exc})")
            continue
        row = {
            "run_id": m.get("run_id"),
            "best_model": m.get("best_model"),
            "cv_mean": round(m.get("best_cv_mean", float("nan")), 4),
        }
        row.update({k: round(v, 4) for k, v in m.get("final_test_metrics", {}).items()})
        rows.append(row)
    if not rows:
        print("No readable run manifests found.")
        return 1
    print(pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "./artifacts"))
