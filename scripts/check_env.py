"""
check_env.py — verify the installed packages match requirements.txt.

A lightweight environment gate: run before training to confirm the pinned
versions are present, so results are reproducible.

    python scripts/check_env.py
"""
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def main() -> int:
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    problems = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, want = line.partition("==")
        try:
            have = version(name)
        except PackageNotFoundError:
            problems.append(f"{name}: NOT INSTALLED (requires {want})")
            continue
        if want and have != want:
            problems.append(f"{name}: installed {have}, requirements {want}")

    if problems:
        print("Environment mismatch:")
        print("\n".join("  - " + p for p in problems))
        return 1
    print("Environment OK — all pinned packages present at the required versions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
