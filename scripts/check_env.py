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
    """Compare installed package versions with pinned requirements.txt.

    Returns 0 when every pinned package is present at the required version, and 1 on
    any mismatch or if requirements.txt is missing. Non-``==`` specifiers are skipped
    with a note rather than treated as errors.
    """
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    if not req.is_file():
        print(f"requirements.txt not found at {req}")
        return 1
    problems = []
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            print(f"  (skipping non-pinned specifier: {line})")
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
