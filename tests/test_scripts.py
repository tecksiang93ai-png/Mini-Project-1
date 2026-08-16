"""Tests for the utility scripts' failure handling (scripts/compare_runs.py, check_env.py)."""
from scripts import check_env, compare_runs


def test_compare_runs_missing_dir_returns_1(tmp_path):
    assert compare_runs.main(str(tmp_path / "does_not_exist")) == 1


def test_compare_runs_no_manifests_returns_1(tmp_path):
    assert compare_runs.main(str(tmp_path)) == 1


def test_compare_runs_skips_malformed_manifest(tmp_path, capsys):
    (tmp_path / "run_manifest_bad.json").write_text("{ not valid json", encoding="utf-8")
    rc = compare_runs.main(str(tmp_path))
    assert rc == 1                      # no *readable* manifests
    assert "skipping unreadable" in capsys.readouterr().out


def test_check_env_reports_ok(capsys):
    # The active venv is installed from requirements.txt, so the check should pass.
    assert check_env.main() == 0
    assert "Environment OK" in capsys.readouterr().out
