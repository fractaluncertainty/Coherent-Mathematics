"""Lightweight repository integrity tests without executing long simulations."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"


def experiment_files():
    return sorted(EXPERIMENTS.glob("*.py"))


def test_all_experiment_sources_parse():
    files = experiment_files()
    assert files, "No experiment sources found"
    for path in files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_all_experiments_have_main_guard():
    for path in experiment_files():
        source = path.read_text(encoding="utf-8")
        assert 'if __name__ == "__main__"' in source or "if __name__ == '__main__'" in source

