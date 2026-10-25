"""train_evaluate trains and evaluates the study's arms."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NB = ROOT / "backend" / "notebooks" / "train_evaluate.ipynb"


def _src(code_only=False) -> str:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"]
                     if not code_only or c["cell_type"] == "code")


def test_the_notebook_trains_the_extension_arms_and_writes_the_summary():
    src = _src()
    assert "RUN_ARMS = screen_arms(cfg, ALL_ARMS, optional=ABLATIONS)" in src
    assert "`R_rr`" in src and "`U_edl`" in src
    assert "study_summary.write(REPORT," in src


def test_the_study_keeps_its_weights():
    from backend.config import load
    assert load("notebooks/study").keep_best is True
