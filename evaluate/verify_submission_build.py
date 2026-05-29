"""
Minimal reproducible checks for the submission build pipeline.

Checks:
1) Building from artifacts succeeds and produces importable single-file agent.
2) Parameter changes in the build config change evaluation output.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

from build_submission import (
    BuildConfig,
    DEFAULT_POSITION_HEATMAP,
    build_submission,
    build_submission_with_config,
)


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_PATH = ROOT / "submission.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_build_from_artifacts() -> None:
    # Build in-place so this matches the real Kaggle submission artifact path.
    build_submission(profile="auto", output_path=SUBMISSION_PATH)
    sub = _load_module(SUBMISSION_PATH, "submission_verify_artifacts")
    assert hasattr(sub, "agent"), "Built submission.py does not define agent()"


def verify_parameter_effect() -> None:
    board = [0] * 42
    board[5 * 7 + 0] = 1
    board[5 * 7 + 1] = 1
    board[5 * 7 + 2] = 2

    with tempfile.TemporaryDirectory(prefix="connectx_verify_build_") as tmp_dir:
        tmp = Path(tmp_dir)
        weak = tmp / "submission_weak.py"
        strong = tmp / "submission_strong.py"

        weak_cfg = BuildConfig(
            w_score=5.0,
            w_threat=50.0,
            w_hmap=0.3,
            w_odd_even=80.0,
            heatmap=[row[:] for row in DEFAULT_POSITION_HEATMAP],
            td_weights=None,
            source="verify.weak",
        )
        strong_cfg = BuildConfig(
            w_score=500.0,
            w_threat=50.0,
            w_hmap=0.3,
            w_odd_even=80.0,
            heatmap=[row[:] for row in DEFAULT_POSITION_HEATMAP],
            td_weights=None,
            source="verify.strong",
        )

        build_submission_with_config(SUBMISSION_PATH, weak, weak_cfg)
        build_submission_with_config(SUBMISSION_PATH, strong, strong_cfg)

        mod_weak = _load_module(weak, "submission_verify_weak")
        mod_strong = _load_module(strong, "submission_verify_strong")

        b1_w, b2_w = mod_weak._list_to_bitboards(board)
        h_w = mod_weak._get_heights_from_list(board)
        s_w = mod_weak.evaluate(b1_w, b2_w, h_w)

        b1_s, b2_s = mod_strong._list_to_bitboards(board)
        h_s = mod_strong._get_heights_from_list(board)
        s_s = mod_strong.evaluate(b1_s, b2_s, h_s)

        diff = s_s - s_w
        print(f"[VERIFY] weak_score={s_w}")
        print(f"[VERIFY] strong_score={s_s}")
        print(f"[VERIFY] diff={diff}")
        assert abs(diff) > 1e-6, "Parameter change did not affect evaluation output"


if __name__ == "__main__":
    verify_build_from_artifacts()
    verify_parameter_effect()
    print("[VERIFY] submission build pipeline checks passed.")
