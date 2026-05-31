import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import azlite.submission_agent as sa


ROOT = Path(__file__).resolve().parents[1]


def _tiny_payload(seed: int = 123) -> dict:
    rng = np.random.default_rng(seed)

    w1 = (rng.standard_normal((2, 3, 3, 3)) * 0.05).astype(np.float32)
    b1 = np.zeros((2,), dtype=np.float32)
    w2 = (rng.standard_normal((2, 2, 3, 3)) * 0.05).astype(np.float32)
    b2 = np.zeros((2,), dtype=np.float32)
    w3 = (rng.standard_normal((2, 2, 3, 3)) * 0.05).astype(np.float32)
    b3 = np.zeros((2,), dtype=np.float32)
    w_fc = (rng.standard_normal((4, 2 * 6 * 7)) * 0.05).astype(np.float32)
    b_fc = np.zeros((4,), dtype=np.float32)
    w_pi = (rng.standard_normal((7, 4)) * 0.05).astype(np.float32)
    b_pi = np.zeros((7,), dtype=np.float32)
    w_v = (rng.standard_normal((1, 4)) * 0.05).astype(np.float32)
    b_v = np.zeros((1,), dtype=np.float32)

    return {
        "version": 1,
        "arch": "tiny_test_net",
        "runtime_mode": "numpy",
        "channels": 3,
        "rows": 6,
        "columns": 7,
        "inarow": 4,
        "normalization": {
            "type": "binary_current_player_planes",
            "include_legal_channel": True,
            "value_range": [0.0, 1.0],
            "dtype": "float32",
        },
        "runtime": {
            "inference_mode": "numpy",
            "c_puct": 1.5,
            "simulations_opening": 8,
            "simulations_midgame": 10,
            "simulations_endgame": 12,
            "simulations_min": 4,
            "simulations_max": 20,
            "time_budget_sec": 0.2,
            "time_budget_max_sec": 0.3,
        },
        "tensors": {
            "features.0.weight": w1.tolist(),
            "features.0.bias": b1.tolist(),
            "features.2.weight": w2.tolist(),
            "features.2.bias": b2.tolist(),
            "features.4.weight": w3.tolist(),
            "features.4.bias": b3.tolist(),
            "backbone.1.weight": w_fc.tolist(),
            "backbone.1.bias": b_fc.tolist(),
            "policy_head.weight": w_pi.tolist(),
            "policy_head.bias": b_pi.tolist(),
            "value_head.0.weight": w_v.tolist(),
            "value_head.0.bias": b_v.tolist(),
        },
    }


def _reset_submission_agent(payload: dict) -> None:
    sa.AZLITE_EXPORT = payload
    sa._EXPORT_CACHE = None
    sa._NP_TENSOR_CACHE = None
    sa._TORCH_STATE = None


def test_submission_agent_policy_masking_and_shape():
    payload = _tiny_payload()
    _reset_submission_agent(payload)

    board = np.zeros((6, 7), dtype=np.int8)
    # Fill column 3 completely so it becomes illegal.
    for r in range(6):
        board[r, 3] = 1 if r % 2 == 0 else 2

    pi, v = sa.predict_policy_value_from_board(board, current_player=1)
    assert pi.shape == (7,)
    assert -1.0 <= float(v) <= 1.0
    assert pi[3] == 0.0
    assert np.isclose(float(np.sum(pi)), 1.0, atol=1e-6)


def test_submission_agent_immediate_win():
    payload = _tiny_payload()
    _reset_submission_agent(payload)

    board = np.zeros((6, 7), dtype=np.int8)
    board[5, 0] = 1
    board[5, 1] = 1
    board[5, 2] = 1

    obs = SimpleNamespace(
        board=board.reshape(-1).tolist(),
        mark=1,
        step=3,
        remainingOverageTime=60.0,
    )
    cfg = SimpleNamespace(rows=6, columns=7, inarow=4)
    move = int(sa.agent(obs, cfg))
    assert move == 3


def test_build_script_from_weights_module(tmp_path: Path):
    payload = _tiny_payload()
    weights_module = tmp_path / "azlite_weights.py"
    submission_out = tmp_path / "submission_out.py"
    weights_module.write_text(f"AZLITE_EXPORT = {repr(payload)}\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "build_azlite_submission.py",
        "--weights-module",
        str(weights_module),
        "--output",
        str(submission_out),
        "--mode",
        "numpy",
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    assert submission_out.exists()

    spec = spec_from_file_location("submission_out_mod", str(submission_out))
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    obs = SimpleNamespace(
        board=[0] * 42,
        mark=1,
        step=0,
        remainingOverageTime=60.0,
    )
    cfg = SimpleNamespace(rows=6, columns=7, inarow=4)
    move = int(mod.agent(obs, cfg))
    assert 0 <= move < 7

