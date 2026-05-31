"""Build a Kaggle-ready single-file submission from AlphaZero-lite checkpoint.

Example:
  python build_azlite_submission.py \
    --checkpoint checkpoints/best.pt \
    --output submission.py \
    --mode numpy
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from azlite.export_model import export_checkpoint


ROOT = Path(__file__).resolve().parent
SUBMISSION_TEMPLATE = ROOT / "azlite" / "submission_agent.py"
DEFAULT_WEIGHTS_MODULE = ROOT / "agents" / "azlite_weights.py"


def _load_py_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load python module from: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_export_payload(weights_module_path: Path) -> Dict[str, Any]:
    mod = _load_py_module(weights_module_path, "azlite_export_weights_tmp")
    payload = getattr(mod, "AZLITE_EXPORT", None)
    if payload is None or not isinstance(payload, dict):
        raise RuntimeError(f"Invalid weights module (AZLITE_EXPORT missing): {weights_module_path}")
    return dict(payload)


def _format_payload_literal(payload: Mapping[str, Any]) -> str:
    return repr(dict(payload))


def _embed_payload_into_template(template_text: str, payload: Mapping[str, Any]) -> str:
    marker = "AZLITE_EXPORT = None  # __AZLITE_EXPORT_PLACEHOLDER__"
    if marker not in template_text:
        raise RuntimeError("submission_agent template marker not found")
    literal = _format_payload_literal(payload)
    repl = f"AZLITE_EXPORT = {literal}  # embedded by build_azlite_submission.py"
    return template_text.replace(marker, repl, 1)


def _ensure_runtime_cfg(payload: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(payload)
    runtime = dict(out.get("runtime") or {})
    runtime.update(
        {
            "inference_mode": str(args.mode),
            "c_puct": float(args.c_puct),
            "simulations_opening": int(args.simulations_opening),
            "simulations_midgame": int(args.simulations_midgame),
            "simulations_endgame": int(args.simulations_endgame),
            "simulations_min": int(args.simulations_min),
            "simulations_max": int(args.simulations_max),
            "time_budget_sec": float(args.time_budget_sec),
            "time_budget_max_sec": float(args.time_budget_max_sec),
        }
    )
    out["runtime"] = runtime
    return out


def build_submission_file(payload: Dict[str, Any], output_path: Path) -> Path:
    template = SUBMISSION_TEMPLATE.read_text(encoding="utf-8")
    built = _embed_payload_into_template(template, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(built, encoding="utf-8")
    return output_path


def _smoke_validate_submission_module(submission_path: Path) -> None:
    mod = _load_py_module(submission_path, "azlite_submission_smoke")
    if not hasattr(mod, "agent"):
        raise RuntimeError("Built submission missing agent(observation, configuration)")

    obs = SimpleNamespace(
        board=[0] * 42,
        mark=1,
        step=0,
        remainingOverageTime=60.0,
    )
    cfg = SimpleNamespace(rows=6, columns=7, inarow=4)
    move = int(mod.agent(obs, cfg))
    if move < 0 or move >= 7:
        raise RuntimeError(f"Smoke check failed: move out of range ({move})")


def _random_positions(num_positions: int, seed: int = 42) -> List[Tuple[np.ndarray, int]]:
    rng = random.Random(seed)
    out: List[Tuple[np.ndarray, int]] = []
    rows, cols = 6, 7

    def legal(board: np.ndarray) -> List[int]:
        return [c for c in range(cols) if board[0, c] == 0]

    def apply(board: np.ndarray, col: int, player: int) -> np.ndarray:
        b = board.copy()
        for r in range(rows - 1, -1, -1):
            if b[r, col] == 0:
                b[r, col] = np.int8(player)
                return b
        return b

    for _ in range(max(1, int(num_positions))):
        board = np.zeros((rows, cols), dtype=np.int8)
        mark = 1
        steps = rng.randint(0, 18)
        for _ in range(steps):
            mv = legal(board)
            if not mv:
                break
            col = rng.choice(mv)
            board = apply(board, col, mark)
            mark = 2 if mark == 1 else 1
        out.append((board, mark))
    return out


def validate_policy_value_alignment(
    checkpoint_path: Path,
    submission_path: Path,
    samples: int,
    device: str,
) -> Optional[Dict[str, float]]:
    try:
        from azlite.model import load_checkpoint, predict_policy_value
    except Exception as exc:
        print(f"[build_azlite] WARN: skip checkpoint-vs-submission alignment (deps unavailable): {exc}")
        return None

    mod = _load_py_module(submission_path, "azlite_submission_validate_align")
    if not hasattr(mod, "predict_policy_value_from_board"):
        print("[build_azlite] WARN: built submission missing predict_policy_value_from_board; skip alignment.")
        return None

    model, _meta = load_checkpoint(str(checkpoint_path), device=device)
    positions = _random_positions(num_positions=max(1, int(samples)), seed=123)

    l1_list: List[float] = []
    value_abs_list: List[float] = []
    move_match = 0

    for board, mark in positions:
        p_ref, v_ref = predict_policy_value(model, board, mark, device=device)
        p_sub, v_sub = mod.predict_policy_value_from_board(board, mark)
        p_ref = np.asarray(p_ref, dtype=np.float64)
        p_sub = np.asarray(p_sub, dtype=np.float64)
        l1 = float(np.mean(np.abs(p_ref - p_sub)))
        l1_list.append(l1)
        value_abs_list.append(abs(float(v_ref) - float(v_sub)))

        move_ref = int(np.argmax(p_ref))
        move_sub = int(np.argmax(p_sub))
        if move_ref == move_sub:
            move_match += 1

    result = {
        "policy_l1_mean": float(np.mean(l1_list) if l1_list else 0.0),
        "policy_l1_p95": float(np.percentile(np.asarray(l1_list), 95) if l1_list else 0.0),
        "value_abs_mean": float(np.mean(value_abs_list) if value_abs_list else 0.0),
        "value_abs_p95": float(np.percentile(np.asarray(value_abs_list), 95) if value_abs_list else 0.0),
        "argmax_match_rate": float(move_match / max(1, len(positions))),
    }
    return result


def validate_matchups(
    submission_path: Path,
    games: int,
    simulations: int,
    seed: int,
) -> Dict[str, Dict[str, Any]]:
    from azlite import evaluate as eval_mod

    sub_mod = _load_py_module(submission_path, "azlite_submission_validate_match")
    sub_agent = eval_mod.UnifiedAgent(
        name="azlite_submission",
        fn=lambda obs, cfg: int(sub_mod.agent(obs, cfg)),
        description="built single-file submission agent",
    )

    opponents = ["random", "negamax", "original"]
    results: Dict[str, Dict[str, Any]] = {}
    for i, spec in enumerate(opponents):
        opp = eval_mod.create_agent(
            spec=spec,
            simulations=int(simulations),
            device="cpu",
            seed=int(seed) + i + 1,
        )
        r = eval_mod.play_match(
            sub_agent,
            opp,
            num_games=max(2, int(games)),
            swap_sides=True,
            seed=int(seed) + 100 + i,
        )
        r["opponent"] = spec
        results[spec] = r
    return results


def _print_matchup_result(result: Dict[str, Any]) -> None:
    print(
        f"[build_azlite] vs {result['opponent']}: "
        f"{result['wins']}W/{result['losses']}L/{result['draws']}D "
        f"WR={float(result['win_rate'])*100:.1f}% "
        f"avg_step_ms={float(result['avg_step_time_sec']['agent_a'])*1000:.1f} "
        f"p95_step_ms={float(result['p95_step_time_sec']['agent_a'])*1000:.1f} "
        f"illegal={int(result['illegal_actions']['agent_a'])}"
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Build Kaggle-ready AlphaZero-lite submission.py")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path (required unless --weights-module already exists)")
    parser.add_argument("--weights-module", type=str, default=str(DEFAULT_WEIGHTS_MODULE), help="Export module path, e.g. agents/azlite_weights.py")
    parser.add_argument("--output", type=str, default=str(ROOT / "submission.py"))
    parser.add_argument("--mode", choices=("numpy", "torch"), default="numpy")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--inarow", type=int, default=4)
    parser.add_argument("--no-compress", action="store_true")

    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--simulations-opening", type=int, default=28)
    parser.add_argument("--simulations-midgame", type=int, default=46)
    parser.add_argument("--simulations-endgame", type=int, default=72)
    parser.add_argument("--simulations-min", type=int, default=12)
    parser.add_argument("--simulations-max", type=int, default=100)
    parser.add_argument("--time-budget-sec", type=float, default=1.80)
    parser.add_argument("--time-budget-max-sec", type=float, default=1.94)

    parser.add_argument("--validate", action="store_true", help="Run validation after build")
    parser.add_argument("--validate-samples", type=int, default=24)
    parser.add_argument("--validate-games", type=int, default=20)
    parser.add_argument("--validate-simulations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    weights_module_path = Path(args.weights_module)
    output_path = Path(args.output)

    if args.checkpoint:
        payload = export_checkpoint(
            checkpoint_path=Path(args.checkpoint),
            output_path=weights_module_path,
            rows=int(args.rows),
            columns=int(args.columns),
            inarow=int(args.inarow),
            runtime_mode=str(args.mode),
            dtype=str(args.dtype),
            compress=not args.no_compress,
        )
    else:
        if not weights_module_path.exists():
            raise RuntimeError("--checkpoint is required when --weights-module does not exist")
        payload = _load_export_payload(weights_module_path)

    payload = _ensure_runtime_cfg(payload, args)
    build_submission_file(payload, output_path)
    _smoke_validate_submission_module(output_path)

    print(f"[build_azlite] weights_module={weights_module_path.resolve()}")
    print(f"[build_azlite] output={output_path.resolve()}")
    print(
        f"[build_azlite] mode={args.mode} channels={payload.get('channels')} "
        f"rows={payload.get('rows')} cols={payload.get('columns')} inarow={payload.get('inarow')}"
    )

    if args.validate:
        align = None
        if args.checkpoint:
            align = validate_policy_value_alignment(
                checkpoint_path=Path(args.checkpoint),
                submission_path=output_path,
                samples=int(args.validate_samples),
                device=str(args.device),
            )
        if align is not None:
            print(
                "[build_azlite] alignment: "
                + json.dumps(align, ensure_ascii=False)
            )

        matchups = validate_matchups(
            submission_path=output_path,
            games=int(args.validate_games),
            simulations=int(args.validate_simulations),
            seed=int(args.seed),
        )
        for key in ("random", "negamax", "original"):
            if key in matchups:
                _print_matchup_result(matchups[key])


if __name__ == "__main__":
    _main()
