"""AlphaZero-lite training loop: self-play + replay buffer + optimization.

Entry:
  python -m azlite.train
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from azlite import eval_core
from azlite.board import find_immediate_block, find_immediate_win, legal_moves, obs_board_to_numpy
from azlite.model import ConnectXNet, NeuralEvaluator, save_checkpoint
from azlite.puct_mcts import run_mcts
from azlite.replay_buffer import ReplayBuffer
from azlite.self_play import generate_self_play_games


VALUE_LOSS_WEIGHT = 1.0
GRAD_CLIP_MAX_NORM = 5.0
VALUE_SAT_THRESH = 0.999
EPS = 1e-12
COMPOSITE_WEIGHTS: Dict[str, float] = {
    "random": 0.03,
    "negamax": 0.42,
    "mcts_lite": 0.30,
    "previous_best": 0.25,
}
COMPOSITE_KEY_METRICS = ("negamax", "mcts_lite", "previous_best")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_metadata(raw) -> Dict[str, object]:
    try:
        if isinstance(raw, np.ndarray):
            if raw.shape == ():
                raw = raw.item()
            elif raw.size == 1:
                raw = raw.reshape(-1)[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            return json.loads(raw)
    except Exception:
        pass
    return {}


@dataclass
class EvalResult:
    wins: int
    losses: int
    draws: int
    win_rate: float
    first_player_win_rate: float
    second_player_win_rate: float
    candidate_timeouts: int = 0
    opponent_timeouts: int = 0
    candidate_invalid_actions: int = 0
    opponent_invalid_actions: int = 0
    candidate_timeout_rate: float = 0.0
    opponent_timeout_rate: float = 0.0
    reliable: bool = True
    unreliable_reasons: Optional[list[str]] = None
    reliability_warning: Optional[str] = None


def _mcts_model_agent(model: ConnectXNet, device: str, simulations: int):
    evaluator = NeuralEvaluator(model, device=device)

    def _agent(observation, configuration):
        board = obs_board_to_numpy(
            observation.board,
            rows=int(configuration.rows),
            columns=int(configuration.columns),
        )
        valid = legal_moves(board)
        if not valid:
            return 0
        # Keep tactical guards at battle time.
        win_col = find_immediate_win(board, int(observation.mark))
        if win_col is not None:
            return int(win_col)
        opp = 2 if int(observation.mark) == 1 else 1
        block_col = find_immediate_block(board, int(observation.mark), opp)
        if block_col is not None:
            return int(block_col)

        out = run_mcts(
            board=board,
            current_player=int(observation.mark),
            evaluator=evaluator,
            num_simulations=max(1, int(simulations)),
            c_puct=1.5,
            temperature=0.0,
            add_dirichlet_noise=False,
            use_tactical_shortcuts=True,
            return_root=False,
        )
        move = int(out["move"])
        if move in valid:
            return move
        return int(valid[0])

    return _agent


def _evaluate_vs(
    agent_a,
    agent_b,
    games: int = 16,
    seed: Optional[int] = None,
    simulations: int = 100,
    device: str = "cpu",
    candidate_timeout_ms: float = 2000.0,
    opponent_timeout_ms: float = 4000.0,
    eval_profile: str = "quick",
    opponent_time_budget_ms: Optional[float] = None,
    max_opponent_timeout_rate: float = 0.05,
    max_candidate_timeout_rate: float = 0.01,
    timeout_result_policy: str = "fail_eval",
) -> EvalResult:
    candidate = eval_core.as_unified_agent(agent_a, default_name="train_candidate")
    if isinstance(agent_b, str):
        opponent = eval_core.create_agent(
            agent_b,
            simulations=int(simulations),
            device=str(device),
            seed=seed,
            time_budget_ms=opponent_time_budget_ms,
        )
    else:
        opponent = eval_core.as_unified_agent(agent_b, default_name="train_opponent")

    match = eval_core.play_match(
        candidate,
        opponent,
        num_games=int(games),
        swap_sides=True,
        seed=seed,
        candidate_timeout_ms=float(candidate_timeout_ms),
        opponent_timeout_ms=float(opponent_timeout_ms),
        eval_profile=str(eval_profile),
        max_opponent_timeout_rate=float(max_opponent_timeout_rate),
        max_candidate_timeout_rate=float(max_candidate_timeout_rate),
        timeout_result_policy=str(timeout_result_policy),
    )

    reliability = match.get("reliability") or {}
    invalid = match.get("invalid_actions") or {}
    return EvalResult(
        wins=int(match["wins"]),
        losses=int(match["losses"]),
        draws=int(match["draws"]),
        win_rate=float(match["win_rate"]),
        first_player_win_rate=float(match.get("first_player_win_rate", 0.0)),
        second_player_win_rate=float(match.get("second_player_win_rate", 0.0)),
        candidate_timeouts=int(match.get("candidate_timeouts", match["timeouts"]["agent_a"])),
        opponent_timeouts=int(match.get("opponent_timeouts", match["timeouts"]["agent_b"])),
        candidate_invalid_actions=int(invalid.get("candidate", match["illegal_actions"]["agent_a"])),
        opponent_invalid_actions=int(invalid.get("opponent", match["illegal_actions"]["agent_b"])),
        candidate_timeout_rate=float(match.get("candidate_timeout_rate", 0.0)),
        opponent_timeout_rate=float(match.get("opponent_timeout_rate", 0.0)),
        reliable=bool(match.get("reliable", reliability.get("reliable", True))),
        unreliable_reasons=list(match.get("unreliable_reasons") or []),
        reliability_warning=reliability.get("warning"),
    )


def _format_eval(name: str, result: Optional[EvalResult]) -> str:
    if result is None:
        return f"{name}: skipped"
    msg = (
        f"{name}: {result.wins}W/{result.losses}L/{result.draws}D "
        f"WR={result.win_rate*100:.1f}% "
        f"FP={result.first_player_win_rate*100:.1f}% "
        f"SP={result.second_player_win_rate*100:.1f}%"
    )
    msg += (
        f" TO(cand/opp)={result.candidate_timeouts}/{result.opponent_timeouts} "
        f"IL(cand/opp)={result.candidate_invalid_actions}/{result.opponent_invalid_actions} "
        f"TO_RATE(cand/opp)={result.candidate_timeout_rate*100:.1f}%/{result.opponent_timeout_rate*100:.1f}%"
    )
    if (not result.reliable) and (result.unreliable_reasons or result.reliability_warning):
        reasons = "; ".join(result.unreliable_reasons or [])
        if not reasons and result.reliability_warning:
            reasons = str(result.reliability_warning)
        msg += f" [UNRELIABLE: {reasons}]"
    return msg


def _soft_target_ce_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_probs * log_probs).sum(dim=1).mean()


def _load_model_and_optimizer(
    checkpoint_path: Optional[Path],
    device: str,
    lr: float,
) -> Tuple[ConnectXNet, torch.optim.Optimizer, Dict[str, object]]:
    if checkpoint_path is None:
        model = ConnectXNet(in_channels=3).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
        return model, optimizer, {}

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    model_cfg = ckpt.get("model_config", {})
    in_channels = int(model_cfg.get("in_channels", 3))
    model = ConnectXNet(in_channels=in_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    opt_state = ckpt.get("optimizer_state_dict")
    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except Exception:
            pass
    for group in optimizer.param_groups:
        group["lr"] = float(lr)
    return model, optimizer, dict(ckpt.get("metadata", {}))


def _check_policy_targets(states: np.ndarray, policies: np.ndarray, iteration: int, step: int) -> None:
    if np.isnan(policies).any():
        raise RuntimeError(f"[iter={iteration} step={step}] policy_target has NaN")
    row_sums = np.sum(policies, axis=1)
    zero_rows = np.where(row_sums <= EPS)[0]
    if zero_rows.size > 0:
        sample = zero_rows[:10].tolist()
        raise RuntimeError(
            f"[iter={iteration} step={step}] policy_target has all-zero rows, "
            f"indices(sample)={sample}"
        )

    # Illegal-column probability check from occupancy on top row:
    # legal col iff top cell is empty.
    if states.shape[1] < 2:
        return

    occ_top = states[:, 0, 0, :] + states[:, 1, 0, :]
    legal_mask = occ_top < 0.5
    illegal_mask = ~legal_mask
    bad_rows = []
    for i in range(policies.shape[0]):
        if np.any(policies[i][illegal_mask[i]] > 1e-6):
            bad_rows.append(i)
            if len(bad_rows) >= 10:
                break
    if bad_rows:
        raise RuntimeError(
            f"[iter={iteration} step={step}] illegal columns have target probability > 0, "
            f"rows(sample)={bad_rows}"
        )


def _train_steps(
    model: ConnectXNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    train_steps: int,
    device: str,
    iteration: int,
) -> Dict[str, float]:
    model.train()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    batch_count = 0

    for step in range(1, int(train_steps) + 1):
        states, policies, values = replay.sample_batch(batch_size)

        if np.isnan(values).any():
            raise RuntimeError(f"[iter={iteration} step={step}] value target has NaN")
        _check_policy_targets(states, policies, iteration, step)

        xb = torch.from_numpy(states).to(device=device, dtype=torch.float32)
        pib = torch.from_numpy(policies).to(device=device, dtype=torch.float32)
        vb = torch.from_numpy(values).to(device=device, dtype=torch.float32)

        logits, value_pred = model(xb)
        value_pred = value_pred.squeeze(1)

        if torch.all(value_pred > VALUE_SAT_THRESH):
            raise RuntimeError(
                f"[iter={iteration} step={step}] model value saturated near +1 for full batch"
            )
        if torch.all(value_pred < -VALUE_SAT_THRESH):
            raise RuntimeError(
                f"[iter={iteration} step={step}] model value saturated near -1 for full batch"
            )
        if torch.all(torch.abs(value_pred) > VALUE_SAT_THRESH):
            raise RuntimeError(
                f"[iter={iteration} step={step}] model value saturated near +/-1 for full batch"
            )

        policy_loss = _soft_target_ce_loss(logits, pib)
        value_loss = F.mse_loss(value_pred, vb)
        loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

        if torch.isnan(loss):
            raise RuntimeError(f"[iter={iteration} step={step}] loss became NaN")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()

        total_policy += float(policy_loss.item())
        total_value += float(value_loss.item())
        total_loss += float(loss.item())
        batch_count += 1

    return {
        "policy_loss": total_policy / max(1, batch_count),
        "value_loss": total_value / max(1, batch_count),
        "total_loss": total_loss / max(1, batch_count),
    }


def _build_composite_details(
    eval_random: Optional[EvalResult],
    eval_negamax: Optional[EvalResult],
    eval_mcts_lite: Optional[EvalResult],
    eval_prev_best: Optional[EvalResult],
) -> Dict[str, object]:
    mapping = {
        "random": eval_random,
        "negamax": eval_negamax,
        "mcts_lite": eval_mcts_lite,
        "previous_best": eval_prev_best,
    }
    components: Dict[str, Dict[str, object]] = {}
    included_weight_sum = 0.0
    for metric, weight in COMPOSITE_WEIGHTS.items():
        r = mapping.get(metric)
        present = r is not None
        reliable = bool(r.reliable) if present else False
        included = bool(present and reliable)
        if included:
            included_weight_sum += float(weight)
        components[metric] = {
            "configured_weight": float(weight),
            "normalized_weight": 0.0,
            "present": bool(present),
            "reliable": bool(reliable),
            "included": bool(included),
            "win_rate": float(r.win_rate) if present else None,
            "status": (
                "included"
                if included
                else ("unreliable" if present else "missing")
            ),
        }

    score = None
    if included_weight_sum > 0:
        score = 0.0
        for metric, comp in components.items():
            if not bool(comp["included"]):
                continue
            norm_w = float(comp["configured_weight"]) / float(included_weight_sum)
            comp["normalized_weight"] = norm_w
            score += norm_w * float(comp["win_rate"])

    all_key_unreliable = all(
        (mapping.get(metric) is None) or (not bool(mapping[metric].reliable))
        for metric in COMPOSITE_KEY_METRICS
    )
    return {
        "score": score,
        "weights": dict(COMPOSITE_WEIGHTS),
        "components": components,
        "included_weight_sum": float(included_weight_sum),
        "all_key_metrics_unreliable": bool(all_key_unreliable),
        "key_metrics": list(COMPOSITE_KEY_METRICS),
    }


def _should_promote_to_best(
    eval_random: Optional[EvalResult],
    eval_negamax: Optional[EvalResult],
    eval_mcts_lite: Optional[EvalResult],
    eval_prev_best: Optional[EvalResult],
    best_metrics: Dict[str, float],
) -> Tuple[bool, str, float, Dict[str, object]]:
    composite_details = _build_composite_details(
        eval_random=eval_random,
        eval_negamax=eval_negamax,
        eval_mcts_lite=eval_mcts_lite,
        eval_prev_best=eval_prev_best,
    )
    score_raw = composite_details.get("score")
    score = float(score_raw) if score_raw is not None else float("nan")
    best_score = float(best_metrics.get("best_composite", -1.0))
    best_negamax = float(best_metrics.get("best_negamax_win_rate", -1.0))
    best_mcts = float(best_metrics.get("best_mcts_lite_win_rate", -1.0))

    if bool(composite_details.get("all_key_metrics_unreliable", False)):
        return False, "all key metrics unreliable/missing", score, composite_details

    if eval_negamax is None:
        return False, "missing negamax eval", score, composite_details
    if not eval_negamax.reliable:
        return False, "negamax matchup unreliable due opponent timeouts", score, composite_details

    if eval_mcts_lite is not None and (not eval_mcts_lite.reliable):
        return False, "mcts_lite matchup unreliable due opponent timeouts", score, composite_details

    # Guardrails: random is only sanity-check; rely on stronger opponents.
    if eval_prev_best is not None and eval_prev_best.win_rate < 0.5:
        return False, "failed vs previous best (<50%)", score, composite_details
    if eval_prev_best is not None and (not eval_prev_best.reliable):
        return False, "previous-best matchup unreliable due opponent timeouts", score, composite_details
    if best_negamax >= 0.0 and eval_negamax.win_rate + 1e-9 < (best_negamax - 0.03):
        return False, "negamax regressed too much", score, composite_details
    if eval_mcts_lite is not None and best_mcts >= 0.0 and eval_mcts_lite.win_rate + 1e-9 < (best_mcts - 0.03):
        return False, "mcts_lite regressed too much", score, composite_details
    if math.isnan(score):
        return False, "composite unavailable (no reliable component)", score, composite_details
    if score <= best_score + 1e-9:
        return False, "composite score not improved", score, composite_details
    return True, "composite improved with stability checks", score, composite_details


def _load_train_state(state_path: Path) -> Dict[str, object]:
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_train_state(state_path: Path, state: Dict[str, object]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_teacher_data_list(spec: Optional[str]) -> list[str]:
    if not spec:
        return []
    return [p.strip() for p in str(spec).split(",") if p.strip()]


def _load_selfplay_metadata(npz_path: Path) -> Dict[str, object]:
    data = np.load(str(npz_path), allow_pickle=True)
    return _parse_metadata(data["metadata"] if "metadata" in data else None)


def _main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero-lite training loop")
    parser.add_argument("--checkpoint", type=str, default=None, help="Initial model checkpoint")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--self-play-games", type=int, default=6)
    parser.add_argument("--simulations", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/azlite_train")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--teacher-data",
        type=str,
        default=None,
        help="Optional NPZ path(s), comma-separated, loaded into replay buffer before training",
    )
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-act-timeout-sec", type=float, default=eval_core.KAGGLE_ACT_TIMEOUT_SEC)
    parser.add_argument("--max-opponent-timeout-rate", type=float, default=0.05)
    parser.add_argument("--max-candidate-timeout-rate", type=float, default=0.01)
    parser.add_argument(
        "--timeout-result-policy",
        type=str,
        choices=("loss", "exclude", "fail_eval"),
        default="fail_eval",
    )
    parser.add_argument(
        "--eval-profile",
        type=str,
        choices=("quick", "strong_local", "kaggle_like"),
        default="quick",
        help="Training quick-eval profile (default quick)",
    )
    parser.add_argument("--candidate-timeout-ms", type=float, default=None)
    parser.add_argument("--opponent-timeout-ms", type=float, default=None)
    parser.add_argument("--eval-games-random", type=int, default=None)
    parser.add_argument("--eval-games-negamax", type=int, default=None)
    parser.add_argument("--eval-games-mcts-lite", type=int, default=None)
    parser.add_argument("--eval-games-previous-best", type=int, default=None)
    parser.add_argument(
        "--eval-mcts-lite-every",
        type=int,
        default=2,
        help="Evaluate vs mcts_lite every N eval rounds (1 means every round).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    legacy_timeout_override_ms = None
    if abs(float(args.eval_act_timeout_sec) - float(eval_core.KAGGLE_ACT_TIMEOUT_SEC)) > 1e-9:
        legacy_timeout_override_ms = float(args.eval_act_timeout_sec) * 1000.0

    eval_cfg = eval_core.resolve_eval_config(
        eval_profile=str(args.eval_profile),
        base_games=int(args.eval_games),
        candidate_timeout_ms=(
            args.candidate_timeout_ms
            if args.candidate_timeout_ms is not None
            else legacy_timeout_override_ms
        ),
        opponent_timeout_ms=args.opponent_timeout_ms,
        eval_games_random=args.eval_games_random,
        eval_games_negamax=args.eval_games_negamax,
        eval_games_mcts_lite=args.eval_games_mcts_lite,
        eval_games_previous_best=args.eval_games_previous_best,
    )

    _set_seed(int(args.seed))
    device = str(args.device)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest.pt"
    best_ckpt_path = ckpt_dir / "best.pt"
    replay_path = ckpt_dir / "replay_buffer_latest.npz"
    state_path = ckpt_dir / "train_state.json"
    selfplay_output_dir = ckpt_dir / "selfplay_npz"

    state = _load_train_state(state_path)
    best_metrics = {
        "best_composite": float(state.get("best_composite", -1.0)),
        "best_negamax_win_rate": float(state.get("best_negamax_win_rate", -1.0)),
        "best_mcts_lite_win_rate": float(state.get("best_mcts_lite_win_rate", -1.0)),
    }
    best_composite_details = state.get("best_composite_details")
    best_iteration = int(state.get("best_iteration", 0) or 0)

    # 1) Load model + optimizer
    if args.resume and latest_ckpt_path.exists():
        model, optimizer, ckpt_meta = _load_model_and_optimizer(
            latest_ckpt_path, device=device, lr=float(args.lr)
        )
        print(f"[train] resumed model from latest checkpoint: {latest_ckpt_path.resolve()}")
    elif args.checkpoint:
        model, optimizer, ckpt_meta = _load_model_and_optimizer(
            Path(args.checkpoint), device=device, lr=float(args.lr)
        )
        print(f"[train] loaded initial checkpoint: {Path(args.checkpoint).resolve()}")
    else:
        model, optimizer, ckpt_meta = _load_model_and_optimizer(
            None, device=device, lr=float(args.lr)
        )
        print("[train] initialized new model")
    del ckpt_meta

    # 2) Load replay buffer
    replay = ReplayBuffer(max_size=int(args.buffer_size))
    if args.resume and replay_path.exists():
        replay.load(replay_path)
        print(f"[train] resumed replay buffer: size={len(replay)}")

    teacher_paths = _parse_teacher_data_list(args.teacher_data)
    for td in teacher_paths:
        added = replay.add_npz(td)
        print(f"[train] loaded teacher data: {td}  added={added}  buffer={len(replay)}")

    start_iteration = 1
    if args.resume:
        last_it = int(state.get("last_iteration", 0))
        start_iteration = last_it + 1
    end_iteration = start_iteration + int(args.iterations) - 1

    print(
        f"[train] iterations {start_iteration} -> {end_iteration}, "
        f"self_play_games={args.self_play_games}, sims={args.simulations}, "
        f"batch={args.batch_size}, train_steps={args.train_steps}"
    )
    print(f"[train] evaluator_backend={eval_core.EVALUATOR_BACKEND}")
    print(
        "[train] eval_profile={profile} candidate_timeout_ms={cand:.0f} opponent_timeout_ms={opp:.0f} "
        "games(random/negamax/mcts/previous_best)={gr}/{gn}/{gm}/{gp}".format(
            profile=eval_cfg["eval_profile"],
            cand=float(eval_cfg["candidate_timeout_ms"]),
            opp=float(eval_cfg["opponent_timeout_ms"]),
            gr=eval_core.games_for_opponent(eval_cfg, "random"),
            gn=eval_core.games_for_opponent(eval_cfg, "negamax"),
            gm=eval_core.games_for_opponent(eval_cfg, "mcts_lite"),
            gp=eval_core.games_for_opponent(eval_cfg, "previous_best"),
        )
    )

    for iteration in range(start_iteration, end_iteration + 1):
        print("\n" + "=" * 72)
        print(f"[train] iteration {iteration}")
        print("=" * 72)

        # 2) generate self-play games
        model.eval()
        selfplay_npz = generate_self_play_games(
            model=model,
            num_games=int(args.self_play_games),
            num_simulations=int(args.simulations),
            device=device,
            augment=True,
            output_dir=str(selfplay_output_dir),
            c_puct=1.5,
            add_dirichlet_noise=True,
            use_tactical_shortcuts=False,
        )
        selfplay_meta = _load_selfplay_metadata(selfplay_npz)
        avg_game_length = float(selfplay_meta.get("avg_game_length", 0.0))
        winner_distribution = selfplay_meta.get("winner_distribution", {})

        # 3) add to replay buffer
        added = replay.add_npz(selfplay_npz)
        print(f"[train] self-play added={added}, buffer size={len(replay)}")

        if len(replay) == 0:
            raise RuntimeError("Replay buffer is empty after self-play generation")

        # 4) train
        metrics = _train_steps(
            model=model,
            optimizer=optimizer,
            replay=replay,
            batch_size=int(args.batch_size),
            train_steps=int(args.train_steps),
            device=device,
            iteration=iteration,
        )

        # 5) save latest checkpoint + replay snapshot
        latest_meta = {
            "iteration": int(iteration),
            "train_steps": int(args.train_steps),
            "simulations": int(args.simulations),
            "eval_results": {},
            "eval_profile": str(eval_cfg["eval_profile"]),
            "candidate_timeout_ms": float(eval_cfg["candidate_timeout_ms"]),
            "opponent_timeout_ms": float(eval_cfg["opponent_timeout_ms"]),
            "buffer_size": int(len(replay)),
            "loss": metrics,
            "self_play_avg_game_length": avg_game_length,
            "self_play_winner_distribution": winner_distribution,
            "created_at": _utc_now_iso(),
        }
        save_checkpoint(model, optimizer, latest_ckpt_path, metadata=latest_meta)
        replay.save(replay_path)

        eval_random = None
        eval_negamax = None
        eval_mcts_lite = None
        eval_prev_best = None
        gate_passed = False
        gate_reason = "evaluation skipped"
        composite = float("nan")
        composite_details: Dict[str, object] = {}

        # 6) quick evaluation
        if int(args.eval_interval) > 0 and (iteration % int(args.eval_interval) == 0):
            model.eval()
            current_agent = _mcts_model_agent(model, device=device, simulations=int(args.simulations))
            eval_random = _evaluate_vs(
                current_agent,
                "random",
                games=eval_core.games_for_opponent(eval_cfg, "random"),
                seed=int(args.seed) + iteration * 100 + 1,
                simulations=int(args.simulations),
                device=device,
                candidate_timeout_ms=float(eval_cfg["candidate_timeout_ms"]),
                opponent_timeout_ms=float(eval_cfg["opponent_timeout_ms"]),
                eval_profile=str(eval_cfg["eval_profile"]),
                opponent_time_budget_ms=float(eval_cfg["opponent_timeout_ms"]),
                max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
                max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
                timeout_result_policy=str(args.timeout_result_policy),
            )
            eval_negamax = _evaluate_vs(
                current_agent,
                "negamax",
                games=eval_core.games_for_opponent(eval_cfg, "negamax"),
                seed=int(args.seed) + iteration * 100 + 2,
                simulations=int(args.simulations),
                device=device,
                candidate_timeout_ms=float(eval_cfg["candidate_timeout_ms"]),
                opponent_timeout_ms=float(eval_cfg["opponent_timeout_ms"]),
                eval_profile=str(eval_cfg["eval_profile"]),
                opponent_time_budget_ms=float(eval_cfg["opponent_timeout_ms"]),
                max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
                max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
                timeout_result_policy=str(args.timeout_result_policy),
            )
            mcts_every = max(1, int(args.eval_mcts_lite_every))
            if iteration % mcts_every == 0:
                eval_mcts_lite = _evaluate_vs(
                    current_agent,
                    "mcts_lite",
                    games=eval_core.games_for_opponent(eval_cfg, "mcts_lite"),
                    seed=int(args.seed) + iteration * 100 + 4,
                    simulations=int(args.simulations),
                    device=device,
                    candidate_timeout_ms=float(eval_cfg["candidate_timeout_ms"]),
                    opponent_timeout_ms=float(eval_cfg["opponent_timeout_ms"]),
                    eval_profile=str(eval_cfg["eval_profile"]),
                    opponent_time_budget_ms=float(eval_cfg["opponent_timeout_ms"]),
                    max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
                    max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
                    timeout_result_policy=str(args.timeout_result_policy),
                )

            if best_ckpt_path.exists():
                best_model, _, _ = _load_model_and_optimizer(best_ckpt_path, device=device, lr=float(args.lr))
                best_agent = _mcts_model_agent(best_model, device=device, simulations=int(args.simulations))
                eval_prev_best = _evaluate_vs(
                    current_agent,
                    best_agent,
                    games=max(8, eval_core.games_for_opponent(eval_cfg, "previous_best")),
                    seed=int(args.seed) + iteration * 100 + 3,
                    simulations=int(args.simulations),
                    device=device,
                    candidate_timeout_ms=float(eval_cfg["candidate_timeout_ms"]),
                    opponent_timeout_ms=float(eval_cfg["opponent_timeout_ms"]),
                    eval_profile=str(eval_cfg["eval_profile"]),
                    max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
                    max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
                    timeout_result_policy=str(args.timeout_result_policy),
                )

            # 7) gating for best checkpoint
            gate_passed, gate_reason, composite, composite_details = _should_promote_to_best(
                eval_random=eval_random,
                eval_negamax=eval_negamax,
                eval_mcts_lite=eval_mcts_lite,
                eval_prev_best=eval_prev_best,
                best_metrics=best_metrics,
            )
            if gate_passed:
                best_meta = dict(latest_meta)
                best_meta["eval_results"] = {
                    "evaluator_backend": eval_core.EVALUATOR_BACKEND,
                    "eval_profile": str(eval_cfg["eval_profile"]),
                    "candidate_timeout_ms": float(eval_cfg["candidate_timeout_ms"]),
                    "opponent_timeout_ms": float(eval_cfg["opponent_timeout_ms"]),
                    "random": eval_random.__dict__,
                    "negamax": eval_negamax.__dict__,
                    "mcts_lite": eval_mcts_lite.__dict__ if eval_mcts_lite else None,
                    "previous_best": eval_prev_best.__dict__ if eval_prev_best else None,
                    "composite_score": float(composite),
                    "composite": composite_details,
                    "gate_reason": gate_reason,
                }
                save_checkpoint(model, optimizer, best_ckpt_path, metadata=best_meta)
                best_metrics["best_composite"] = float(composite)
                best_metrics["best_negamax_win_rate"] = float(eval_negamax.win_rate)
                if eval_mcts_lite is not None:
                    best_metrics["best_mcts_lite_win_rate"] = float(eval_mcts_lite.win_rate)
                best_composite_details = composite_details
                best_iteration = int(iteration)

        # 5/6/7 logs
        print(
            f"[train] iteration={iteration} "
            f"buffer={len(replay)} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} "
            f"total_loss={metrics['total_loss']:.4f}"
        )
        print(
            f"[train] self-play avg_game_length={avg_game_length:.2f} "
            f"winner_distribution={winner_distribution}"
        )
        print("[train] " + _format_eval("eval vs random", eval_random))
        print("[train] " + _format_eval("eval vs negamax", eval_negamax))
        print("[train] " + _format_eval("eval vs mcts_lite", eval_mcts_lite))
        print("[train] " + _format_eval("eval vs previous best", eval_prev_best))
        if eval_random is not None:
            print(
                f"[train] gating passed={gate_passed} "
                f"reason={gate_reason} "
                f"composite={composite:.4f}"
            )
        print(f"[train] latest checkpoint: {latest_ckpt_path.resolve()}")
        if gate_passed:
            print(f"[train] best checkpoint:   {best_ckpt_path.resolve()}")

        latest_composite = None
        if eval_random is not None and (not math.isnan(float(composite))):
            latest_composite = float(composite)
        latest_negamax_win_rate = float(eval_negamax.win_rate) if eval_negamax is not None else None

        state.update(
            {
                "last_iteration": int(iteration),
                "best_iteration": int(best_iteration),
                "best_composite": float(best_metrics["best_composite"]),
                "best_negamax_win_rate": float(best_metrics["best_negamax_win_rate"]),
                "best_mcts_lite_win_rate": float(best_metrics["best_mcts_lite_win_rate"]),
                "best_composite_details": best_composite_details,
                "latest_composite": latest_composite,
                "latest_negamax_win_rate": latest_negamax_win_rate,
                "latest_mcts_lite_win_rate": (
                    float(eval_mcts_lite.win_rate) if eval_mcts_lite is not None else None
                ),
                "latest_composite_details": composite_details if composite_details else None,
                "latest_checkpoint": str(latest_ckpt_path.resolve()),
                "best_checkpoint": str(best_ckpt_path.resolve()) if best_ckpt_path.exists() else "",
                "eval_profile": str(eval_cfg["eval_profile"]),
                "candidate_timeout_ms": float(eval_cfg["candidate_timeout_ms"]),
                "opponent_timeout_ms": float(eval_cfg["opponent_timeout_ms"]),
                "eval_games_by_opponent": dict(eval_cfg.get("games_by_opponent", {})),
                "eval_mcts_lite_every": int(args.eval_mcts_lite_every),
                "buffer_size": int(len(replay)),
                "updated_at": _utc_now_iso(),
            }
        )
        _save_train_state(state_path, state)


if __name__ == "__main__":
    _main()
