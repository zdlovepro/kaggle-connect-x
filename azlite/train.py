"""AlphaZero-lite training loop: self-play + replay buffer + optimization.

Entry:
  python -m azlite.train
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
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
from azlite.runtime import configure_cpu_runtime, suggest_cpu_plan
from azlite.self_play import MCTS_TARGET_VERSION, generate_self_play_games


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
PREVIOUS_BEST_SIDE_BIAS_THRESHOLD = 0.40


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
    first_player_wins: int
    first_player_losses: int
    first_player_draws: int
    first_player_win_rate: float
    second_player_wins: int
    second_player_losses: int
    second_player_draws: int
    second_player_win_rate: float
    side_bias: float = 0.0
    candidate_timeout_moves: int = 0
    opponent_timeout_moves: int = 0
    candidate_timeout_games: int = 0
    opponent_timeout_games: int = 0
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
) -> Optional[EvalResult]:
    if int(games) <= 0:
        return None
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
        first_player_wins=int(match.get("first_player_wins", 0)),
        first_player_losses=int(match.get("first_player_losses", 0)),
        first_player_draws=int(match.get("first_player_draws", 0)),
        first_player_win_rate=float(match.get("first_player_win_rate", 0.0)),
        second_player_wins=int(match.get("second_player_wins", 0)),
        second_player_losses=int(match.get("second_player_losses", 0)),
        second_player_draws=int(match.get("second_player_draws", 0)),
        second_player_win_rate=float(match.get("second_player_win_rate", 0.0)),
        side_bias=float(match.get("side_bias", 0.0)),
        candidate_timeout_moves=int(match.get("candidate_timeout_moves", match.get("candidate_timeouts", 0))),
        opponent_timeout_moves=int(match.get("opponent_timeout_moves", match.get("opponent_timeouts", 0))),
        candidate_timeout_games=int(match.get("candidate_timeout_games", 0)),
        opponent_timeout_games=int(match.get("opponent_timeout_games", 0)),
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
        f"FP={result.first_player_wins}/{result.first_player_losses}/{result.first_player_draws} "
        f"({result.first_player_win_rate*100:.1f}%) "
        f"SP={result.second_player_wins}/{result.second_player_losses}/{result.second_player_draws} "
        f"({result.second_player_win_rate*100:.1f}%) "
        f"BIAS={result.side_bias*100:.1f}%"
    )
    msg += (
        f" TO_GAME(cand/opp)={result.candidate_timeout_games}/{result.opponent_timeout_games} "
        f"TO_MOVE(cand/opp)={result.candidate_timeout_moves}/{result.opponent_timeout_moves} "
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


def _compute_teacher_batch_ratio(
    iteration: int,
    start_iteration: int,
    end_iteration: int,
    ratio_start: float,
    ratio_end: float,
) -> float:
    hi = max(0.0, min(1.0, float(ratio_start)))
    lo = max(0.0, min(1.0, float(ratio_end)))
    if end_iteration <= start_iteration:
        return hi
    progress = (int(iteration) - int(start_iteration)) / max(1, int(end_iteration) - int(start_iteration))
    progress = max(0.0, min(1.0, float(progress)))
    return float(hi + (lo - hi) * progress)


def _sample_training_batch(
    teacher_replay: ReplayBuffer,
    selfplay_replay: ReplayBuffer,
    batch_size: int,
    teacher_ratio: float,
):
    batch = max(1, int(batch_size))
    teacher_n = 0
    selfplay_n = 0

    if len(teacher_replay) > 0 and len(selfplay_replay) > 0:
        teacher_n = int(round(batch * max(0.0, min(1.0, float(teacher_ratio)))))
        teacher_n = max(0, min(batch, teacher_n))
        selfplay_n = batch - teacher_n
        if selfplay_n <= 0:
            selfplay_n = 1
            teacher_n = batch - 1
        if teacher_n <= 0:
            teacher_n = 1
            selfplay_n = batch - 1
    elif len(selfplay_replay) > 0:
        selfplay_n = batch
    elif len(teacher_replay) > 0:
        teacher_n = batch
    else:
        raise ValueError("Both teacher and self-play replay buffers are empty")

    states_parts = []
    policies_parts = []
    values_parts = []
    if teacher_n > 0:
        s, p, v = teacher_replay.sample_batch(teacher_n)
        states_parts.append(s)
        policies_parts.append(p)
        values_parts.append(v)
    if selfplay_n > 0:
        s, p, v = selfplay_replay.sample_batch(selfplay_n)
        states_parts.append(s)
        policies_parts.append(p)
        values_parts.append(v)

    states = np.concatenate(states_parts, axis=0).astype(np.float32, copy=False)
    policies = np.concatenate(policies_parts, axis=0).astype(np.float32, copy=False)
    values = np.concatenate(values_parts, axis=0).astype(np.float32, copy=False)
    order = np.random.permutation(states.shape[0])
    return (
        states[order],
        policies[order],
        values[order],
        teacher_n,
        selfplay_n,
    )


def _train_steps(
    model: ConnectXNet,
    optimizer: torch.optim.Optimizer,
    teacher_replay: ReplayBuffer,
    selfplay_replay: ReplayBuffer,
    batch_size: int,
    train_steps: int,
    device: str,
    iteration: int,
    teacher_batch_ratio: float,
) -> Dict[str, float]:
    model.train()
    total_policy = 0.0
    total_value = 0.0
    total_loss = 0.0
    batch_count = 0
    total_teacher = 0
    total_selfplay = 0

    for step in range(1, int(train_steps) + 1):
        states, policies, values, teacher_n, selfplay_n = _sample_training_batch(
            teacher_replay=teacher_replay,
            selfplay_replay=selfplay_replay,
            batch_size=batch_size,
            teacher_ratio=teacher_batch_ratio,
        )

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
        total_teacher += int(teacher_n)
        total_selfplay += int(selfplay_n)

    return {
        "policy_loss": total_policy / max(1, batch_count),
        "value_loss": total_value / max(1, batch_count),
        "total_loss": total_loss / max(1, batch_count),
        "teacher_examples_seen": int(total_teacher),
        "selfplay_examples_seen": int(total_selfplay),
        "teacher_batch_ratio_actual": (
            float(total_teacher) / max(1.0, float(total_teacher + total_selfplay))
        ),
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
    previous_best_required: bool = False,
) -> Tuple[bool, str, float, Dict[str, object]]:
    ignore_prev_best_due_side_bias = False
    prev_bias_note = ""
    if eval_prev_best is not None and float(eval_prev_best.side_bias) > PREVIOUS_BEST_SIDE_BIAS_THRESHOLD:
        if abs(float(eval_prev_best.win_rate) - 0.5) <= 0.05:
            ignore_prev_best_due_side_bias = True
            prev_bias_note = (
                "previous-best WR~50% with extreme FP/SP side bias; "
                "ignored due extreme side bias"
            )
    eval_prev_best_for_scoring = None if ignore_prev_best_due_side_bias else eval_prev_best
    composite_details = _build_composite_details(
        eval_random=eval_random,
        eval_negamax=eval_negamax,
        eval_mcts_lite=eval_mcts_lite,
        eval_prev_best=eval_prev_best_for_scoring,
    )
    def _with_bias_note(reason: str) -> str:
        return f"{reason}; {prev_bias_note}" if prev_bias_note else reason
    score_raw = composite_details.get("score")
    score = float(score_raw) if score_raw is not None else float("nan")
    best_score = float(best_metrics.get("best_composite", -1.0))
    best_negamax = float(best_metrics.get("best_negamax_win_rate", -1.0))
    best_mcts = float(best_metrics.get("best_mcts_lite_win_rate", -1.0))

    if bool(composite_details.get("all_key_metrics_unreliable", False)):
        return False, _with_bias_note("all key metrics unreliable/missing"), score, composite_details

    if eval_negamax is not None and (not eval_negamax.reliable):
        return False, _with_bias_note("negamax matchup unreliable due opponent timeouts"), score, composite_details

    if eval_mcts_lite is not None and (not eval_mcts_lite.reliable):
        return False, _with_bias_note("mcts_lite matchup unreliable due opponent timeouts"), score, composite_details

    if previous_best_required and eval_prev_best_for_scoring is None and (not ignore_prev_best_due_side_bias):
        return False, _with_bias_note("previous-best comparison missing"), score, composite_details

    # Guardrails: random is only sanity-check; rely on stronger opponents.
    if eval_prev_best_for_scoring is not None and eval_prev_best_for_scoring.win_rate < 0.5:
        return False, _with_bias_note("failed vs previous best (<50%)"), score, composite_details
    if eval_prev_best_for_scoring is not None and (not eval_prev_best_for_scoring.reliable):
        return False, _with_bias_note("previous-best matchup unreliable due opponent timeouts"), score, composite_details
    if eval_prev_best_for_scoring is not None and float(eval_prev_best_for_scoring.side_bias) > PREVIOUS_BEST_SIDE_BIAS_THRESHOLD:
        return (
            False,
            _with_bias_note("previous-best eval unstable due first/second-player side bias; require more games"),
            score,
            composite_details,
        )
    if eval_negamax is not None and best_negamax >= 0.0 and eval_negamax.win_rate + 1e-9 < (best_negamax - 0.03):
        return False, _with_bias_note("negamax regressed too much"), score, composite_details
    if eval_mcts_lite is not None and best_mcts >= 0.0 and eval_mcts_lite.win_rate + 1e-9 < (best_mcts - 0.03):
        return False, _with_bias_note("mcts_lite regressed too much"), score, composite_details
    if math.isnan(score):
        return False, _with_bias_note("composite unavailable (no reliable component)"), score, composite_details
    if score <= best_score + 1e-9:
        return False, _with_bias_note("composite score not improved"), score, composite_details
    return True, _with_bias_note("composite improved with stability checks"), score, composite_details


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


def _legacy_teacher_data_reasons(npz_path: Path) -> list[str]:
    meta = _load_selfplay_metadata(npz_path)
    reasons: list[str] = []
    if not meta:
        reasons.append("missing metadata")
        return reasons
    if str(meta.get("value_mode", "")) == "score":
        reasons.append("legacy score-valued teacher labels")
    max_state_repeats = meta.get("max_state_repeats")
    if max_state_repeats is None:
        reasons.append("missing deduplicated-sampling metadata")
    else:
        try:
            if int(max_state_repeats) > 1:
                reasons.append(f"max_state_repeats={int(max_state_repeats)}")
        except Exception:
            reasons.append(f"invalid max_state_repeats={max_state_repeats}")
    return reasons


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
    parser.add_argument(
        "--teacher-batch-ratio-start",
        type=float,
        default=0.30,
        help="Teacher sample ratio at the start of training when teacher data is present.",
    )
    parser.add_argument(
        "--teacher-batch-ratio-end",
        type=float,
        default=0.05,
        help="Teacher sample ratio near the end of training when teacher data is present.",
    )
    parser.add_argument(
        "--selfplay-actor-mode",
        choices=("latest", "best", "alternate"),
        default="alternate",
        help="Which checkpoint drives self-play data generation once best.pt exists.",
    )
    parser.add_argument("--self-play-workers", type=int, default=None)
    parser.add_argument("--main-cpu-threads", type=int, default=None)
    parser.add_argument("--worker-cpu-threads", type=int, default=1)
    parser.add_argument(
        "--allow-legacy-teacher-data",
        action="store_true",
        help="Allow loading older teacher NPZ files even if metadata indicates legacy labels or no deduplicated sampling.",
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
    cpu_plan = suggest_cpu_plan()
    main_cpu_threads = (
        int(args.main_cpu_threads)
        if args.main_cpu_threads is not None
        else int(cpu_plan["main_threads"])
    )
    self_play_workers = (
        int(args.self_play_workers)
        if args.self_play_workers is not None
        else int(cpu_plan["selfplay_workers"])
    )
    worker_cpu_threads = max(1, int(args.worker_cpu_threads))
    if device == "cpu":
        configure_cpu_runtime(main_cpu_threads)
    elif self_play_workers > 1:
        print("[train] non-cpu device detected; forcing self_play_workers=1 for stability")
        self_play_workers = 1

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt_path = ckpt_dir / "latest.pt"
    best_ckpt_path = ckpt_dir / "best.pt"
    replay_path = ckpt_dir / "replay_buffer_latest.npz"
    state_path = ckpt_dir / "train_state.json"
    selfplay_output_dir = ckpt_dir / "selfplay_npz"
    archived_best_dir = ckpt_dir / "archived_best"
    archived_best_dir.mkdir(parents=True, exist_ok=True)

    state = _load_train_state(state_path)
    best_metrics = {
        "best_composite": float(state.get("best_composite", -1.0)),
        "best_negamax_win_rate": float(state.get("best_negamax_win_rate", -1.0)),
        "best_mcts_lite_win_rate": float(state.get("best_mcts_lite_win_rate", -1.0)),
    }
    best_composite_details = state.get("best_composite_details")
    best_iteration = int(state.get("best_iteration", 0) or 0)
    previous_best_checkpoint = str(state.get("previous_best_checkpoint", "") or "")
    archived_best_checkpoint = str(state.get("archived_best_checkpoint", "") or "")
    teacher_paths = _parse_teacher_data_list(args.teacher_data)

    # 1) Load model + optimizer
    if args.resume and latest_ckpt_path.exists():
        model, optimizer, ckpt_meta = _load_model_and_optimizer(
            latest_ckpt_path, device=device, lr=float(args.lr)
        )
        resumed_target_version = str(ckpt_meta.get("mcts_target_version", "") or "")
        resumed_iteration = int(ckpt_meta.get("iteration", 0) or 0)
        if resumed_iteration > 0 and resumed_target_version != MCTS_TARGET_VERSION:
            raise RuntimeError(
                "Refusing to resume latest checkpoint trained with legacy/missing "
                f"MCTS target version (found={resumed_target_version or 'missing'} "
                f"expected={MCTS_TARGET_VERSION}). Start from a fresh checkpoint/pretrain instead."
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

    # 2) Load replay buffers
    teacher_replay = ReplayBuffer(max_size=max(1_000_000, int(args.buffer_size)))
    selfplay_replay = ReplayBuffer(max_size=int(args.buffer_size))
    if args.resume and replay_path.exists():
        replay_meta = _load_selfplay_metadata(replay_path)
        replay_kind = str(replay_meta.get("buffer_kind", "") or "")
        replay_target_version = str(replay_meta.get("mcts_target_version", "") or "")
        if replay_kind == "selfplay_only" and replay_target_version == MCTS_TARGET_VERSION:
            selfplay_replay.load(replay_path)
            print(f"[train] resumed self-play replay buffer: size={len(selfplay_replay)}")
        elif replay_kind == "selfplay_only":
            print(
                "[train] replay buffer target version mismatch; skipping resume load "
                f"(found={replay_target_version or 'missing'} expected={MCTS_TARGET_VERSION})"
            )
        elif teacher_paths:
            print(
                "[train] legacy replay buffer detected; skipping resume load because teacher data "
                "will be reloaded separately and old combined replay would double-count teacher samples"
            )
        else:
            selfplay_replay.load(replay_path)
            print(
                "[train] WARNING resumed legacy replay buffer without teacher/source metadata; "
                f"size={len(selfplay_replay)}"
            )

    for td in teacher_paths:
        td_path = Path(td)
        legacy_reasons = _legacy_teacher_data_reasons(td_path)
        if legacy_reasons and (not bool(args.allow_legacy_teacher_data)):
            print(
                "[train] skipping legacy teacher data: {path}  reasons={reasons}".format(
                    path=td_path,
                    reasons="; ".join(legacy_reasons),
                )
            )
            continue
        if legacy_reasons:
            print(
                "[train] WARNING loading legacy teacher data: {path}  reasons={reasons}".format(
                    path=td_path,
                    reasons="; ".join(legacy_reasons),
                )
            )
        added = teacher_replay.add_npz(td_path)
        print(f"[train] loaded teacher data: {td_path}  added={added}  teacher_buffer={len(teacher_replay)}")

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
    print(
        "[train] teacher buffer={tb} selfplay buffer={sb} teacher_batch_ratio(start/end)={ts:.2f}/{te:.2f} selfplay_actor_mode={am}".format(
            tb=len(teacher_replay),
            sb=len(selfplay_replay),
            ts=float(args.teacher_batch_ratio_start),
            te=float(args.teacher_batch_ratio_end),
            am=str(args.selfplay_actor_mode),
        )
    )
    print(
        "[train] cpu plan total={total} main_threads={main} self_play_workers={spw} worker_threads={wt} reserve={reserve}".format(
            total=cpu_plan["total_cpus"],
            main=main_cpu_threads,
            spw=max(1, int(self_play_workers)),
            wt=worker_cpu_threads,
            reserve=cpu_plan["reserve_cores"],
        )
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
    candidate_internal_budget_negamax_ms = eval_core.derive_internal_time_budget_ms(
        float(eval_cfg["candidate_timeout_ms"]),
        default_budget_ms=eval_core.DEFAULT_NEGAMAX_TIME_BUDGET_MS,
    )
    candidate_internal_budget_mcts_ms = eval_core.derive_internal_time_budget_ms(
        float(eval_cfg["candidate_timeout_ms"]),
        default_budget_ms=eval_core.DEFAULT_MCTS_LITE_TIME_BUDGET_MS,
    )
    opponent_internal_budget_negamax_ms = eval_core.derive_internal_time_budget_ms(
        float(eval_cfg["opponent_timeout_ms"]),
        default_budget_ms=eval_core.DEFAULT_NEGAMAX_TIME_BUDGET_MS,
    )
    opponent_internal_budget_mcts_ms = eval_core.derive_internal_time_budget_ms(
        float(eval_cfg["opponent_timeout_ms"]),
        default_budget_ms=eval_core.DEFAULT_MCTS_LITE_TIME_BUDGET_MS,
    )
    print(
        "[train] budget candidate(ext/internal negamax/mcts)={ce:.0f}/{cni:.0f}/{cmi:.0f}ms "
        "opponent(ext/internal negamax/mcts)={oe:.0f}/{oni:.0f}/{omi:.0f}ms".format(
            ce=float(eval_cfg["candidate_timeout_ms"]),
            cni=float(candidate_internal_budget_negamax_ms),
            cmi=float(candidate_internal_budget_mcts_ms),
            oe=float(eval_cfg["opponent_timeout_ms"]),
            oni=float(opponent_internal_budget_negamax_ms),
            omi=float(opponent_internal_budget_mcts_ms),
        )
    )

    for iteration in range(start_iteration, end_iteration + 1):
        print("\n" + "=" * 72)
        print(f"[train] iteration {iteration}")
        print("=" * 72)

        # 2) generate self-play games
        actor_model = model
        actor_source = "latest"
        if best_ckpt_path.exists():
            actor_mode = str(args.selfplay_actor_mode)
            use_best_actor = (
                actor_mode == "best"
                or (actor_mode == "alternate" and iteration % 2 == 0)
            )
            if use_best_actor:
                actor_model, _, _ = _load_model_and_optimizer(
                    best_ckpt_path,
                    device=device,
                    lr=float(args.lr),
                )
                actor_model.eval()
                actor_source = "best"
        else:
            model.eval()

        selfplay_t0 = time.perf_counter()
        selfplay_checkpoint_path: Optional[Path] = None
        if max(1, int(self_play_workers)) > 1:
            if actor_source == "best":
                selfplay_checkpoint_path = best_ckpt_path
            else:
                selfplay_checkpoint_path = ckpt_dir / "_selfplay_actor_tmp.pt"
                save_checkpoint(
                    actor_model,
                    None,
                    selfplay_checkpoint_path,
                    metadata={
                        "iteration": int(iteration),
                        "simulations": int(args.simulations),
                        "actor_source": actor_source,
                    },
                )
        selfplay_npz = generate_self_play_games(
            model=actor_model,
            num_games=int(args.self_play_games),
            num_simulations=int(args.simulations),
            device=device,
            augment=True,
            output_dir=str(selfplay_output_dir),
            c_puct=1.5,
            add_dirichlet_noise=True,
            use_tactical_shortcuts=False,
            checkpoint_path=str(selfplay_checkpoint_path) if selfplay_checkpoint_path else None,
            workers=max(1, int(self_play_workers)),
            main_cpu_threads=main_cpu_threads,
            worker_cpu_threads=worker_cpu_threads,
            seed=int(args.seed) + (int(iteration) * 10007),
        )
        if selfplay_checkpoint_path is not None and selfplay_checkpoint_path.name == "_selfplay_actor_tmp.pt":
            try:
                selfplay_checkpoint_path.unlink()
            except Exception:
                pass
        selfplay_sec = time.perf_counter() - selfplay_t0
        selfplay_meta = _load_selfplay_metadata(selfplay_npz)
        avg_game_length = float(selfplay_meta.get("avg_game_length", 0.0))
        winner_distribution = selfplay_meta.get("winner_distribution", {})

        # 3) add to replay buffer
        added = selfplay_replay.add_npz(selfplay_npz)
        total_buffer = len(teacher_replay) + len(selfplay_replay)
        print(
            f"[train] self-play actor={actor_source} added={added}, "
            f"teacher_buffer={len(teacher_replay)} selfplay_buffer={len(selfplay_replay)} total_buffer={total_buffer}"
        )

        if total_buffer == 0:
            raise RuntimeError("Training buffers are empty after self-play generation")

        # 4) train
        teacher_batch_ratio = _compute_teacher_batch_ratio(
            iteration=iteration,
            start_iteration=start_iteration,
            end_iteration=end_iteration,
            ratio_start=float(args.teacher_batch_ratio_start),
            ratio_end=float(args.teacher_batch_ratio_end),
        )
        train_t0 = time.perf_counter()
        metrics = _train_steps(
            model=model,
            optimizer=optimizer,
            teacher_replay=teacher_replay,
            selfplay_replay=selfplay_replay,
            batch_size=int(args.batch_size),
            train_steps=int(args.train_steps),
            device=device,
            iteration=iteration,
            teacher_batch_ratio=teacher_batch_ratio,
        )
        train_sec = time.perf_counter() - train_t0

        # 5) save latest checkpoint + replay snapshot
        latest_meta = {
            "iteration": int(iteration),
            "train_steps": int(args.train_steps),
            "simulations": int(args.simulations),
            "eval_results": {},
            "eval_profile": str(eval_cfg["eval_profile"]),
            "candidate_timeout_ms": float(eval_cfg["candidate_timeout_ms"]),
            "opponent_timeout_ms": float(eval_cfg["opponent_timeout_ms"]),
            "previous_best_checkpoint": previous_best_checkpoint,
            "archived_best_checkpoint": archived_best_checkpoint,
            "teacher_buffer_size": int(len(teacher_replay)),
            "selfplay_buffer_size": int(len(selfplay_replay)),
            "buffer_size": int(len(teacher_replay) + len(selfplay_replay)),
            "teacher_batch_ratio": float(teacher_batch_ratio),
            "loss": metrics,
            "self_play_avg_game_length": avg_game_length,
            "self_play_winner_distribution": winner_distribution,
            "self_play_actor_source": actor_source,
            "self_play_seconds": float(selfplay_sec),
            "train_seconds": float(train_sec),
            "mcts_target_version": MCTS_TARGET_VERSION,
            "created_at": _utc_now_iso(),
        }
        save_checkpoint(model, optimizer, latest_ckpt_path, metadata=latest_meta)
        selfplay_replay.save(
            replay_path,
            metadata_extra={
                "buffer_kind": "selfplay_only",
                "teacher_buffer_size": int(len(teacher_replay)),
                "selfplay_buffer_size": int(len(selfplay_replay)),
                "mcts_target_version": MCTS_TARGET_VERSION,
            },
        )

        eval_random = None
        eval_negamax = None
        eval_mcts_lite = None
        eval_prev_best = None
        gate_passed = False
        gate_reason = "evaluation skipped"
        composite = float("nan")
        composite_details: Dict[str, object] = {}
        previous_best_required = False
        eval_sec = 0.0

        # 6) quick evaluation
        if int(args.eval_interval) > 0 and (iteration % int(args.eval_interval) == 0):
            model.eval()
            eval_t0 = time.perf_counter()
            current_agent = _mcts_model_agent(model, device=device, simulations=int(args.simulations))
            random_games = eval_core.games_for_opponent(eval_cfg, "random")
            negamax_games = eval_core.games_for_opponent(eval_cfg, "negamax")
            mcts_games = eval_core.games_for_opponent(eval_cfg, "mcts_lite")
            prev_best_games = eval_core.games_for_opponent(eval_cfg, "previous_best")

            if random_games > 0:
                eval_random = _evaluate_vs(
                    current_agent,
                    "random",
                    games=int(random_games),
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
            if negamax_games > 0:
                eval_negamax = _evaluate_vs(
                    current_agent,
                    "negamax",
                    games=int(negamax_games),
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
            if mcts_games > 0 and iteration % mcts_every == 0:
                eval_mcts_lite = _evaluate_vs(
                    current_agent,
                    "mcts_lite",
                    games=int(mcts_games),
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
                previous_best_required = bool(prev_best_games > 0)
                snapshot_name = f"previous_best_snapshot_iter{int(iteration):04d}.pt"
                snapshot_path = archived_best_dir / snapshot_name
                shutil.copy2(best_ckpt_path, snapshot_path)
                previous_best_checkpoint = str(snapshot_path.resolve())
                archived_best_checkpoint = previous_best_checkpoint

                if prev_best_games > 0:
                    best_model, _, _ = _load_model_and_optimizer(snapshot_path, device=device, lr=float(args.lr))
                    best_agent = _mcts_model_agent(best_model, device=device, simulations=int(args.simulations))
                    eval_prev_best = _evaluate_vs(
                        current_agent,
                        best_agent,
                        games=int(prev_best_games),
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
                previous_best_required=previous_best_required,
            )
            eval_sec = time.perf_counter() - eval_t0
            if gate_passed:
                best_meta = dict(latest_meta)
                best_meta["eval_results"] = {
                    "evaluator_backend": eval_core.EVALUATOR_BACKEND,
                    "eval_profile": str(eval_cfg["eval_profile"]),
                    "candidate_timeout_ms": float(eval_cfg["candidate_timeout_ms"]),
                    "opponent_timeout_ms": float(eval_cfg["opponent_timeout_ms"]),
                    "random": eval_random.__dict__ if eval_random else None,
                    "negamax": eval_negamax.__dict__ if eval_negamax else None,
                    "mcts_lite": eval_mcts_lite.__dict__ if eval_mcts_lite else None,
                    "previous_best": eval_prev_best.__dict__ if eval_prev_best else None,
                    "previous_best_checkpoint": previous_best_checkpoint,
                    "side_bias_warning": bool(
                        eval_prev_best is not None
                        and float(eval_prev_best.side_bias) > PREVIOUS_BEST_SIDE_BIAS_THRESHOLD
                    ),
                    "unstable_previous_best_eval": bool(
                        eval_prev_best is not None
                        and float(eval_prev_best.side_bias) > PREVIOUS_BEST_SIDE_BIAS_THRESHOLD
                    ),
                    "composite_score": float(composite),
                    "composite": composite_details,
                    "gate_reason": gate_reason,
                }
                save_checkpoint(model, optimizer, best_ckpt_path, metadata=best_meta)
                best_metrics["best_composite"] = float(composite)
                if eval_negamax is not None:
                    best_metrics["best_negamax_win_rate"] = float(eval_negamax.win_rate)
                if eval_mcts_lite is not None:
                    best_metrics["best_mcts_lite_win_rate"] = float(eval_mcts_lite.win_rate)
                best_composite_details = composite_details
                best_iteration = int(iteration)

        # 5/6/7 logs
        print(
            f"[train] iteration={iteration} "
            f"teacher_buffer={len(teacher_replay)} "
            f"selfplay_buffer={len(selfplay_replay)} "
            f"buffer={len(teacher_replay) + len(selfplay_replay)} "
            f"policy_loss={metrics['policy_loss']:.4f} "
            f"value_loss={metrics['value_loss']:.4f} "
            f"total_loss={metrics['total_loss']:.4f}"
        )
        print(
            f"[train] self-play avg_game_length={avg_game_length:.2f} "
            f"winner_distribution={winner_distribution} "
            f"actor={actor_source} teacher_batch_ratio={teacher_batch_ratio:.3f} "
            f"batch_mix(actual teacher/selfplay)={metrics['teacher_examples_seen']}/{metrics['selfplay_examples_seen']}"
        )
        print(
            f"[train] stage_time_sec selfplay={selfplay_sec:.1f} train={train_sec:.1f} eval={eval_sec:.1f}"
        )
        print("[train] " + _format_eval("eval vs random", eval_random))
        print("[train] " + _format_eval("eval vs negamax", eval_negamax))
        print("[train] " + _format_eval("eval vs mcts_lite", eval_mcts_lite))
        print("[train] " + _format_eval("eval vs previous best", eval_prev_best))
        if any(x is not None for x in (eval_random, eval_negamax, eval_mcts_lite, eval_prev_best)):
            print(
                f"[train] gating passed={gate_passed} "
                f"reason={gate_reason} "
                f"composite={composite:.4f}"
            )
        print(f"[train] latest checkpoint: {latest_ckpt_path.resolve()}")
        if gate_passed:
            print(f"[train] best checkpoint:   {best_ckpt_path.resolve()}")

        latest_composite = None
        if not math.isnan(float(composite)):
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
                "side_bias_warning": bool(
                    eval_prev_best is not None
                    and float(eval_prev_best.side_bias) > PREVIOUS_BEST_SIDE_BIAS_THRESHOLD
                ),
                "unstable_previous_best_eval": bool(
                    eval_prev_best is not None
                    and float(eval_prev_best.side_bias) > PREVIOUS_BEST_SIDE_BIAS_THRESHOLD
                ),
                "latest_composite_details": composite_details if composite_details else None,
                "latest_checkpoint": str(latest_ckpt_path.resolve()),
                "best_checkpoint": str(best_ckpt_path.resolve()) if best_ckpt_path.exists() else "",
                "previous_best_checkpoint": previous_best_checkpoint,
                "archived_best_checkpoint": archived_best_checkpoint,
                "eval_profile": str(eval_cfg["eval_profile"]),
                "candidate_timeout_ms": float(eval_cfg["candidate_timeout_ms"]),
                "opponent_timeout_ms": float(eval_cfg["opponent_timeout_ms"]),
                "eval_games_by_opponent": dict(eval_cfg.get("games_by_opponent", {})),
                "eval_mcts_lite_every": int(args.eval_mcts_lite_every),
                "teacher_buffer_size": int(len(teacher_replay)),
                "selfplay_buffer_size": int(len(selfplay_replay)),
                "buffer_size": int(len(teacher_replay) + len(selfplay_replay)),
                "teacher_batch_ratio_start": float(args.teacher_batch_ratio_start),
                "teacher_batch_ratio_end": float(args.teacher_batch_ratio_end),
                "teacher_batch_ratio_current": float(teacher_batch_ratio),
                "selfplay_actor_mode": str(args.selfplay_actor_mode),
                "self_play_workers": int(self_play_workers),
                "main_cpu_threads": int(main_cpu_threads),
                "worker_cpu_threads": int(worker_cpu_threads),
                "updated_at": _utc_now_iso(),
            }
        )
        _save_train_state(state_path, state)


if __name__ == "__main__":
    _main()
