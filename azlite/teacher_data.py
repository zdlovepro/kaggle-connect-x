"""Teacher-supervised dataset generation for AlphaZero-lite cold start.

This module builds pretraining samples from mixed position sources and strong-ish
teachers (submission/negamax/heuristic/MCTS). Output is a single `.npz` file:
  - states:   (N, C, 6, 7)
  - policies: (N, 7)
  - values:   (N,)
  - metadata: JSON string
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from azlite.board import (
    apply_move,
    find_immediate_block,
    find_immediate_win,
    legal_moves,
    terminal_value,
    to_tensor,
)
from azlite.puct_mcts import HeuristicEvaluator, run_mcts

try:
    import submission as _submission
except Exception:
    _submission = None

try:
    from agents import minimax_bitboard as _mm
except Exception:
    _mm = None

try:
    from agents.mcts_agent import MCTSAgent
except Exception:
    MCTSAgent = None


ROWS = 6
COLS = 7
MAX_MOVES = ROWS * COLS

DEFAULT_SOURCE_MIX = (
    "random:0.10,"
    "heuristic_vs_random:0.10,"
    "mcts_vs_random:0.15,"
    "mcts_vs_negamax:0.55,"
    "negamax_selfplay:0.10"
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_cfg() -> SimpleNamespace:
    return SimpleNamespace(columns=COLS, rows=ROWS, inarow=4)


def _build_obs(board: np.ndarray, mark: int) -> SimpleNamespace:
    return SimpleNamespace(
        board=board.reshape(-1).astype(np.int8).tolist(),
        mark=int(mark),
        step=int(np.count_nonzero(board)),
        remainingOverageTime=60.0,
    )


def _normalize_mix(mix: Sequence[Tuple[str, float]]) -> List[Tuple[str, float]]:
    cleaned: List[Tuple[str, float]] = []
    for name, weight in mix:
        try:
            w = float(weight)
        except (TypeError, ValueError):
            continue
        if w > 0:
            cleaned.append((str(name), w))
    if not cleaned:
        return [("random", 1.0)]
    s = sum(w for _, w in cleaned)
    return [(name, w / s) for name, w in cleaned]


def parse_source_mix(spec: str) -> List[Tuple[str, float]]:
    """Parse source mix, e.g. 'random:0.2,heuristic_vs_random:0.8'."""
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    parsed: List[Tuple[str, float]] = []
    for p in parts:
        if ":" in p:
            name, w = p.split(":", 1)
            try:
                parsed.append((name.strip(), float(w.strip())))
            except ValueError:
                continue
        else:
            parsed.append((p, 1.0))
    return _normalize_mix(parsed)


def _sample_source(mix: Sequence[Tuple[str, float]], rng: random.Random) -> str:
    x = rng.random()
    acc = 0.0
    for name, weight in mix:
        acc += weight
        if x <= acc:
            return name
    return mix[-1][0]


def _allocate_source_targets(
    positions: int,
    source_mix: Sequence[Tuple[str, float]],
) -> Dict[str, int]:
    mix = _normalize_mix(source_mix)
    total = max(0, int(positions))
    raw = [(name, float(total) * float(weight)) for name, weight in mix]
    targets = {name: int(math.floor(v)) for name, v in raw}
    remainder = total - sum(targets.values())
    ranked = sorted(raw, key=lambda item: (-(item[1] - math.floor(item[1])), item[0]))
    for i in range(remainder):
        name = ranked[i % len(ranked)][0]
        targets[name] += 1
    return targets


def _sample_source_by_remaining_quota(
    targets: Mapping[str, int],
    kept_counts: Mapping[str, int],
    rng: random.Random,
) -> Optional[str]:
    deficits = [
        (name, int(target) - int(kept_counts.get(name, 0)))
        for name, target in targets.items()
        if int(target) > int(kept_counts.get(name, 0))
    ]
    if not deficits:
        return None
    total = sum(deficit for _, deficit in deficits)
    x = rng.randrange(total)
    acc = 0
    for name, deficit in deficits:
        acc += deficit
        if x < acc:
            return name
    return deficits[-1][0]


@dataclass
class TeacherContext:
    teacher_depth: int
    mcts_sims: int
    mcts_time_ms: float
    negamax_time_ms: float

    def __post_init__(self) -> None:
        self.cfg = _build_cfg()
        self.heur_eval = HeuristicEvaluator()
        self.mcts_agent = (
            MCTSAgent(time_budget_ms=float(self.mcts_time_ms)) if MCTSAgent is not None else None
        )
        self.negamax_agent = None
        if _mm is not None:
            self.negamax_agent = _mm.MinimaxBitboardAgent(
                max_depth=max(1, int(self.teacher_depth)),
                time_budget_ms=float(self.negamax_time_ms),
                use_tt=True,
            )


def _pick_random(board: np.ndarray, rng: random.Random) -> int:
    valid = legal_moves(board)
    if not valid:
        return 0
    return int(rng.choice(valid))


def _pick_heuristic(board: np.ndarray, mark: int, ctx: TeacherContext) -> int:
    valid = legal_moves(board)
    if not valid:
        return 0
    opp = 2 if mark == 1 else 1
    win_col = find_immediate_win(board, mark)
    if win_col is not None:
        return int(win_col)
    block_col = find_immediate_block(board, mark, opp)
    if block_col is not None:
        return int(block_col)
    policy, _ = ctx.heur_eval.evaluate(board, mark)
    best = max(valid, key=lambda c: float(policy[c]))
    return int(best)


def _pick_mcts_lite(board: np.ndarray, mark: int, ctx: TeacherContext, rng: random.Random) -> int:
    if ctx.mcts_agent is None:
        return _pick_heuristic(board, mark, ctx)
    valid = legal_moves(board)
    if not valid:
        return 0
    obs = _build_obs(board, mark)
    try:
        move = int(ctx.mcts_agent.select_action(obs, ctx.cfg))
        if move in valid:
            return move
    except Exception:
        pass
    return _pick_random(board, rng)


def _pick_negamax(board: np.ndarray, mark: int, ctx: TeacherContext, rng: random.Random) -> int:
    if ctx.negamax_agent is None:
        return _pick_heuristic(board, mark, ctx)
    valid = legal_moves(board)
    if not valid:
        return 0
    obs = _build_obs(board, mark)
    try:
        move = int(ctx.negamax_agent.select_action(obs, ctx.cfg))
        if move in valid:
            return move
    except Exception:
        pass
    return _pick_random(board, rng)


def _select_move_by_policy(
    policy_name: str,
    board: np.ndarray,
    mark: int,
    ctx: TeacherContext,
    rng: random.Random,
) -> int:
    if policy_name == "random":
        return _pick_random(board, rng)
    if policy_name == "heuristic":
        return _pick_heuristic(board, mark, ctx)
    if policy_name == "mcts":
        return _pick_mcts_lite(board, mark, ctx, rng)
    if policy_name == "negamax":
        return _pick_negamax(board, mark, ctx, rng)
    # Fallback keeps generation robust.
    return _pick_random(board, rng)


def _source_roles(source: str, game_index: int) -> Tuple[str, str]:
    """Map source name -> (policy_for_player1, policy_for_player2)."""
    if source == "random":
        return "random", "random"
    if source == "heuristic_vs_random":
        return ("heuristic", "random") if game_index % 2 == 0 else ("random", "heuristic")
    if source == "mcts_vs_random":
        return ("mcts", "random") if game_index % 2 == 0 else ("random", "mcts")
    if source == "mcts_vs_negamax":
        return ("mcts", "negamax") if game_index % 2 == 0 else ("negamax", "mcts")
    if source == "negamax_selfplay":
        return "negamax", "negamax"
    return "heuristic", "random"


def collect_positions(
    positions: int,
    source_mix: Sequence[Tuple[str, float]],
    ctx: TeacherContext,
    seed: int = 42,
    max_state_repeats: int = 1,
    source_sampling_mode: str = "quota",
    max_source_stall_games: int = 128,
) -> List[Tuple[np.ndarray, int, str]]:
    """Collect non-terminal positions from mixed game sources."""
    rng = random.Random(seed)
    out: List[Tuple[np.ndarray, int, str]] = []
    state_counts: Dict[bytes, int] = {}
    source_targets = _allocate_source_targets(int(positions), source_mix)
    kept_counts: Dict[str, int] = {name: 0 for name in source_targets}
    stalled_games: Dict[str, int] = {name: 0 for name in source_targets}
    games = 0
    repeat_limit = max(1, int(max_state_repeats))
    stall_limit = max(8, int(max_source_stall_games))
    mode = str(source_sampling_mode).strip().lower() or "quota"

    while len(out) < positions:
        if mode == "quota":
            source = _sample_source_by_remaining_quota(source_targets, kept_counts, rng)
            if source is None:
                mode = "weighted_fill"
                source = _sample_source(source_mix, rng)
            else:
                deficit = int(source_targets.get(source, 0)) - int(kept_counts.get(source, 0))
                if deficit > 0 and int(stalled_games.get(source, 0)) >= stall_limit:
                    source_targets[source] = int(kept_counts.get(source, 0))
                    source = _sample_source_by_remaining_quota(source_targets, kept_counts, rng)
                    if source is None:
                        mode = "weighted_fill"
                        source = _sample_source(source_mix, rng)
        else:
            source = _sample_source(source_mix, rng)

        p1_policy, p2_policy = _source_roles(source, games)
        games += 1

        board = np.zeros((ROWS, COLS), dtype=np.int8)
        mark = 1
        added_this_game = 0

        for _ in range(MAX_MOVES):
            if terminal_value(board, mark) is not None:
                break

            state_key = board.tobytes() + bytes((int(mark),))
            seen = state_counts.get(state_key, 0)
            target_remaining = int(source_targets.get(source, 0)) - int(kept_counts.get(source, 0))
            quota_allows_append = (mode != "quota") or (target_remaining > 0)
            if quota_allows_append and seen < repeat_limit:
                out.append((board.copy(), int(mark), source))
                state_counts[state_key] = seen + 1
                kept_counts[source] = kept_counts.get(source, 0) + 1
                added_this_game += 1
                if len(out) >= positions:
                    break

            policy_name = p1_policy if mark == 1 else p2_policy
            move = _select_move_by_policy(policy_name, board, mark, ctx, rng)
            valid = legal_moves(board)
            if move not in valid:
                move = valid[0] if valid else 0
            board = apply_move(board, move, mark)
            mark = 2 if mark == 1 else 1

        if source in stalled_games:
            if added_this_game <= 0:
                stalled_games[source] = stalled_games.get(source, 0) + 1
            else:
                stalled_games[source] = 0
        if games % 50 == 0 or len(out) >= positions:
            summary = ", ".join(
                f"{name}:{kept_counts.get(name, 0)}/{source_targets.get(name, 0)}"
                for name, _ in source_mix
                if name in source_targets
            )
            print(
                f"[teacher_data] collecting games={games} kept={len(out)}/{positions} "
                f"mode={mode} sources=[{summary}]"
            )

    return out


def _softmax_on_legal(scores: np.ndarray, legal: Sequence[int], temperature: float) -> np.ndarray:
    policy = np.zeros(COLS, dtype=np.float32)
    if not legal:
        return policy

    t = max(1e-6, float(temperature))
    legal_scores = np.array([scores[c] for c in legal], dtype=np.float64) / t
    legal_scores -= np.max(legal_scores)
    exp = np.exp(legal_scores)
    s = float(exp.sum())
    if s <= 1e-12:
        policy[list(legal)] = 1.0 / len(legal)
        return policy

    probs = exp / s
    for i, c in enumerate(legal):
        policy[c] = np.float32(probs[i])
    return policy


def _policy_target(
    scores: np.ndarray,
    legal: Sequence[int],
    policy_mode: str,
    temperature: float,
) -> np.ndarray:
    policy = np.zeros(COLS, dtype=np.float32)
    if not legal:
        return policy

    if policy_mode == "one_hot":
        best_col = max(legal, key=lambda c: float(scores[c]))
        policy[best_col] = 1.0
        return policy

    return _softmax_on_legal(scores, legal, temperature)


def _score_with_heuristic(board: np.ndarray, mark: int, ctx: TeacherContext) -> Tuple[np.ndarray, float]:
    valid = legal_moves(board)
    scores = np.full(COLS, -np.inf, dtype=np.float64)
    if not valid:
        return scores, 0.0
    policy, value = ctx.heur_eval.evaluate(board, mark)
    for c in valid:
        scores[c] = float(policy[c])
    return scores, float(max(-1.0, min(1.0, value)))


def _score_with_mcts(board: np.ndarray, mark: int, ctx: TeacherContext) -> Tuple[np.ndarray, float]:
    valid = legal_moves(board)
    scores = np.full(COLS, -np.inf, dtype=np.float64)
    if not valid:
        return scores, 0.0

    result = run_mcts(
        board=board,
        current_player=mark,
        evaluator=ctx.heur_eval,
        num_simulations=max(1, int(ctx.mcts_sims)),
        c_puct=1.5,
        temperature=1.0,
        add_dirichlet_noise=False,
        use_tactical_shortcuts=False,
        return_root=False,
    )
    visits = np.asarray(result["visit_counts"], dtype=np.float64)
    for c in valid:
        scores[c] = visits[c]
    value = float(result.get("root_value", 0.0))
    return scores, max(-1.0, min(1.0, value))


def _score_with_negamax(board: np.ndarray, mark: int, depth: int, ctx: TeacherContext) -> Tuple[np.ndarray, float]:
    valid = legal_moves(board)
    scores = np.full(COLS, -np.inf, dtype=np.float64)
    if not valid or _mm is None or ctx.negamax_agent is None:
        return _score_with_heuristic(board, mark, ctx)

    flat = board.reshape(-1).astype(np.int8).tolist()
    b1, b2, heights = _mm._list_to_bitboards(flat)
    if mark == 1:
        b_self, b_opp = b1, b2
    else:
        b_self, b_opp = b2, b1

    hash_key = _mm.compute_hash(b_self, b_opp)
    ctx.negamax_agent.nodes_visited = 0
    ctx.negamax_agent.search_start = time.perf_counter()
    # Keep timeout checks effectively disabled during teacher labeling.
    ctx.negamax_agent.time_budget_ms = 1e9

    d = max(1, int(depth))
    for col in valid:
        pos = _mm._drop_piece_bb(heights, col)
        mask = np.uint64(1) << np.uint64(pos)
        child_self = b_self | mask
        if _mm.has_won(child_self):
            scores[col] = 10000.0
            continue

        heights[col] += 1
        child_hash = _mm.hash_update(hash_key, pos, 1)
        child_score, _ = ctx.negamax_agent._negascout(
            child_self,
            b_opp,
            heights,
            d - 1,
            -math.inf,
            math.inf,
            -1.0,
            child_hash,
        )
        heights[col] -= 1

        if ctx.negamax_agent.nodes_visited == -1:
            ctx.negamax_agent.nodes_visited = 0
            scores[col] = 0.0
        else:
            scores[col] = -float(child_score)

    best_score = float(np.max(scores[valid]))
    value = float(np.tanh(best_score / 10000.0))
    return scores, max(-1.0, min(1.0, value))


def _score_with_strong_submission(
    board: np.ndarray,
    mark: int,
    depth: int,
    ctx: TeacherContext,
) -> Tuple[np.ndarray, float]:
    valid = legal_moves(board)
    scores = np.full(COLS, -np.inf, dtype=np.float64)
    if not valid or _submission is None:
        return _score_with_negamax(board, mark, depth, ctx)

    flat = board.reshape(-1).astype(np.int8).tolist()
    b1, b2 = _submission._list_to_bitboards(flat)
    if mark == 1:
        b_self, b_opp = b1, b2
    else:
        b_self, b_opp = b2, b1
    heights = _submission._get_heights_from_list(flat)

    hash_key = _submission._zobrist_init(b_self, b_opp)
    mirror_hash_key = _submission._zobrist_init_mirror(b_self, b_opp)

    _submission._search_start = time.perf_counter()
    _submission._search_soft_budget_ms = 1e12
    _submission._search_hard_budget_ms = 1e12
    _submission._timed_out = False
    _submission._nodes = 0

    d = max(1, int(depth))
    for col in valid:
        pos = _submission._drop_pos(heights, col)
        mirror_pos = _submission._MIRROR_POS[pos]
        mask = np.uint64(1) << np.uint64(pos)

        child_self = b_self | mask
        if _submission._has_won(child_self):
            scores[col] = float(_submission.MATE_SCORE)
            continue

        heights[col] += 1
        child_hash = _submission._zobrist_update(hash_key, pos, 1)
        child_mirror = _submission._zobrist_update(mirror_hash_key, mirror_pos, 1)
        child_score, _ = _submission._negascout(
            child_self,
            b_opp,
            heights,
            d - 1,
            -math.inf,
            math.inf,
            -1.0,
            child_hash,
            child_mirror,
            exact_mode=False,
        )
        heights[col] -= 1

        if _submission._timed_out:
            _submission._timed_out = False
            scores[col] = 0.0
        else:
            scores[col] = -float(child_score)

    best_score = float(np.max(scores[valid]))
    denom = max(1.0, float(getattr(_submission, "MATE_SCORE", 100000.0)))
    value = float(np.tanh(best_score / denom))
    return scores, max(-1.0, min(1.0, value))


def teacher_scores(
    board: np.ndarray,
    mark: int,
    teacher: str,
    teacher_depth: int,
    ctx: TeacherContext,
) -> Tuple[np.ndarray, float]:
    """Return action scores and value from current player perspective."""
    if teacher == "strong":
        return _score_with_strong_submission(board, mark, teacher_depth, ctx)
    if teacher == "negamax":
        return _score_with_negamax(board, mark, teacher_depth, ctx)
    if teacher == "mcts":
        return _score_with_mcts(board, mark, ctx)
    if teacher == "heuristic":
        return _score_with_heuristic(board, mark, ctx)

    # auto: strongest available fallback chain
    if _submission is not None:
        return _score_with_strong_submission(board, mark, teacher_depth, ctx)
    if _mm is not None:
        return _score_with_negamax(board, mark, teacher_depth, ctx)
    return _score_with_heuristic(board, mark, ctx)


def _value_from_scores(
    scores: np.ndarray,
    legal: Sequence[int],
    teacher_value: float,
    value_mode: str,
    value_scale: float,
) -> float:
    if not legal:
        return 0.0
    if value_mode == "score":
        best = float(np.max(scores[list(legal)]))
        finite = np.abs(np.asarray(scores[list(legal)], dtype=np.float64))
        finite = finite[np.isfinite(finite)]
        inferred = float(np.max(finite)) if finite.size > 0 else 1.0
        denom = max(1e-6, float(value_scale if value_scale > 0.0 else inferred))
        return float(max(-1.0, min(1.0, math.tanh(best / denom))))
    return float(max(-1.0, min(1.0, teacher_value)))


def _rollout_outcome_value(
    board: np.ndarray,
    mark: int,
    ctx: TeacherContext,
    rng: random.Random,
    rollout_policy: str,
) -> float:
    """Complete game rollout for value target mode B."""
    root_mark = int(mark)
    cur_board = board.copy()
    cur_mark = int(mark)

    for _ in range(MAX_MOVES):
        tv = terminal_value(cur_board, cur_mark)
        if tv is not None:
            # tv is from cur_mark perspective; convert to root_mark perspective.
            if cur_mark == root_mark:
                return float(tv)
            return float(-tv)

        move = _select_move_by_policy(rollout_policy, cur_board, cur_mark, ctx, rng)
        valid = legal_moves(cur_board)
        if move not in valid:
            move = valid[0] if valid else 0
        cur_board = apply_move(cur_board, move, cur_mark)
        cur_mark = 2 if cur_mark == 1 else 1

    return 0.0


def build_teacher_dataset(
    positions: int,
    teacher: str,
    teacher_depth: int,
    policy_mode: str,
    policy_temperature: float,
    value_mode: str,
    value_scale: float,
    source_mix: Sequence[Tuple[str, float]],
    include_legal_channel: bool,
    seed: int,
    mcts_sims: int,
    mcts_time_ms: float,
    negamax_time_ms: float,
    rollout_policy: str,
    max_state_repeats: int = 1,
    source_sampling_mode: str = "quota",
    max_source_stall_games: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    ctx = TeacherContext(
        teacher_depth=teacher_depth,
        mcts_sims=mcts_sims,
        mcts_time_ms=mcts_time_ms,
        negamax_time_ms=negamax_time_ms,
    )
    rng = random.Random(seed)

    sampled = collect_positions(
        positions=positions,
        source_mix=source_mix,
        ctx=ctx,
        seed=seed,
        max_state_repeats=max_state_repeats,
        source_sampling_mode=source_sampling_mode,
        max_source_stall_games=max_source_stall_games,
    )

    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    values: List[float] = []
    source_counter: Dict[str, int] = {}

    t0 = time.perf_counter()
    for idx, (board, mark, source_name) in enumerate(sampled, start=1):
        valid = legal_moves(board)
        if not valid:
            continue

        scores, teacher_value = teacher_scores(
            board=board,
            mark=mark,
            teacher=teacher,
            teacher_depth=teacher_depth,
            ctx=ctx,
        )

        pi = _policy_target(
            scores=scores,
            legal=valid,
            policy_mode=policy_mode,
            temperature=policy_temperature,
        )
        if value_mode == "rollout":
            v = _rollout_outcome_value(board, mark, ctx, rng, rollout_policy)
        else:
            v = _value_from_scores(
                scores=scores,
                legal=valid,
                teacher_value=teacher_value,
                value_mode=value_mode,
                value_scale=value_scale,
            )

        x = to_tensor(
            board=board,
            current_player=mark,
            include_legal_channel=include_legal_channel,
            dtype=np.float32,
        )

        states.append(x)
        policies.append(pi.astype(np.float32, copy=False))
        values.append(float(v))
        source_counter[source_name] = source_counter.get(source_name, 0) + 1

        if idx % 1000 == 0 or idx == len(sampled):
            elapsed = time.perf_counter() - t0
            print(
                f"[teacher_data] labeled {idx}/{len(sampled)} "
                f"(kept={len(states)})  t={elapsed:.1f}s"
            )

    states_arr = np.asarray(states, dtype=np.float32)
    policies_arr = np.asarray(policies, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)

    metadata = {
        "teacher_type": teacher,
        "teacher_depth": int(teacher_depth),
        "positions_requested": int(positions),
        "positions_saved": int(states_arr.shape[0]),
        "policy_mode": policy_mode,
        "policy_temperature": float(policy_temperature),
        "value_mode": value_mode,
        "value_scale": float(value_scale),
        "channels": int(states_arr.shape[1]) if states_arr.ndim == 4 else 0,
        "source_mix": [{"name": n, "weight": float(w)} for n, w in source_mix],
        "source_targets": _allocate_source_targets(int(positions), source_mix),
        "source_counts": source_counter,
        "source_sampling_mode": str(source_sampling_mode),
        "max_source_stall_games": int(max_source_stall_games),
        "rollout_policy": rollout_policy,
        "mcts_sims": int(mcts_sims),
        "mcts_time_ms": float(mcts_time_ms),
        "negamax_time_ms": float(negamax_time_ms),
        "max_state_repeats": int(max_state_repeats),
        "deduplicated_sampling": bool(int(max_state_repeats) == 1),
        "created_at": _utc_now_iso(),
    }
    return states_arr, policies_arr, values_arr, metadata


def save_dataset_npz(
    output: Path,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    metadata: Dict[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        states=states,
        policies=policies,
        values=values,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate teacher-supervised ConnectX dataset")
    parser.add_argument("--positions", type=int, default=20000)
    parser.add_argument(
        "--teacher",
        choices=("auto", "strong", "negamax", "heuristic", "mcts"),
        default="auto",
    )
    parser.add_argument("--teacher-depth", type=int, default=5)
    parser.add_argument(
        "--policy-mode",
        choices=("soft", "one_hot"),
        default="soft",
        help="soft: legal-score softmax; one_hot: best move only",
    )
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument(
        "--value-mode",
        choices=("score", "teacher", "rollout"),
        default="teacher",
        help="score: tanh(best_score/scale); teacher: use teacher returned value; rollout: full game rollout outcome",
    )
    parser.add_argument(
        "--value-scale",
        type=float,
        default=0.0,
        help="Only used when --value-mode=score. <=0 means infer scale from teacher scores.",
    )
    parser.add_argument("--source-mix", type=str, default=DEFAULT_SOURCE_MIX)
    parser.add_argument("--rollout-policy", choices=("heuristic", "negamax", "mcts", "random"), default="heuristic")
    parser.add_argument("--mcts-sims", type=int, default=96)
    parser.add_argument("--mcts-time-ms", type=float, default=90.0)
    parser.add_argument("--negamax-time-ms", type=float, default=220.0)
    parser.add_argument(
        "--source-sampling-mode",
        choices=("quota", "game_weighted"),
        default="quota",
        help="quota: try to match final retained source counts to source_mix; game_weighted: sample game sources directly by source_mix.",
    )
    parser.add_argument(
        "--max-source-stall-games",
        type=int,
        default=128,
        help="In quota mode, give up on a source after this many zero-added games and fill the remainder from other sources.",
    )
    parser.add_argument(
        "--max-state-repeats",
        type=int,
        default=1,
        help="Exact board+player repeat cap during teacher sampling. 1 means exact dedup.",
    )
    parser.add_argument("--no-legal-channel", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    source_mix = parse_source_mix(args.source_mix)
    print("[teacher_data] source mix:", ", ".join(f"{n}:{w:.2f}" for n, w in source_mix))

    states, policies, values, metadata = build_teacher_dataset(
        positions=int(args.positions),
        teacher=str(args.teacher),
        teacher_depth=int(args.teacher_depth),
        policy_mode=str(args.policy_mode),
        policy_temperature=float(args.policy_temperature),
        value_mode=str(args.value_mode),
        value_scale=float(args.value_scale),
        source_mix=source_mix,
        include_legal_channel=not args.no_legal_channel,
        seed=int(args.seed),
        mcts_sims=int(args.mcts_sims),
        mcts_time_ms=float(args.mcts_time_ms),
        negamax_time_ms=float(args.negamax_time_ms),
        rollout_policy=str(args.rollout_policy),
        max_state_repeats=int(args.max_state_repeats),
        source_sampling_mode=(
            "quota" if str(args.source_sampling_mode) == "quota" else "game_weighted"
        ),
        max_source_stall_games=int(args.max_source_stall_games),
    )

    output = Path(args.output)
    save_dataset_npz(output, states, policies, values, metadata)

    print("[teacher_data] saved:", output.resolve())
    print(
        f"[teacher_data] shapes states={states.shape}, "
        f"policies={policies.shape}, values={values.shape}"
    )
    print(
        "[teacher_data] metadata:",
        json.dumps(
            {
                "teacher_type": metadata["teacher_type"],
                "teacher_depth": metadata["teacher_depth"],
                "positions_saved": metadata["positions_saved"],
                "channels": metadata["channels"],
                "created_at": metadata["created_at"],
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    _main()
