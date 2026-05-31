"""Robust evaluation matrix for ConnectX AlphaZero-lite.

Goal:
  avoid misleading conclusions from random-only win rates.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from azlite.board import (
    apply_move,
    find_immediate_block,
    find_immediate_win,
    get_winner,
    is_draw,
    legal_moves,
    obs_board_to_numpy,
)
from azlite.puct_mcts import HeuristicEvaluator, run_mcts

try:
    import submission as _submission
except Exception:
    _submission = None

try:
    from agents.minimax_bitboard import MinimaxBitboardAgent
except Exception:
    MinimaxBitboardAgent = None

try:
    from agents.mcts_agent import MCTSAgent
except Exception:
    MCTSAgent = None


ROWS = 6
COLS = 7
INAROW = 4
MAX_MOVES = ROWS * COLS
KAGGLE_ACT_TIMEOUT_SEC = 2.0
KAGGLE_P95_TIMEOUT_SEC = 2.0


@dataclass
class UnifiedAgent:
    name: str
    fn: Callable[[Any, Any], int]
    description: str = ""


def _ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _default_cfg() -> SimpleNamespace:
    return SimpleNamespace(rows=ROWS, columns=COLS, inarow=INAROW)


def _load_model_symbols():
    """Lazy-load model module so baseline eval works even without torch installed."""
    try:
        from azlite.model import NeuralEvaluator, load_checkpoint, predict_policy_value
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "Neural checkpoint agents require azlite.model + torch. "
            "Install PyTorch first, or use non-checkpoint agents."
        ) from exc
    return NeuralEvaluator, load_checkpoint, predict_policy_value


def _make_obs(board: np.ndarray, mark: int, step: int) -> SimpleNamespace:
    return SimpleNamespace(
        board=board.reshape(-1).astype(np.int8).tolist(),
        mark=int(mark),
        step=int(step),
        remainingOverageTime=60.0,
    )


def _safe_p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95))


def _build_random_agent(seed: Optional[int] = None) -> UnifiedAgent:
    rng = random.Random(seed)

    def _agent(observation, configuration):
        board = obs_board_to_numpy(
            observation.board,
            rows=int(configuration.rows),
            columns=int(configuration.columns),
        )
        valid = legal_moves(board)
        if not valid:
            return 0
        return int(rng.choice(valid))

    return UnifiedAgent(name="random", fn=_agent, description="uniform legal random")


def _build_negamax_agent() -> UnifiedAgent:
    if MinimaxBitboardAgent is None:
        raise RuntimeError("MinimaxBitboardAgent is unavailable")
    engine = MinimaxBitboardAgent(
        name="negamax",
        max_depth=20,
        time_budget_ms=1900.0,
        use_tt=True,
    )

    def _agent(observation, configuration):
        return int(engine.select_action(observation, configuration))

    return UnifiedAgent(name="negamax", fn=_agent, description="bitboard negamax baseline")


def _build_original_agent() -> UnifiedAgent:
    if _submission is None or not hasattr(_submission, "agent"):
        raise RuntimeError("submission.agent is unavailable")

    def _agent(observation, configuration):
        return int(_submission.agent(observation, configuration))

    return UnifiedAgent(
        name="original",
        fn=_agent,
        description="current feature/mcts-lite original submission agent",
    )


def _build_heuristic_agent() -> UnifiedAgent:
    heuristic = HeuristicEvaluator()

    def _agent(observation, configuration):
        board = obs_board_to_numpy(
            observation.board,
            rows=int(configuration.rows),
            columns=int(configuration.columns),
        )
        mark = int(observation.mark)
        valid = legal_moves(board)
        if not valid:
            return 0

        win_col = find_immediate_win(board, mark)
        if win_col is not None:
            return int(win_col)
        opp = 2 if mark == 1 else 1
        block_col = find_immediate_block(board, mark, opp)
        if block_col is not None:
            return int(block_col)

        policy, _ = heuristic.evaluate(board, mark)
        best = max(valid, key=lambda c: float(policy[c]))
        return int(best)

    return UnifiedAgent(name="heuristic", fn=_agent, description="heuristic evaluator argmax")


def _build_mcts_lite_agent() -> UnifiedAgent:
    if MCTSAgent is None:
        raise RuntimeError("MCTS-lite agent is unavailable")
    engine = MCTSAgent(name="mcts_lite", c_param=1.414, time_budget_ms=1900.0)

    def _agent(observation, configuration):
        return int(engine.select_action(observation, configuration))

    return UnifiedAgent(name="mcts_lite", fn=_agent, description="legacy UCB1 MCTS-lite")


def _build_checkpoint_puct_agent(
    checkpoint_path: str,
    device: str,
    simulations: int,
) -> UnifiedAgent:
    NeuralEvaluator, load_checkpoint, _ = _load_model_symbols()
    model, _ = load_checkpoint(checkpoint_path, device=device)
    evaluator = NeuralEvaluator(model, device=device)

    def _agent(observation, configuration):
        board = obs_board_to_numpy(
            observation.board,
            rows=int(configuration.rows),
            columns=int(configuration.columns),
        )
        mark = int(observation.mark)
        valid = legal_moves(board)
        if not valid:
            return 0
        result = run_mcts(
            board=board,
            current_player=mark,
            evaluator=evaluator,
            num_simulations=max(1, int(simulations)),
            c_puct=1.5,
            temperature=0.0,
            add_dirichlet_noise=False,
            use_tactical_shortcuts=True,
            return_root=False,
        )
        move = int(result["move"])
        if move in valid:
            return move
        return int(valid[0])

    return UnifiedAgent(
        name="checkpoint_puct",
        fn=_agent,
        description=f"checkpoint+puct ({Path(checkpoint_path).name})",
    )


def _build_checkpoint_policy_agent(
    checkpoint_path: str,
    device: str,
) -> UnifiedAgent:
    _, load_checkpoint, predict_policy_value = _load_model_symbols()
    model, _ = load_checkpoint(checkpoint_path, device=device)

    def _agent(observation, configuration):
        board = obs_board_to_numpy(
            observation.board,
            rows=int(configuration.rows),
            columns=int(configuration.columns),
        )
        mark = int(observation.mark)
        valid = legal_moves(board)
        if not valid:
            return 0
        win_col = find_immediate_win(board, mark)
        if win_col is not None:
            return int(win_col)
        opp = 2 if mark == 1 else 1
        block_col = find_immediate_block(board, mark, opp)
        if block_col is not None:
            return int(block_col)

        policy, _ = predict_policy_value(model, board, mark, device=device)
        best = max(valid, key=lambda c: float(policy[c]))
        return int(best)

    return UnifiedAgent(
        name="checkpoint_policy",
        fn=_agent,
        description=f"checkpoint policy-only ({Path(checkpoint_path).name})",
    )


def create_agent(
    spec: str,
    checkpoint: Optional[str] = None,
    previous_best_checkpoint: Optional[str] = None,
    simulations: int = 100,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> UnifiedAgent:
    key = spec.strip().lower()
    if key == "random":
        return _build_random_agent(seed=seed)
    if key == "negamax":
        return _build_negamax_agent()
    if key in ("original", "submission", "feature_mcts_lite_original"):
        return _build_original_agent()
    if key in ("heuristic",):
        return _build_heuristic_agent()
    if key in ("mcts_lite", "mcts-lite", "mcts"):
        return _build_mcts_lite_agent()
    if key in ("checkpoint_puct", "azlite_checkpoint_puct", "candidate"):
        if not checkpoint:
            raise ValueError(f"agent '{spec}' requires --checkpoint")
        return _build_checkpoint_puct_agent(checkpoint, device=device, simulations=simulations)
    if key in ("checkpoint_policy", "policy_only", "policy-only"):
        if not checkpoint:
            raise ValueError(f"agent '{spec}' requires --checkpoint")
        return _build_checkpoint_policy_agent(checkpoint, device=device)
    if key in ("previous_best", "previous_best_puct"):
        if not previous_best_checkpoint:
            raise ValueError("agent 'previous_best' requires --previous-best-checkpoint")
        return _build_checkpoint_puct_agent(
            previous_best_checkpoint,
            device=device,
            simulations=simulations,
        )
    if key in ("previous_best_policy",):
        if not previous_best_checkpoint:
            raise ValueError("agent 'previous_best_policy' requires --previous-best-checkpoint")
        return _build_checkpoint_policy_agent(previous_best_checkpoint, device=device)
    raise ValueError(f"Unsupported agent spec: {spec}")


def _play_single_game(
    agent_p1: UnifiedAgent,
    agent_p2: UnifiedAgent,
    act_timeout_sec: float = KAGGLE_ACT_TIMEOUT_SEC,
) -> Dict[str, Any]:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    cfg = _default_cfg()
    illegal = {1: 0, 2: 0}
    timeouts = {1: 0, 2: 0}
    errors = {1: 0, 2: 0}
    step_times = {1: [], 2: []}
    winner = 0

    for step in range(MAX_MOVES):
        mark = 1 if step % 2 == 0 else 2
        agent = agent_p1 if mark == 1 else agent_p2
        obs = _make_obs(board, mark, step)

        t0 = time.perf_counter()
        action = None
        error_raised = False
        try:
            action = agent.fn(obs, cfg)
        except Exception:
            error_raised = True
        elapsed = float(time.perf_counter() - t0)
        step_times[mark].append(elapsed)

        if elapsed > float(act_timeout_sec):
            timeouts[mark] += 1
            winner = 2 if mark == 1 else 1
            break

        if error_raised:
            errors[mark] += 1
            illegal[mark] += 1
            winner = 2 if mark == 1 else 1
            break

        try:
            col = int(action)
        except Exception:
            illegal[mark] += 1
            winner = 2 if mark == 1 else 1
            break

        valid = legal_moves(board)
        if col not in valid:
            illegal[mark] += 1
            winner = 2 if mark == 1 else 1
            break

        board = apply_move(board, col, mark)
        w = get_winner(board)
        if w is not None:
            winner = int(w)
            break
        if is_draw(board):
            winner = 0
            break

    return {
        "winner": int(winner),
        "steps": int(np.count_nonzero(board)),
        "illegal": illegal,
        "timeouts": timeouts,
        "errors": errors,
        "step_times": step_times,
    }


def play_match(agent_a, agent_b, num_games, swap_sides=True, seed=None):
    """Play match between two unified agents, tracking reliability metrics."""
    if not isinstance(agent_a, UnifiedAgent):
        raise TypeError("agent_a must be UnifiedAgent")
    if not isinstance(agent_b, UnifiedAgent):
        raise TypeError("agent_b must be UnifiedAgent")

    n = max(1, int(num_games))
    swap = bool(swap_sides)
    rng = random.Random(seed)

    wins = losses = draws = 0
    total_steps = 0
    illegal_a = illegal_b = 0
    timeout_a = timeout_b = 0
    error_a = error_b = 0
    step_times_a: List[float] = []
    step_times_b: List[float] = []

    first_games = n // 2 if swap else n
    if swap and n % 2 != 0:
        first_games += 1

    for g in range(n):
        a_first = (not swap) or (g < first_games)
        if a_first:
            p1, p2 = agent_a, agent_b
        else:
            p1, p2 = agent_b, agent_a

        game = _play_single_game(
            p1,
            p2,
            act_timeout_sec=KAGGLE_ACT_TIMEOUT_SEC,
        )
        winner = int(game["winner"])
        total_steps += int(game["steps"])

        if a_first:
            if winner == 1:
                wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
            illegal_a += int(game["illegal"][1])
            illegal_b += int(game["illegal"][2])
            timeout_a += int(game["timeouts"][1])
            timeout_b += int(game["timeouts"][2])
            error_a += int(game["errors"][1])
            error_b += int(game["errors"][2])
            step_times_a.extend(game["step_times"][1])
            step_times_b.extend(game["step_times"][2])
        else:
            if winner == 2:
                wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
            illegal_a += int(game["illegal"][2])
            illegal_b += int(game["illegal"][1])
            timeout_a += int(game["timeouts"][2])
            timeout_b += int(game["timeouts"][1])
            error_a += int(game["errors"][2])
            error_b += int(game["errors"][1])
            step_times_a.extend(game["step_times"][2])
            step_times_b.extend(game["step_times"][1])

        # Add tiny randomized jitter to side alternation only when seed provided:
        # keeps deterministic loop order while still consuming RNG for reproducibility.
        if seed is not None:
            _ = rng.random()

    avg_steps = float(total_steps / n)
    avg_step_time_a = float(np.mean(step_times_a)) if step_times_a else 0.0
    avg_step_time_b = float(np.mean(step_times_b)) if step_times_b else 0.0
    p95_step_time_a = _safe_p95(step_times_a)
    p95_step_time_b = _safe_p95(step_times_b)
    step_time_sum_a = float(np.sum(step_times_a)) if step_times_a else 0.0
    step_time_sum_b = float(np.sum(step_times_b)) if step_times_b else 0.0

    return {
        "agent_a": agent_a.name,
        "agent_b": agent_b.name,
        "num_games": n,
        "swap_sides": swap,
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": float(wins / n),
        "illegal_actions": {
            "agent_a": int(illegal_a),
            "agent_b": int(illegal_b),
            "total": int(illegal_a + illegal_b),
        },
        "timeouts": {
            "agent_a": int(timeout_a),
            "agent_b": int(timeout_b),
            "total": int(timeout_a + timeout_b),
        },
        "errors": {
            "agent_a": int(error_a),
            "agent_b": int(error_b),
            "total": int(error_a + error_b),
        },
        "avg_steps": avg_steps,
        "avg_step_time_sec": {
            "agent_a": avg_step_time_a,
            "agent_b": avg_step_time_b,
        },
        "p95_step_time_sec": {
            "agent_a": p95_step_time_a,
            "agent_b": p95_step_time_b,
        },
        "step_time_sum_sec": {
            "agent_a": step_time_sum_a,
            "agent_b": step_time_sum_b,
        },
        "step_count": {
            "agent_a": len(step_times_a),
            "agent_b": len(step_times_b),
        },
    }


def _aggregate_candidate_metrics(candidate_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    total_illegal = 0
    total_timeout = 0
    total_errors = 0
    total_step_time_sum = 0.0
    total_step_count = 0
    max_p95 = 0.0
    weighted_steps = 0.0
    total_games = 0

    for _, r in candidate_results.items():
        total_illegal += int(r["illegal_actions"]["agent_a"])
        total_timeout += int(r["timeouts"]["agent_a"])
        total_errors += int(r["errors"]["agent_a"])
        total_step_time_sum += float(r["step_time_sum_sec"]["agent_a"])
        total_step_count += int(r["step_count"]["agent_a"])
        max_p95 = max(max_p95, float(r["p95_step_time_sec"]["agent_a"]))
        weighted_steps += float(r["avg_steps"]) * int(r["num_games"])
        total_games += int(r["num_games"])

    avg_step = total_step_time_sum / max(1, total_step_count)
    avg_steps = weighted_steps / max(1, total_games)
    return {
        "total_illegal_agent_a": int(total_illegal),
        "total_timeout_agent_a": int(total_timeout),
        "total_errors_agent_a": int(total_errors),
        "avg_step_time_sec_agent_a": avg_step,
        "p95_step_time_sec_agent_a": max_p95,
        "avg_steps": avg_steps,
    }


def should_promote_candidate(candidate_results, previous_best_results):
    """Gating rules for checkpoint promotion.

    Rules:
      1. vs negamax win-rate must be >= previous best.
      2. vs previous_best must be >= 55%.
      3. random is sanity check only (not gating).
      4. illegal actions must be 0.
      5. avg per-step time < Kaggle limit.
      6. p95 step time acceptable.
    """
    passed = True
    reasons: List[str] = []

    cand_negamax = candidate_results.get("negamax")
    prev_negamax = None if not previous_best_results else previous_best_results.get("negamax")
    cand_prevbest = candidate_results.get("previous_best")

    agg = _aggregate_candidate_metrics(candidate_results)
    if agg["total_illegal_agent_a"] != 0:
        passed = False
        reasons.append(f"illegal actions > 0 ({agg['total_illegal_agent_a']})")

    if agg["total_timeout_agent_a"] != 0:
        passed = False
        reasons.append(f"timeouts > 0 ({agg['total_timeout_agent_a']})")

    if agg["total_errors_agent_a"] != 0:
        passed = False
        reasons.append(f"agent runtime errors > 0 ({agg['total_errors_agent_a']})")

    if agg["avg_step_time_sec_agent_a"] >= KAGGLE_ACT_TIMEOUT_SEC:
        passed = False
        reasons.append(
            f"avg step time {agg['avg_step_time_sec_agent_a']:.4f}s "
            f">= Kaggle limit {KAGGLE_ACT_TIMEOUT_SEC:.2f}s"
        )

    if agg["p95_step_time_sec_agent_a"] >= KAGGLE_P95_TIMEOUT_SEC:
        passed = False
        reasons.append(
            f"p95 step time {agg['p95_step_time_sec_agent_a']:.4f}s "
            f">= acceptable {KAGGLE_P95_TIMEOUT_SEC:.2f}s"
        )

    if cand_negamax is None:
        passed = False
        reasons.append("missing negamax evaluation in candidate_results")
    elif prev_negamax is not None:
        if float(cand_negamax["win_rate"]) < float(prev_negamax["win_rate"]):
            passed = False
            reasons.append(
                "vs negamax win rate lower than previous best "
                f"({cand_negamax['win_rate']:.3f} < {prev_negamax['win_rate']:.3f})"
            )

    if previous_best_results is not None:
        if cand_prevbest is None:
            passed = False
            reasons.append("missing vs previous_best match in candidate_results")
        elif float(cand_prevbest["win_rate"]) < 0.55:
            passed = False
            reasons.append(
                "vs previous_best win rate below 55% "
                f"({cand_prevbest['win_rate']:.3f} < 0.550)"
            )

    if passed and not reasons:
        reasons.append("all gating checks passed")
    return passed, reasons


def _format_console_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "Opponent",
        "W/L/D",
        "WR",
        "Illegal(A)",
        "Timeout(A)",
        "AvgSteps",
        "AvgStepMs(A)",
        "P95StepMs(A)",
    ]
    col_widths = [max(len(h), 12) for h in headers]

    formatted_rows: List[List[str]] = []
    for r in rows:
        line = [
            str(r["opponent"]),
            f"{r['wins']}/{r['losses']}/{r['draws']}",
            f"{r['win_rate']*100:.1f}%",
            str(r["illegal_actions"]["agent_a"]),
            str(r["timeouts"]["agent_a"]),
            f"{r['avg_steps']:.2f}",
            f"{r['avg_step_time_sec']['agent_a']*1000:.1f}",
            f"{r['p95_step_time_sec']['agent_a']*1000:.1f}",
        ]
        formatted_rows.append(line)
        for i, cell in enumerate(line):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_line(cols: Sequence[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols))

    sep = "-+-".join("-" * w for w in col_widths)
    out = [_fmt_line(headers), sep]
    for line in formatted_rows:
        out.append(_fmt_line(line))
    return "\n".join(out)


def _format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    head = (
        "| Opponent | W/L/D | WR | Illegal(A) | Timeout(A) | AvgSteps | "
        "AvgStepMs(A) | P95StepMs(A) |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            "| {opp} | {w}/{l}/{d} | {wr:.1f}% | {ill} | {to} | {steps:.2f} | {avgms:.1f} | {p95ms:.1f} |".format(
                opp=r["opponent"],
                w=r["wins"],
                l=r["losses"],
                d=r["draws"],
                wr=r["win_rate"] * 100.0,
                ill=r["illegal_actions"]["agent_a"],
                to=r["timeouts"]["agent_a"],
                steps=r["avg_steps"],
                avgms=r["avg_step_time_sec"]["agent_a"] * 1000.0,
                p95ms=r["p95_step_time_sec"]["agent_a"] * 1000.0,
            )
        )
    return head + ("\n" + "\n".join(body_lines) if body_lines else "")


def _parse_agent_list(spec: str) -> List[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _warn_random_overfit(candidate_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    r_rand = candidate_results.get("random")
    r_nega = candidate_results.get("negamax")
    if r_rand is None or r_nega is None:
        return None
    if float(r_rand["win_rate"]) >= 0.80 and float(r_nega["win_rate"]) <= 0.55:
        return (
            "random is not a reliable gating metric; model may be overfitting weak play "
            "or relying on tactical shortcuts only."
        )
    return None


def _main() -> None:
    parser = argparse.ArgumentParser(description="Robust evaluation matrix for AlphaZero-lite")
    parser.add_argument("--checkpoint", type=str, default=None, help="Candidate checkpoint path")
    parser.add_argument(
        "--candidate-agent",
        type=str,
        default="checkpoint_puct",
        help="Candidate agent spec (checkpoint_puct/checkpoint_policy/original/mcts_lite/...)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="random,negamax,mcts_lite",
        help="Comma-separated opponents specs",
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--previous-best-checkpoint",
        type=str,
        default=None,
        help="Optional previous-best checkpoint for comparison and gating",
    )
    parser.add_argument("--logs-dir", type=str, default="logs")
    args = parser.parse_args()

    candidate = create_agent(
        args.candidate_agent,
        checkpoint=args.checkpoint,
        previous_best_checkpoint=args.previous_best_checkpoint,
        simulations=int(args.simulations),
        device=args.device,
        seed=int(args.seed),
    )
    opponents = _parse_agent_list(args.opponents)
    if not opponents:
        raise ValueError("--opponents cannot be empty")

    opponent_agents: List[Tuple[str, UnifiedAgent]] = []
    for i, opp_spec in enumerate(opponents):
        opp = create_agent(
            opp_spec,
            checkpoint=args.checkpoint,
            previous_best_checkpoint=args.previous_best_checkpoint,
            simulations=int(args.simulations),
            device=args.device,
            seed=int(args.seed) + i + 1,
        )
        opponent_agents.append((opp_spec, opp))

    candidate_results: Dict[str, Dict[str, Any]] = {}
    for i, (opp_spec, opp_agent) in enumerate(opponent_agents):
        result = play_match(
            candidate,
            opp_agent,
            num_games=int(args.games),
            swap_sides=True,
            seed=int(args.seed) + 1000 + i,
        )
        result["opponent"] = opp_spec
        candidate_results[opp_spec] = result

    # Always evaluate candidate vs previous_best when checkpoint provided.
    if args.previous_best_checkpoint:
        prev_best_opp = create_agent(
            "previous_best",
            checkpoint=args.checkpoint,
            previous_best_checkpoint=args.previous_best_checkpoint,
            simulations=int(args.simulations),
            device=args.device,
            seed=int(args.seed) + 9999,
        )
        r_prev = play_match(
            candidate,
            prev_best_opp,
            num_games=max(20, int(args.games)),
            swap_sides=True,
            seed=int(args.seed) + 2000,
        )
        r_prev["opponent"] = "previous_best"
        candidate_results["previous_best"] = r_prev

    previous_best_results = None
    if args.previous_best_checkpoint:
        previous_best_results = {}
        previous_best = create_agent(
            "previous_best",
            checkpoint=args.checkpoint,
            previous_best_checkpoint=args.previous_best_checkpoint,
            simulations=int(args.simulations),
            device=args.device,
            seed=int(args.seed) + 3000,
        )
        # Compare previous best against same public opponents (except itself).
        for i, (opp_spec, opp_agent) in enumerate(opponent_agents):
            if opp_spec == "previous_best":
                continue
            rr = play_match(
                previous_best,
                opp_agent,
                num_games=int(args.games),
                swap_sides=True,
                seed=int(args.seed) + 4000 + i,
            )
            rr["opponent"] = opp_spec
            previous_best_results[opp_spec] = rr

    passed, gate_reasons = should_promote_candidate(candidate_results, previous_best_results)
    warning = _warn_random_overfit(candidate_results)

    rows = [candidate_results[k] for k in candidate_results.keys()]
    table_text = _format_console_table(rows)
    print("\nEvaluation Matrix")
    print(table_text)

    print("\nGating")
    print(f"passed={passed}")
    for reason in gate_reasons:
        print(f"- {reason}")
    if warning:
        print(f"\nWARNING: {warning}")

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts_now()
    json_path = logs_dir / f"eval_{ts}.json"
    md_path = logs_dir / "eval_latest.md"

    payload = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "candidate_agent": args.candidate_agent,
            "checkpoint": args.checkpoint,
            "opponents": opponents,
            "games": int(args.games),
            "simulations": int(args.simulations),
            "device": args.device,
            "seed": int(args.seed),
            "previous_best_checkpoint": args.previous_best_checkpoint,
        },
        "candidate_results": candidate_results,
        "previous_best_results": previous_best_results,
        "gating": {
            "passed": bool(passed),
            "reasons": gate_reasons,
            "warning": warning,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Evaluation Report",
        "",
        f"- Timestamp: `{payload['created_at']}`",
        f"- Candidate: `{args.candidate_agent}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Games per matchup: `{int(args.games)}`",
        f"- Simulations: `{int(args.simulations)}`",
        "",
        "## Matrix",
        _format_markdown_table(rows),
        "",
        "## Gating",
        f"- Passed: `{passed}`",
    ]
    for reason in gate_reasons:
        md_lines.append(f"- {reason}")
    if warning:
        md_lines.append("")
        md_lines.append("## Warning")
        md_lines.append(
            "> random is not a reliable gating metric; model may be overfitting weak play or relying on tactical shortcuts only."
        )
    md_lines.append("")
    md_lines.append(f"- JSON log: `{json_path}`")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nSaved JSON: {json_path.resolve()}")
    print(f"Saved Markdown: {md_path.resolve()}")


if __name__ == "__main__":
    _main()
