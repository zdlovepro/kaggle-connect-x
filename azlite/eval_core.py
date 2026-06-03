"""Shared evaluation backend for train/evaluate/pipeline.

This module is the single source of truth for:
  - agent construction from spec
  - match execution
  - W/L/D accounting
  - first/second-player split
  - timeout/illegal/error accounting
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence

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
DEFAULT_NEGAMAX_TIME_BUDGET_MS = 1900.0
DEFAULT_MCTS_LITE_TIME_BUDGET_MS = 1900.0

# Keep this visible in logs so future runs can confirm metric provenance.
EVALUATOR_BACKEND = "local_turn_engine_v2"

EVAL_PROFILE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "quick": {
        "candidate_timeout_ms": 2000.0,
        "opponent_timeout_ms": 4000.0,
        "games": {
            "default": 100,
            "random": 20,
            "negamax": 100,
            "mcts_lite": 100,
            "previous_best": 100,
        },
        "diagnostic_only": False,
    },
    "strong_local": {
        "candidate_timeout_ms": 2000.0,
        "opponent_timeout_ms": 5000.0,
        "games": {
            "default": 300,
            "random": 50,
            "negamax": 300,
            "mcts_lite": 300,
            "previous_best": 300,
        },
        "diagnostic_only": True,
    },
    "kaggle_like": {
        "candidate_timeout_ms": 2000.0,
        "opponent_timeout_ms": 2000.0,
        "games": {
            "default": 300,
            "random": 50,
            "negamax": 300,
            "mcts_lite": 300,
            "previous_best": 300,
        },
        "diagnostic_only": False,
    },
}

def derive_internal_time_budget_ms(
    external_timeout_ms: Optional[float],
    default_budget_ms: float,
) -> float:
    """Derive engine-internal search budget from external act-time timeout.

    Rule:
      - when external timeout is unavailable, fallback to default budget
      - internal = external - margin
      - margin = clamp(external * 0.10, 100ms, 300ms)
      - floor at 50ms
      - never exceed external timeout
    """
    if external_timeout_ms is None:
        return float(max(50.0, default_budget_ms))

    ext = float(max(0.0, external_timeout_ms))
    if ext <= 0.0:
        return float(max(50.0, default_budget_ms))

    margin = min(300.0, max(100.0, ext * 0.10))
    budget = ext - margin

    budget = max(50.0, budget)
    budget = min(budget, ext)

    return float(budget)


@dataclass
class UnifiedAgent:
    name: str
    fn: Callable[[Any, Any], int]
    description: str = ""


def _default_cfg() -> SimpleNamespace:
    return SimpleNamespace(rows=ROWS, columns=COLS, inarow=INAROW)


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


def normalize_eval_profile(eval_profile: Optional[str]) -> str:
    key = str(eval_profile or "kaggle_like").strip().lower()
    if key not in EVAL_PROFILE_DEFAULTS:
        raise ValueError(
            f"Unsupported eval profile '{eval_profile}'. "
            f"Expected one of: {', '.join(sorted(EVAL_PROFILE_DEFAULTS.keys()))}"
        )
    return key


def resolve_eval_config(
    eval_profile: str,
    base_games: int = 120,
    candidate_timeout_ms: Optional[float] = None,
    opponent_timeout_ms: Optional[float] = None,
    eval_games_random: Optional[int] = None,
    eval_games_negamax: Optional[int] = None,
    eval_games_mcts_lite: Optional[int] = None,
    eval_games_previous_best: Optional[int] = None,
) -> Dict[str, Any]:
    profile = normalize_eval_profile(eval_profile)
    profile_defaults = EVAL_PROFILE_DEFAULTS[profile]
    games_defaults = dict(profile_defaults.get("games", {}))

    default_games = int(base_games) if int(base_games) > 0 else int(games_defaults.get("default", 24))
    games_by_opponent = {
        "default": default_games,
        "random": int(eval_games_random) if eval_games_random is not None else int(games_defaults.get("random", default_games)),
        "negamax": int(eval_games_negamax) if eval_games_negamax is not None else int(games_defaults.get("negamax", default_games)),
        "mcts_lite": int(eval_games_mcts_lite) if eval_games_mcts_lite is not None else int(games_defaults.get("mcts_lite", default_games)),
        "previous_best": int(eval_games_previous_best) if eval_games_previous_best is not None else int(games_defaults.get("previous_best", default_games)),
    }
    for key, value in list(games_by_opponent.items()):
        games_by_opponent[key] = max(0, int(value))

    cand_ms = float(candidate_timeout_ms) if candidate_timeout_ms is not None else float(profile_defaults["candidate_timeout_ms"])
    opp_ms = float(opponent_timeout_ms) if opponent_timeout_ms is not None else float(profile_defaults["opponent_timeout_ms"])

    return {
        "eval_profile": profile,
        "candidate_timeout_ms": cand_ms,
        "opponent_timeout_ms": opp_ms,
        "candidate_timeout_sec": cand_ms / 1000.0,
        "opponent_timeout_sec": opp_ms / 1000.0,
        "games_by_opponent": games_by_opponent,
        "diagnostic_only": bool(profile_defaults.get("diagnostic_only", False)),
    }


def games_for_opponent(config: Dict[str, Any], opponent: str) -> int:
    games = config.get("games_by_opponent", {})
    if not isinstance(games, dict):
        return int(config.get("games", 24))
    key = str(opponent or "").strip().lower().replace("-", "_")
    return int(games.get(key, games.get("default", 24)))


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


def _build_negamax_agent(time_budget_ms: float = DEFAULT_NEGAMAX_TIME_BUDGET_MS) -> UnifiedAgent:
    if MinimaxBitboardAgent is None:
        raise RuntimeError("MinimaxBitboardAgent is unavailable")
    engine = MinimaxBitboardAgent(
        name="negamax",
        max_depth=20,
        time_budget_ms=float(time_budget_ms),
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


def _build_mcts_lite_agent(time_budget_ms: float = DEFAULT_MCTS_LITE_TIME_BUDGET_MS) -> UnifiedAgent:
    if MCTSAgent is None:
        raise RuntimeError("MCTS-lite agent is unavailable")
    engine = MCTSAgent(name="mcts_lite", c_param=1.414, time_budget_ms=float(time_budget_ms))

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
    time_budget_ms: Optional[float] = None,
) -> UnifiedAgent:
    key = spec.strip().lower()
    external_timeout_ms = float(time_budget_ms) if time_budget_ms is not None else None
    budget_ms = derive_internal_time_budget_ms(
        external_timeout_ms,
        default_budget_ms=DEFAULT_NEGAMAX_TIME_BUDGET_MS,
    )
    if key == "random":
        return _build_random_agent(seed=seed)
    if key == "negamax":
        return _build_negamax_agent(time_budget_ms=budget_ms)
    if key in ("original", "submission", "feature_mcts_lite_original"):
        return _build_original_agent()
    if key in ("heuristic",):
        return _build_heuristic_agent()
    if key in ("mcts_lite", "mcts-lite", "mcts"):
        mcts_budget_ms = derive_internal_time_budget_ms(
            external_timeout_ms,
            default_budget_ms=DEFAULT_MCTS_LITE_TIME_BUDGET_MS,
        )
        return _build_mcts_lite_agent(
            time_budget_ms=mcts_budget_ms
        )
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


def as_unified_agent(agent: Any, default_name: str = "custom_agent") -> UnifiedAgent:
    if isinstance(agent, UnifiedAgent):
        return agent
    if callable(agent):
        return UnifiedAgent(name=default_name, fn=agent, description="callable agent")
    raise TypeError(f"Unsupported agent type: {type(agent)!r}")


def _play_single_game(
    agent_p1: UnifiedAgent,
    agent_p2: UnifiedAgent,
    act_timeout_sec: float = KAGGLE_ACT_TIMEOUT_SEC,
    timeout_sec_by_mark: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    cfg = _default_cfg()
    illegal = {1: 0, 2: 0}
    timeouts = {1: 0, 2: 0}
    timeout_games = {1: 0, 2: 0}
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

        mark_timeout = float(act_timeout_sec)
        if timeout_sec_by_mark is not None:
            mark_timeout = float(timeout_sec_by_mark.get(int(mark), mark_timeout))
        if elapsed > mark_timeout:
            timeouts[mark] += 1
            timeout_games[mark] = 1
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
        "timeout_games": timeout_games,
        "errors": errors,
        "step_times": step_times,
    }


def play_match(
    agent_a: UnifiedAgent,
    agent_b: UnifiedAgent,
    num_games: int,
    swap_sides: bool = True,
    seed: Optional[int] = None,
    act_timeout_sec: float = KAGGLE_ACT_TIMEOUT_SEC,
    candidate_timeout_ms: Optional[float] = None,
    opponent_timeout_ms: Optional[float] = None,
    eval_profile: str = "kaggle_like",
    max_opponent_timeout_rate: float = 0.05,
    max_candidate_timeout_rate: float = 0.01,
    timeout_result_policy: str = "fail_eval",
) -> Dict[str, Any]:
    """Play a match between two agents with unified accounting.

    The returned dict is intentionally verbose because both `train.py` and
    `evaluate.py` consume it directly, and pipeline logs need full detail.
    """
    agent_a = as_unified_agent(agent_a, default_name="agent_a")
    agent_b = as_unified_agent(agent_b, default_name="agent_b")

    n = max(0, int(num_games))
    swap = bool(swap_sides)
    rng = random.Random(seed)
    timeout_policy = str(timeout_result_policy).strip().lower()
    if timeout_policy not in {"loss", "exclude", "fail_eval"}:
        raise ValueError(f"Unsupported timeout_result_policy: {timeout_result_policy}")
    profile_key = normalize_eval_profile(eval_profile)

    default_timeout_ms = float(act_timeout_sec) * 1000.0
    candidate_timeout_ms_final = (
        float(candidate_timeout_ms) if candidate_timeout_ms is not None else default_timeout_ms
    )
    opponent_timeout_ms_final = (
        float(opponent_timeout_ms) if opponent_timeout_ms is not None else default_timeout_ms
    )
    candidate_timeout_sec = candidate_timeout_ms_final / 1000.0
    opponent_timeout_sec = opponent_timeout_ms_final / 1000.0

    if n == 0:
        return {
            "evaluator_backend": EVALUATOR_BACKEND,
            "eval_profile": profile_key,
            "agent_a": agent_a.name,
            "agent_b": agent_b.name,
            "games": 0,
            "num_games": 0,
            "swap_sides": bool(swap_sides),
            "skipped": True,
            "skip_reason": "num_games=0",
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "win_rate": 0.0,
            "reliable_win_rate": 0.0,
            "reliable_games": 0,
            "reliable_wins": 0,
            "reliable_losses": 0,
            "reliable_draws": 0,
            "first_player_games": 0,
            "first_player_wins": 0,
            "first_player_losses": 0,
            "first_player_draws": 0,
            "first_player_win_rate": 0.0,
            "second_player_games": 0,
            "second_player_wins": 0,
            "second_player_losses": 0,
            "second_player_draws": 0,
            "second_player_win_rate": 0.0,
            "side_bias": 0.0,
            "candidate_timeout_moves": 0,
            "opponent_timeout_moves": 0,
            "candidate_timeout_games": 0,
            "opponent_timeout_games": 0,
            "candidate_timeouts": 0,
            "opponent_timeouts": 0,
            "candidate_timeout_rate": 0.0,
            "opponent_timeout_rate": 0.0,
            "candidate_timeout_ms": candidate_timeout_ms_final,
            "opponent_timeout_ms": opponent_timeout_ms_final,
            "candidate_avg_move_ms": 0.0,
            "candidate_max_move_ms": 0.0,
            "opponent_avg_move_ms": 0.0,
            "opponent_max_move_ms": 0.0,
            "invalid_actions": {"candidate": 0, "opponent": 0, "total": 0},
            "timeout_result_policy": timeout_policy,
            "reliable": True,
            "unreliable_reasons": [],
            "reliability": {
                "reliable": True,
                "reliable_win_rate": 0.0,
                "warning": None,
                "candidate_timeout_rate": 0.0,
                "opponent_timeout_rate": 0.0,
                "max_candidate_timeout_rate": float(max_candidate_timeout_rate),
                "max_opponent_timeout_rate": float(max_opponent_timeout_rate),
            },
            "illegal_actions": {"agent_a": 0, "agent_b": 0, "total": 0},
            "timeouts": {"agent_a": 0, "agent_b": 0, "total": 0},
            "errors": {"agent_a": 0, "agent_b": 0, "total": 0},
            "avg_steps": 0.0,
            "avg_step_time_sec": {"agent_a": 0.0, "agent_b": 0.0},
            "p95_step_time_sec": {"agent_a": 0.0, "agent_b": 0.0},
            "step_time_sum_sec": {"agent_a": 0.0, "agent_b": 0.0},
            "step_count": {"agent_a": 0, "agent_b": 0},
        }

    wins = losses = draws = 0
    reliable_wins = reliable_losses = reliable_draws = 0
    reliable_games = 0
    total_steps = 0
    illegal_a = illegal_b = 0
    timeout_moves_a = timeout_moves_b = 0
    timeout_games_a = timeout_games_b = 0
    error_a = error_b = 0
    step_times_a: List[float] = []
    step_times_b: List[float] = []

    first_games = n // 2 if swap else n
    if swap and n % 2 != 0:
        first_games += 1
    second_games = n - first_games if swap else 0
    first_wins = 0
    first_losses = 0
    first_draws = 0
    second_wins = 0
    second_losses = 0
    second_draws = 0

    for g in range(n):
        a_first = (not swap) or (g < first_games)
        if a_first:
            p1, p2 = agent_a, agent_b
            timeout_by_mark = {1: candidate_timeout_sec, 2: opponent_timeout_sec}
        else:
            p1, p2 = agent_b, agent_a
            timeout_by_mark = {1: opponent_timeout_sec, 2: candidate_timeout_sec}

        game = _play_single_game(
            p1,
            p2,
            act_timeout_sec=float(act_timeout_sec),
            timeout_sec_by_mark=timeout_by_mark,
        )
        winner = int(game["winner"])
        total_steps += int(game["steps"])

        if a_first:
            a_timeout_this = int(game["timeouts"][1])
            b_timeout_this = int(game["timeouts"][2])
            if winner == 1:
                wins += 1
                first_wins += 1
            elif winner == 2:
                losses += 1
                first_losses += 1
            else:
                draws += 1
                first_draws += 1
            illegal_a += int(game["illegal"][1])
            illegal_b += int(game["illegal"][2])
            timeout_moves_a += a_timeout_this
            timeout_moves_b += b_timeout_this
            timeout_games_a += int(game.get("timeout_games", {}).get(1, 0))
            timeout_games_b += int(game.get("timeout_games", {}).get(2, 0))
            error_a += int(game["errors"][1])
            error_b += int(game["errors"][2])
            step_times_a.extend(game["step_times"][1])
            step_times_b.extend(game["step_times"][2])
        else:
            a_timeout_this = int(game["timeouts"][2])
            b_timeout_this = int(game["timeouts"][1])
            if winner == 2:
                wins += 1
                second_wins += 1
            elif winner == 1:
                losses += 1
                second_losses += 1
            else:
                draws += 1
                second_draws += 1
            illegal_a += int(game["illegal"][2])
            illegal_b += int(game["illegal"][1])
            timeout_moves_a += a_timeout_this
            timeout_moves_b += b_timeout_this
            timeout_games_a += int(game.get("timeout_games", {}).get(2, 0))
            timeout_games_b += int(game.get("timeout_games", {}).get(1, 0))
            error_a += int(game["errors"][2])
            error_b += int(game["errors"][1])
            step_times_a.extend(game["step_times"][2])
            step_times_b.extend(game["step_times"][1])

        timeout_game = (a_timeout_this + b_timeout_this) > 0
        include_reliable = True
        if timeout_policy == "exclude" and timeout_game:
            include_reliable = False
        if include_reliable:
            reliable_games += 1
            if a_first:
                if winner == 1:
                    reliable_wins += 1
                elif winner == 2:
                    reliable_losses += 1
                else:
                    reliable_draws += 1
            else:
                if winner == 2:
                    reliable_wins += 1
                elif winner == 1:
                    reliable_losses += 1
                else:
                    reliable_draws += 1

        if seed is not None:
            _ = rng.random()

    avg_steps = float(total_steps / n)
    avg_step_time_a = float(np.mean(step_times_a)) if step_times_a else 0.0
    avg_step_time_b = float(np.mean(step_times_b)) if step_times_b else 0.0
    max_step_time_a = float(np.max(step_times_a)) if step_times_a else 0.0
    max_step_time_b = float(np.max(step_times_b)) if step_times_b else 0.0
    p95_step_time_a = _safe_p95(step_times_a)
    p95_step_time_b = _safe_p95(step_times_b)
    step_time_sum_a = float(np.sum(step_times_a)) if step_times_a else 0.0
    step_time_sum_b = float(np.sum(step_times_b)) if step_times_b else 0.0

    first_wr = float(first_wins / max(1, first_games))
    second_wr = float(second_wins / max(1, second_games))
    side_bias = float(abs(first_wr - second_wr))
    win_rate = float(wins / n)

    candidate_timeout_rate = float(timeout_games_a / max(1, n))
    opponent_timeout_rate = float(timeout_games_b / max(1, n))
    reliable_win_rate = float(reliable_wins / max(1, reliable_games))

    unreliable_reasons: List[str] = []
    if candidate_timeout_rate > float(max_candidate_timeout_rate):
        unreliable_reasons.append(
            "candidate_timeout_rate_exceeded: "
            f"{candidate_timeout_rate:.3f} > {float(max_candidate_timeout_rate):.3f}"
        )
    if opponent_timeout_rate > float(max_opponent_timeout_rate):
        unreliable_reasons.append(
            "opponent_timeout_rate_exceeded: "
            f"{opponent_timeout_rate:.3f} > {float(max_opponent_timeout_rate):.3f}"
        )
    if timeout_policy == "fail_eval" and opponent_timeout_rate > float(max_opponent_timeout_rate):
        unreliable_reasons.append("timeout_result_policy=fail_eval")

    reliable = len(unreliable_reasons) == 0
    reliability_warning = None
    if not reliable:
        reliability_warning = "; ".join(unreliable_reasons)

    return {
        "evaluator_backend": EVALUATOR_BACKEND,
        "eval_profile": profile_key,
        "agent_a": agent_a.name,
        "agent_b": agent_b.name,
        "games": n,
        "num_games": n,
        "swap_sides": swap,
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": win_rate,
        "reliable_win_rate": reliable_win_rate,
        "reliable_games": int(reliable_games),
        "reliable_wins": int(reliable_wins),
        "reliable_losses": int(reliable_losses),
        "reliable_draws": int(reliable_draws),
        "first_player_games": int(first_games),
        "first_player_wins": int(first_wins),
        "first_player_losses": int(first_losses),
        "first_player_draws": int(first_draws),
        "first_player_win_rate": first_wr,
        "second_player_games": int(second_games),
        "second_player_wins": int(second_wins),
        "second_player_losses": int(second_losses),
        "second_player_draws": int(second_draws),
        "second_player_win_rate": second_wr,
        "side_bias": side_bias,
        "candidate_timeout_moves": int(timeout_moves_a),
        "opponent_timeout_moves": int(timeout_moves_b),
        "candidate_timeout_games": int(timeout_games_a),
        "opponent_timeout_games": int(timeout_games_b),
        "candidate_timeouts": int(timeout_moves_a),
        "opponent_timeouts": int(timeout_moves_b),
        "candidate_timeout_rate": candidate_timeout_rate,
        "opponent_timeout_rate": opponent_timeout_rate,
        "candidate_timeout_ms": candidate_timeout_ms_final,
        "opponent_timeout_ms": opponent_timeout_ms_final,
        "candidate_avg_move_ms": avg_step_time_a * 1000.0,
        "candidate_max_move_ms": max_step_time_a * 1000.0,
        "opponent_avg_move_ms": avg_step_time_b * 1000.0,
        "opponent_max_move_ms": max_step_time_b * 1000.0,
        "invalid_actions": {
            "candidate": int(illegal_a),
            "opponent": int(illegal_b),
            "total": int(illegal_a + illegal_b),
        },
        "timeout_result_policy": timeout_policy,
        "reliable": bool(reliable),
        "unreliable_reasons": list(unreliable_reasons),
        "reliability": {
            "reliable": bool(reliable),
            "reliable_win_rate": reliable_win_rate,
            "warning": reliability_warning,
            "candidate_timeout_rate": candidate_timeout_rate,
            "opponent_timeout_rate": opponent_timeout_rate,
            "max_candidate_timeout_rate": float(max_candidate_timeout_rate),
            "max_opponent_timeout_rate": float(max_opponent_timeout_rate),
        },
        # Backward-compatible fields:
        "illegal_actions": {
            "agent_a": int(illegal_a),
            "agent_b": int(illegal_b),
            "total": int(illegal_a + illegal_b),
        },
        "timeouts": {
            "agent_a": int(timeout_moves_a),
            "agent_b": int(timeout_moves_b),
            "total": int(timeout_moves_a + timeout_moves_b),
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
