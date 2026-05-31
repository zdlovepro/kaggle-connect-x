"""AlphaZero-lite ablation runner.

Purpose:
  Verify whether each new component truly improves practical strength, instead
  of only increasing complexity.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from azlite.board import (
    apply_move,
    find_immediate_block,
    find_immediate_win,
    get_winner,
    is_draw,
    legal_moves,
    obs_board_to_numpy,
    to_tensor,
)
from azlite.puct_mcts import Evaluator, HeuristicEvaluator, UniformEvaluator, run_mcts

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


@dataclass
class AgentHandle:
    name: str
    fn: Any  # callable(obs, cfg)->int
    notes: str = ""


@dataclass
class CandidateSpec:
    key: str
    group: str
    kind: str
    tactical_shortcuts: Optional[bool]
    scan_params: bool
    checkpoint_role: Optional[str] = None  # teacher / selfplay


def _ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(rows=ROWS, columns=COLS, inarow=INAROW)


def _obs(board: np.ndarray, mark: int, step: int) -> SimpleNamespace:
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


def _parse_int_list(spec: str, default: Sequence[int]) -> List[int]:
    if not spec:
        return list(default)
    out: List[int] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out if out else list(default)


def _parse_float_list(spec: str, default: Sequence[float]) -> List[float]:
    if not spec:
        return list(default)
    out: List[float] = []
    for tok in str(spec).split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    return out if out else list(default)


def _make_random_agent(seed: int) -> AgentHandle:
    rng = random.Random(seed)

    def _fn(observation, configuration):
        board = obs_board_to_numpy(observation.board, rows=int(configuration.rows), columns=int(configuration.columns))
        moves = legal_moves(board)
        if not moves:
            return 0
        return int(rng.choice(moves))

    return AgentHandle(name="random", fn=_fn, notes="uniform legal random")


def _make_negamax_agent() -> AgentHandle:
    if MinimaxBitboardAgent is None:
        raise RuntimeError("MinimaxBitboardAgent is unavailable")
    engine = MinimaxBitboardAgent(
        name="negamax",
        max_depth=20,
        time_budget_ms=1900.0,
        use_tt=True,
    )

    def _fn(observation, configuration):
        return int(engine.select_action(observation, configuration))

    return AgentHandle(name="negamax", fn=_fn, notes="bitboard negamax")


def _make_original_agent() -> AgentHandle:
    if _submission is None or not hasattr(_submission, "agent"):
        raise RuntimeError("submission.agent unavailable")

    def _fn(observation, configuration):
        return int(_submission.agent(observation, configuration))

    return AgentHandle(name="original", fn=_fn, notes="feature/mcts-lite original")


def _make_mcts_lite_agent() -> AgentHandle:
    if MCTSAgent is None:
        raise RuntimeError("MCTSAgent unavailable")
    engine = MCTSAgent(name="mcts_lite", c_param=1.414, time_budget_ms=1900.0)

    def _fn(observation, configuration):
        return int(engine.select_action(observation, configuration))

    return AgentHandle(name="mcts_lite", fn=_fn, notes="legacy UCB1 MCTS")


class RandomNetworkEvaluator(Evaluator):
    """Torch-free random network evaluator for ablation baseline."""

    def __init__(self, seed: int = 1234, channels: int = 3):
        self.channels = int(channels)
        rng = np.random.default_rng(int(seed))
        self.w_policy = (rng.standard_normal((7, self.channels * 6 * 7)) * 0.05).astype(np.float32)
        self.b_policy = (rng.standard_normal((7,)) * 0.01).astype(np.float32)
        self.w_value = (rng.standard_normal((self.channels * 6 * 7,)) * 0.05).astype(np.float32)
        self.b_value = float(rng.standard_normal() * 0.01)

    def evaluate(self, board: np.ndarray, current_player: int) -> Tuple[np.ndarray, float]:
        x = to_tensor(board, current_player=current_player, include_legal_channel=self.channels >= 3, dtype=np.float32)
        flat = x.reshape(-1).astype(np.float32, copy=False)
        logits = self.w_policy @ flat + self.b_policy
        legal = legal_moves(board)
        probs = np.zeros(7, dtype=np.float32)
        if legal:
            ll = np.asarray([logits[c] for c in legal], dtype=np.float64)
            ll -= np.max(ll)
            ee = np.exp(ll)
            denom = float(ee.sum())
            if denom <= 1e-12:
                probs[legal] = 1.0 / len(legal)
            else:
                vals = ee / denom
                for i, c in enumerate(legal):
                    probs[c] = np.float32(vals[i])
        value = float(np.tanh(np.dot(self.w_value, flat) + self.b_value))
        return probs, max(-1.0, min(1.0, value))


def _load_model_symbols():
    try:
        from azlite.model import ConnectXNet, NeuralEvaluator, load_checkpoint, predict_policy_value
    except Exception as exc:
        raise RuntimeError(
            "Neural configs require azlite.model + torch in current runtime."
        ) from exc
    return ConnectXNet, NeuralEvaluator, load_checkpoint, predict_policy_value


def _puct_move(
    board: np.ndarray,
    mark: int,
    evaluator: Evaluator,
    simulations: int,
    c_puct: float,
    temperature: float,
    tactical_shortcuts: bool,
) -> int:
    valid = legal_moves(board)
    if not valid:
        return 0
    out = run_mcts(
        board=board,
        current_player=mark,
        evaluator=evaluator,
        num_simulations=max(1, int(simulations)),
        c_puct=float(c_puct),
        temperature=float(temperature),
        add_dirichlet_noise=False,
        use_tactical_shortcuts=bool(tactical_shortcuts),
        return_root=False,
    )
    mv = int(out["move"])
    if mv in valid:
        return mv
    return int(valid[0])


def _policy_argmax_move(policy: np.ndarray, board: np.ndarray) -> int:
    valid = legal_moves(board)
    if not valid:
        return 0
    best = max(valid, key=lambda c: float(policy[c]))
    return int(best)


def _make_candidate_agent(
    spec: CandidateSpec,
    params: Mapping[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> AgentHandle:
    key = spec.key

    if spec.kind == "original":
        return _make_original_agent()

    sims = int(params.get("simulations", 0) or 0)
    c_puct = float(params.get("c_puct", 1.5))
    temp = float(params.get("temperature", 0.0))
    tactical = bool(spec.tactical_shortcuts) if spec.tactical_shortcuts is not None else True

    if spec.kind == "uniform_puct":
        evaluator = UniformEvaluator()

        def _fn(observation, configuration):
            board = obs_board_to_numpy(observation.board, rows=int(configuration.rows), columns=int(configuration.columns))
            return _puct_move(board, int(observation.mark), evaluator, sims, c_puct, temp, tactical)

        return AgentHandle(name=key, fn=_fn, notes="UniformEvaluator + PUCT")

    if spec.kind == "heuristic_puct":
        evaluator = HeuristicEvaluator()

        def _fn(observation, configuration):
            board = obs_board_to_numpy(observation.board, rows=int(configuration.rows), columns=int(configuration.columns))
            return _puct_move(board, int(observation.mark), evaluator, sims, c_puct, temp, tactical)

        return AgentHandle(name=key, fn=_fn, notes="HeuristicEvaluator + PUCT")

    if spec.kind == "randomnet_puct":
        evaluator = RandomNetworkEvaluator(seed=seed + 17)

        def _fn(observation, configuration):
            board = obs_board_to_numpy(observation.board, rows=int(configuration.rows), columns=int(configuration.columns))
            return _puct_move(board, int(observation.mark), evaluator, sims, c_puct, temp, tactical)

        return AgentHandle(name=key, fn=_fn, notes="RandomNetworkEvaluator + PUCT")

    # Neural checkpoint-driven configs
    ckpt_path: Optional[str] = None
    if spec.checkpoint_role == "teacher":
        ckpt_path = args.teacher_checkpoint
    elif spec.checkpoint_role == "selfplay":
        ckpt_path = args.selfplay_checkpoint
    if not ckpt_path:
        raise RuntimeError(f"{spec.kind} requires checkpoint_role={spec.checkpoint_role}")

    ConnectXNet, NeuralEvaluator, load_checkpoint, predict_policy_value = _load_model_symbols()
    del ConnectXNet  # not used directly
    model, _meta = load_checkpoint(str(ckpt_path), device=str(args.device))

    if spec.kind.endswith("policy"):
        def _fn(observation, configuration):
            board = obs_board_to_numpy(observation.board, rows=int(configuration.rows), columns=int(configuration.columns))
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
            policy, _v = predict_policy_value(model, board, mark, device=str(args.device))
            return _policy_argmax_move(np.asarray(policy, dtype=np.float32), board)

        return AgentHandle(name=key, fn=_fn, notes=f"{spec.checkpoint_role} checkpoint policy-only")

    evaluator = NeuralEvaluator(model, device=str(args.device))

    def _fn(observation, configuration):
        board = obs_board_to_numpy(observation.board, rows=int(configuration.rows), columns=int(configuration.columns))
        return _puct_move(board, int(observation.mark), evaluator, sims, c_puct, temp, tactical)

    return AgentHandle(name=key, fn=_fn, notes=f"{spec.checkpoint_role} checkpoint + PUCT")


def _create_opponents(args: argparse.Namespace) -> Dict[str, AgentHandle]:
    out: Dict[str, AgentHandle] = {}
    wanted = [s.strip() for s in str(args.opponents).split(",") if s.strip()]
    for spec in wanted:
        key = spec.lower()
        if key == "random":
            out["random"] = _make_random_agent(seed=int(args.seed) + 1)
        elif key == "negamax":
            out["negamax"] = _make_negamax_agent()
        elif key == "original":
            out["original"] = _make_original_agent()
        elif key == "mcts_lite":
            out["mcts_lite"] = _make_mcts_lite_agent()
        elif key == "previous_best_azlite":
            if not args.previous_best_checkpoint:
                print("[ablation] skip opponent previous_best_azlite (no --previous-best-checkpoint)")
                continue
            spec_obj = CandidateSpec(
                key="previous_best_azlite",
                group="opponent",
                kind="selfplay_puct",
                tactical_shortcuts=True,
                scan_params=False,
                checkpoint_role="selfplay",
            )
            # temporary namespace override to reuse builder cleanly
            tmp_args = argparse.Namespace(**vars(args))
            tmp_args.selfplay_checkpoint = args.previous_best_checkpoint
            opponent = _make_candidate_agent(
                spec_obj,
                params={
                    "simulations": int(args.previous_best_simulations),
                    "c_puct": float(args.previous_best_c_puct),
                    "temperature": 0.0,
                },
                args=tmp_args,
                seed=int(args.seed) + 99,
            )
            opponent.name = "previous_best_azlite"
            out["previous_best_azlite"] = opponent
        else:
            print(f"[ablation] unknown opponent spec='{spec}', skip")
    return out


def _candidate_specs() -> List[CandidateSpec]:
    specs: List[CandidateSpec] = []
    specs.append(
        CandidateSpec(
            key="original_feature_mcts_lite",
            group="original",
            kind="original",
            tactical_shortcuts=None,
            scan_params=False,
        )
    )

    # PUCT evaluator baselines with/without tactical shortcuts.
    for t in (True, False):
        specs.append(
            CandidateSpec(
                key=f"uniform_puct_tactical_{'on' if t else 'off'}",
                group="uniform_puct",
                kind="uniform_puct",
                tactical_shortcuts=t,
                scan_params=True,
            )
        )
        specs.append(
            CandidateSpec(
                key=f"heuristic_puct_tactical_{'on' if t else 'off'}",
                group="heuristic_puct",
                kind="heuristic_puct",
                tactical_shortcuts=t,
                scan_params=True,
            )
        )
        specs.append(
            CandidateSpec(
                key=f"randomnet_puct_tactical_{'on' if t else 'off'}",
                group="randomnet_puct",
                kind="randomnet_puct",
                tactical_shortcuts=t,
                scan_params=True,
            )
        )

        specs.append(
            CandidateSpec(
                key=f"teacher_puct_tactical_{'on' if t else 'off'}",
                group="teacher_puct",
                kind="teacher_puct",
                tactical_shortcuts=t,
                scan_params=True,
                checkpoint_role="teacher",
            )
        )
        specs.append(
            CandidateSpec(
                key=f"selfplay_puct_tactical_{'on' if t else 'off'}",
                group="selfplay_puct",
                kind="selfplay_puct",
                tactical_shortcuts=t,
                scan_params=True,
                checkpoint_role="selfplay",
            )
        )

    # Policy-only ablations.
    specs.append(
        CandidateSpec(
            key="teacher_policy_only",
            group="teacher_policy",
            kind="teacher_policy",
            tactical_shortcuts=None,
            scan_params=False,
            checkpoint_role="teacher",
        )
    )
    specs.append(
        CandidateSpec(
            key="selfplay_policy_only",
            group="selfplay_policy",
            kind="selfplay_policy",
            tactical_shortcuts=None,
            scan_params=False,
            checkpoint_role="selfplay",
        )
    )
    return specs


def _expand_spec_params(
    spec: CandidateSpec,
    scan_sims: Sequence[int],
    scan_cpuct: Sequence[float],
    scan_temp: Sequence[float],
) -> List[Tuple[str, Dict[str, Any]]]:
    if not spec.scan_params:
        return [(spec.key, {"simulations": None, "c_puct": None, "temperature": None})]
    out: List[Tuple[str, Dict[str, Any]]] = []
    for sims, cp, temp in product(scan_sims, scan_cpuct, scan_temp):
        name = f"{spec.key}|sim={int(sims)}|cp={cp:g}|temp={temp:g}"
        out.append(
            (
                name,
                {"simulations": int(sims), "c_puct": float(cp), "temperature": float(temp)},
            )
        )
    return out


def _play_single_game(
    agent_p1: AgentHandle,
    agent_p2: AgentHandle,
) -> Dict[str, Any]:
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    cfg = _cfg()
    illegal = {1: 0, 2: 0}
    timeouts = {1: 0, 2: 0}
    errors = {1: 0, 2: 0}
    step_times = {1: [], 2: []}
    winner = 0

    for step in range(MAX_MOVES):
        mark = 1 if step % 2 == 0 else 2
        who = agent_p1 if mark == 1 else agent_p2
        observation = _obs(board, mark, step)
        t0 = time.perf_counter()
        action = None
        raised = False
        try:
            action = who.fn(observation, cfg)
        except Exception:
            raised = True
        elapsed = float(time.perf_counter() - t0)
        step_times[mark].append(elapsed)

        if elapsed > KAGGLE_ACT_TIMEOUT_SEC:
            timeouts[mark] += 1
            winner = 2 if mark == 1 else 1
            break
        if raised:
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


def play_match(
    candidate: AgentHandle,
    opponent: AgentHandle,
    num_games: int,
    swap_sides: bool,
    seed: int,
) -> Dict[str, Any]:
    n = max(1, int(num_games))
    swap = bool(swap_sides)
    rng = random.Random(int(seed))

    wins = losses = draws = 0
    total_steps = 0
    illegal_c = timeout_c = errors_c = 0
    step_times_c: List[float] = []
    fp_games = sp_games = fp_wins = sp_wins = 0

    first_games = n // 2 if swap else n
    if swap and n % 2 != 0:
        first_games += 1

    for g in range(n):
        c_first = (not swap) or (g < first_games)
        if c_first:
            p1, p2 = candidate, opponent
        else:
            p1, p2 = opponent, candidate

        game = _play_single_game(p1, p2)
        winner = int(game["winner"])
        total_steps += int(game["steps"])

        if c_first:
            fp_games += 1
            if winner == 1:
                wins += 1
                fp_wins += 1
            elif winner == 2:
                losses += 1
            else:
                draws += 1
            illegal_c += int(game["illegal"][1])
            timeout_c += int(game["timeouts"][1])
            errors_c += int(game["errors"][1])
            step_times_c.extend(game["step_times"][1])
        else:
            sp_games += 1
            if winner == 2:
                wins += 1
                sp_wins += 1
            elif winner == 1:
                losses += 1
            else:
                draws += 1
            illegal_c += int(game["illegal"][2])
            timeout_c += int(game["timeouts"][2])
            errors_c += int(game["errors"][2])
            step_times_c.extend(game["step_times"][2])

        # deterministic random consumption to keep seeds stable
        _ = rng.random()

    avg_steps = float(total_steps / n)
    avg_move_time = float(np.mean(step_times_c)) if step_times_c else 0.0
    p95_move_time = _safe_p95(step_times_c)

    return {
        "games": int(n),
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "win_rate": float(wins / n),
        "first_player_win_rate": float(fp_wins / max(1, fp_games)),
        "second_player_win_rate": float(sp_wins / max(1, sp_games)),
        "avg_steps": avg_steps,
        "avg_move_time_sec": avg_move_time,
        "p95_move_time_sec": p95_move_time,
        "illegal_moves": int(illegal_c),
        "timeouts": int(timeout_c),
        "errors": int(errors_c),
    }


def _row_to_md(row: Mapping[str, Any]) -> str:
    return (
        f"| {row['config_name']} | {row['opponent']} | {row['games']} | "
        f"{row['wins']}/{row['losses']}/{row['draws']} | "
        f"{row['win_rate']*100:.1f}% | {row['first_player_win_rate']*100:.1f}% | "
        f"{row['second_player_win_rate']*100:.1f}% | "
        f"{row['avg_move_time_sec']*1000:.1f} | {row['p95_move_time_sec']*1000:.1f} | "
        f"{row['illegal_moves']} |"
    )


def _score_row(row: Mapping[str, Any]) -> float:
    opp = str(row["opponent"])
    w = 0.05
    if opp == "negamax":
        w = 0.40
    elif opp == "original":
        w = 0.35
    elif opp == "mcts_lite":
        w = 0.30
    elif opp == "previous_best_azlite":
        w = 0.20
    penalty = 0.0
    if int(row.get("illegal_moves", 0)) > 0:
        penalty += 0.50
    if float(row.get("p95_move_time_sec", 0.0)) >= KAGGLE_ACT_TIMEOUT_SEC:
        penalty += 0.25
    return w * float(row["win_rate"]) - penalty


def _collect_best_by_group(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Tuple[str, float]]:
    grouped: Dict[str, Dict[str, float]] = {}
    for r in rows:
        g = str(r["group"])
        c = str(r["config_name"])
        grouped.setdefault(g, {})
        grouped[g][c] = grouped[g].get(c, 0.0) + _score_row(r)
    out: Dict[str, Tuple[str, float]] = {}
    for g, scores in grouped.items():
        best_name = max(scores, key=lambda k: scores[k])
        out[g] = (best_name, float(scores[best_name]))
    return out


def _mean_wr(rows: Sequence[Mapping[str, Any]], name_substr: str, opponent: Optional[str] = None) -> Optional[float]:
    vals = []
    for r in rows:
        if name_substr not in str(r["config_name"]):
            continue
        if opponent is not None and str(r["opponent"]) != opponent:
            continue
        vals.append(float(r["win_rate"]))
    if not vals:
        return None
    return float(np.mean(vals))


def _tactical_delta(rows: Sequence[Mapping[str, Any]], group_prefix: str, opponent: str) -> Optional[float]:
    on_vals: List[float] = []
    off_vals: List[float] = []
    for r in rows:
        c = str(r["config_name"])
        if group_prefix not in c:
            continue
        if str(r["opponent"]) != opponent:
            continue
        if "_tactical_on" in c:
            on_vals.append(float(r["win_rate"]))
        elif "_tactical_off" in c:
            off_vals.append(float(r["win_rate"]))
    if not on_vals or not off_vals:
        return None
    return float(np.mean(on_vals) - np.mean(off_vals))


def _conclusions(rows: Sequence[Mapping[str, Any]]) -> Dict[str, str]:
    if not rows:
        return {
            "component_gain": "No completed rows.",
            "teacher_bootstrap": "No completed rows.",
            "self_play": "No completed rows.",
            "puct_vs_original": "No completed rows.",
            "tactical_shortcuts": "No completed rows.",
            "recommended_defaults": "No completed rows.",
        }

    best_by_group = _collect_best_by_group(rows)
    best_group = max(best_by_group, key=lambda g: best_by_group[g][1]) if best_by_group else "n/a"
    component_gain = (
        f"Largest component gain appears in group '{best_group}' "
        f"(best config={best_by_group.get(best_group, ('n/a', 0.0))[0]})."
    )

    teacher_puct = _mean_wr(rows, "teacher_puct")
    randomnet_puct = _mean_wr(rows, "randomnet_puct")
    if teacher_puct is None or randomnet_puct is None:
        teacher_bootstrap = "Insufficient teacher/randomnet rows to judge bootstrap effect."
    else:
        delta = teacher_puct - randomnet_puct
        teacher_bootstrap = (
            f"Teacher bootstrap {'helps' if delta > 0 else 'does not help'} on average "
            f"(delta WR={delta*100:.1f} pts vs randomnet_puct)."
        )

    selfplay_puct = _mean_wr(rows, "selfplay_puct")
    teacher_puct2 = _mean_wr(rows, "teacher_puct")
    if selfplay_puct is None or teacher_puct2 is None:
        self_play = "Insufficient selfplay/teacher rows to judge self-play effect."
    else:
        delta = selfplay_puct - teacher_puct2
        self_play = (
            f"Self-play {'helps' if delta > 0 else 'does not help'} on average "
            f"(delta WR={delta*100:.1f} pts vs teacher_puct)."
        )

    puct_best = None
    puct_score = -1e9
    orig_score = 0.0
    by_config: Dict[str, float] = {}
    for r in rows:
        c = str(r["config_name"])
        by_config[c] = by_config.get(c, 0.0) + _score_row(r)
    for c, s in by_config.items():
        if c.startswith("original_feature_mcts_lite"):
            orig_score = s
        if "puct" in c and s > puct_score:
            puct_score = s
            puct_best = c
    if puct_best is None:
        puct_vs_original = "No PUCT row completed."
    else:
        stronger = puct_score > orig_score
        puct_vs_original = (
            f"Best PUCT config ({puct_best}) is {'stronger' if stronger else 'weaker'} "
            f"than original by score delta={(puct_score - orig_score):.3f}."
        )

    d_rand = _tactical_delta(rows, "puct", "random")
    d_nega = _tactical_delta(rows, "puct", "negamax")
    if d_rand is None or d_nega is None:
        tactical_shortcuts = "Insufficient tactical on/off rows to judge shortcut effect."
    else:
        tactical_shortcuts = (
            f"Tactical shortcuts delta: random={d_rand*100:.1f} pts, negamax={d_nega*100:.1f} pts. "
            + (
                "Likely mainly boosting weak-opponent win-rate."
                if d_rand > 0 and d_nega <= 0
                else "Benefits are not random-only."
            )
        )

    # recommended defaults: best valid config (no illegal + under timeout p95)
    valid_configs: Dict[str, float] = {}
    for r in rows:
        if int(r.get("illegal_moves", 0)) != 0:
            continue
        if float(r.get("p95_move_time_sec", 0.0)) >= KAGGLE_ACT_TIMEOUT_SEC:
            continue
        c = str(r["config_name"])
        valid_configs[c] = valid_configs.get(c, 0.0) + _score_row(r)
    if valid_configs:
        rec = max(valid_configs, key=lambda k: valid_configs[k])
        recommended_defaults = (
            f"Recommend submission default from: {rec} "
            "(best composite among legal + non-timeout candidates)."
        )
    else:
        recommended_defaults = (
            "No config satisfied strict legality+latency constraints; keep original agent until retuned."
        )

    return {
        "component_gain": component_gain,
        "teacher_bootstrap": teacher_bootstrap,
        "self_play": self_play,
        "puct_vs_original": puct_vs_original,
        "tactical_shortcuts": tactical_shortcuts,
        "recommended_defaults": recommended_defaults,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero-lite ablation runner")
    parser.add_argument("--teacher-checkpoint", type=str, default=None)
    parser.add_argument("--selfplay-checkpoint", type=str, default=None)
    parser.add_argument("--previous-best-checkpoint", type=str, default=None)

    parser.add_argument("--opponents", type=str, default="random,negamax,original,previous_best_azlite")
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--swap-sides", action="store_true", default=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--simulations", type=str, default="25,50,100,200")
    parser.add_argument("--c-puct", type=str, default="0.8,1.2,1.5,2.0")
    parser.add_argument("--temperature", type=str, default="0,0.5,1.0")
    parser.add_argument("--max-configs", type=int, default=0, help="0 means run all.")

    parser.add_argument("--previous-best-simulations", type=int, default=100)
    parser.add_argument("--previous-best-c-puct", type=float, default=1.5)
    parser.add_argument("--logs-dir", type=str, default="logs")
    args = parser.parse_args()

    scan_sims = _parse_int_list(args.simulations, default=[25, 50, 100, 200])
    scan_cpuct = _parse_float_list(args.c_puct, default=[0.8, 1.2, 1.5, 2.0])
    scan_temp = _parse_float_list(args.temperature, default=[0.0, 0.5, 1.0])

    opponents = _create_opponents(args)
    if not opponents:
        raise RuntimeError("No valid opponents resolved; check --opponents and dependencies.")

    specs = _candidate_specs()
    expanded: List[Tuple[CandidateSpec, str, Dict[str, Any]]] = []
    for spec in specs:
        for name, params in _expand_spec_params(spec, scan_sims, scan_cpuct, scan_temp):
            expanded.append((spec, name, params))

    if int(args.max_configs) > 0:
        expanded = expanded[: int(args.max_configs)]

    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    total = len(expanded)
    for idx, (spec, cfg_name, params) in enumerate(expanded, start=1):
        try:
            candidate = _make_candidate_agent(spec, params=params, args=args, seed=int(args.seed) + idx)
        except Exception as exc:
            skipped.append(
                {
                    "config_name": cfg_name,
                    "group": spec.group,
                    "reason": f"build_failed: {exc}",
                }
            )
            print(f"[ablation] skip {cfg_name}: {exc}")
            continue

        for j, (opp_name, opponent) in enumerate(opponents.items(), start=1):
            match_seed = int(args.seed) + idx * 1000 + j
            res = play_match(
                candidate=candidate,
                opponent=opponent,
                num_games=int(args.games),
                swap_sides=bool(args.swap_sides),
                seed=match_seed,
            )
            row = {
                "config_name": cfg_name,
                "group": spec.group,
                "kind": spec.kind,
                "tactical_shortcuts": spec.tactical_shortcuts,
                "simulations": params.get("simulations"),
                "c_puct": params.get("c_puct"),
                "temperature": params.get("temperature"),
                "opponent": opp_name,
                **res,
            }
            rows.append(row)
            print(
                f"[ablation] {idx}/{total} {cfg_name} vs {opp_name}: "
                f"{res['wins']}W/{res['losses']}L/{res['draws']}D "
                f"WR={res['win_rate']*100:.1f}% illegal={res['illegal_moves']}"
            )

    conclusions = _conclusions(rows)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts_now()
    json_path = logs_dir / f"ablation_{ts}.json"
    md_path = logs_dir / "ablation_latest.md"

    payload = {
        "created_at": datetime.now().isoformat(),
        "config": {
            "teacher_checkpoint": args.teacher_checkpoint,
            "selfplay_checkpoint": args.selfplay_checkpoint,
            "previous_best_checkpoint": args.previous_best_checkpoint,
            "opponents": list(opponents.keys()),
            "games": int(args.games),
            "scan": {
                "simulations": scan_sims,
                "c_puct": scan_cpuct,
                "temperature": scan_temp,
            },
            "seed": int(args.seed),
        },
        "rows": rows,
        "skipped": skipped,
        "conclusions": conclusions,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# AlphaZero-lite Ablation",
        "",
        f"- Generated: `{payload['created_at']}`",
        f"- Opponents: `{', '.join(payload['config']['opponents'])}`",
        f"- Games per matchup: `{payload['config']['games']}`",
        "",
        "| config name | opponent | games | W/L/D | win rate | first-player win rate | second-player win rate | avg move time (ms) | p95 move time (ms) | illegal moves |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md_lines.append(_row_to_md(r))

    md_lines.extend(
        [
            "",
            "## Conclusions",
            f"1. 哪个组件提升最大？ {conclusions['component_gain']}",
            f"2. 是否 teacher bootstrap 有帮助？ {conclusions['teacher_bootstrap']}",
            f"3. 是否 self-play 有帮助？ {conclusions['self_play']}",
            f"4. 是否 PUCT 比原始 MCTS-lite 更强？ {conclusions['puct_vs_original']}",
            f"5. tactical shortcuts 是否只提升 random 胜率？ {conclusions['tactical_shortcuts']}",
            f"6. 推荐 Kaggle submission 默认参数。 {conclusions['recommended_defaults']}",
        ]
    )
    if skipped:
        md_lines.append("")
        md_lines.append("## Skipped")
        for s in skipped:
            md_lines.append(f"- {s['config_name']}: {s['reason']}")
    md_lines.append("")
    md_lines.append(f"- JSON: `{json_path}`")

    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[ablation] json={json_path.resolve()}")
    print(f"[ablation] md={md_path.resolve()}")


if __name__ == "__main__":
    _main()
