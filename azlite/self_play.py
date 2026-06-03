"""AlphaZero-lite self-play data generation.

Core rule:
  policy target (pi) must come from MCTS root visit counts, not direct NN output.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from azlite.board import (
    apply_move,
    get_winner,
    is_draw,
    legal_moves,
    terminal_value,
    to_tensor,
)
from azlite.model import ConnectXNet, NeuralEvaluator, load_checkpoint
from azlite.puct_mcts import run_mcts
from azlite.runtime import configure_cpu_runtime, configure_worker_runtime, suggest_cpu_plan


ROWS = 6
COLS = 7
MCTS_TARGET_VERSION = "puct_parent_perspective_v2"


@dataclass
class SelfPlayExample:
    state: np.ndarray  # (C, 6, 7), current-player perspective
    policy_target: np.ndarray  # (7,), from MCTS root visits
    value_target: float  # placeholder, then backfilled as {-1,0,1}
    current_player: int
    move_number: int
    board_before_move: Optional[np.ndarray] = None
    selected_move: int = -1
    visit_counts: np.ndarray | None = None  # (7,)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_visit_counts(visit_counts: np.ndarray, legal: Sequence[int]) -> np.ndarray:
    pi = np.zeros(COLS, dtype=np.float32)
    if not legal:
        return pi
    counts = np.maximum(0.0, np.asarray(visit_counts, dtype=np.float64))
    total = float(np.sum(counts[list(legal)]))
    if total <= 1e-12:
        pi[list(legal)] = 1.0 / len(legal)
        return pi
    pi[list(legal)] = (counts[list(legal)] / total).astype(np.float32)
    return pi


def _sample_from_visits(
    visit_counts: np.ndarray,
    legal: Sequence[int],
    temperature: float,
    rng: np.random.Generator,
) -> int:
    if not legal:
        return 0

    counts = np.maximum(0.0, np.asarray(visit_counts, dtype=np.float64))
    if float(temperature) <= 1e-8:
        best = max(legal, key=lambda c: float(counts[c]))
        return int(best)

    t = max(1e-6, float(temperature))
    scaled = np.zeros(COLS, dtype=np.float64)
    scaled[list(legal)] = np.power(counts[list(legal)], 1.0 / t)
    s = float(np.sum(scaled[list(legal)]))
    if s <= 1e-12:
        probs = np.zeros(COLS, dtype=np.float64)
        probs[list(legal)] = 1.0 / len(legal)
    else:
        probs = scaled / s
    return int(rng.choice(np.arange(COLS), p=probs))


def _temperature_by_move(move_number: int) -> float:
    # First 10 plies: exploratory. Later: sharper exploitation.
    if move_number < 10:
        return 1.0
    return 0.1


def _backfill_values(examples: List[SelfPlayExample], winner: Optional[int]) -> None:
    for ex in examples:
        if winner is None:
            ex.value_target = 0.0
        elif int(winner) == int(ex.current_player):
            ex.value_target = 1.0
        else:
            ex.value_target = -1.0


def play_self_play_game(
    model,
    num_simulations=100,
    device="cpu",
    c_puct=1.5,
    add_dirichlet_noise=True,
    use_tactical_shortcuts=False,
    max_moves=42,
    rng: Optional[np.random.Generator] = None,
) -> List[SelfPlayExample]:
    """Play one self-play game and return (state, pi, z) examples."""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    current_player = 1
    examples: List[SelfPlayExample] = []

    in_channels = int(getattr(model, "in_channels", 3))
    include_legal_channel = in_channels >= 3
    evaluator = NeuralEvaluator(model, device=device)
    rng = rng if rng is not None else np.random.default_rng()

    for move_number in range(int(max_moves)):
        if terminal_value(board, current_player) is not None:
            break

        valid = legal_moves(board)
        if not valid:
            break

        mcts_out = run_mcts(
            board=board,
            current_player=current_player,
            evaluator=evaluator,
            num_simulations=max(1, int(num_simulations)),
            c_puct=float(c_puct),
            temperature=1.0,
            add_dirichlet_noise=bool(add_dirichlet_noise),
            dirichlet_alpha=0.3,
            dirichlet_frac=0.25,
            use_tactical_shortcuts=bool(use_tactical_shortcuts),
            return_root=False,
        )
        visit_counts = np.asarray(mcts_out["visit_counts"], dtype=np.float32)
        policy_target = _normalize_visit_counts(visit_counts, valid)

        tau = _temperature_by_move(move_number)
        selected_move = _sample_from_visits(visit_counts, valid, tau, rng)
        if selected_move not in valid:
            selected_move = int(valid[0])

        state = to_tensor(
            board=board,
            current_player=current_player,
            include_legal_channel=include_legal_channel,
            dtype=np.float32,
        )
        examples.append(
            SelfPlayExample(
                state=state,
                policy_target=policy_target.astype(np.float32, copy=False),
                value_target=0.0,
                current_player=int(current_player),
                move_number=int(move_number),
                board_before_move=board.copy(),
                selected_move=int(selected_move),
                visit_counts=visit_counts.astype(np.float32, copy=False),
            )
        )

        board = apply_move(board, int(selected_move), int(current_player))
        current_player = 2 if current_player == 1 else 1

        if get_winner(board) is not None or is_draw(board):
            break

    winner = get_winner(board)
    _backfill_values(examples, winner)
    return examples


def augment_mirror(example: SelfPlayExample) -> SelfPlayExample:
    """Mirror one self-play example horizontally."""
    mirrored_state = np.flip(np.asarray(example.state), axis=2).copy()
    mirrored_policy = np.flip(np.asarray(example.policy_target), axis=0).copy()
    mirrored_visits = np.flip(np.asarray(example.visit_counts), axis=0).copy()
    mirrored_board = None
    if example.board_before_move is not None:
        mirrored_board = np.fliplr(np.asarray(example.board_before_move)).copy()

    mirrored_move = -1
    if int(example.selected_move) >= 0:
        mirrored_move = (COLS - 1) - int(example.selected_move)

    return SelfPlayExample(
        state=mirrored_state.astype(np.float32, copy=False),
        policy_target=mirrored_policy.astype(np.float32, copy=False),
        value_target=float(example.value_target),
        current_player=int(example.current_player),
        move_number=int(example.move_number),
        board_before_move=mirrored_board,
        selected_move=int(mirrored_move),
        visit_counts=mirrored_visits.astype(np.float32, copy=False),
    )


def _to_arrays(examples: Sequence[SelfPlayExample]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not examples:
        return (
            np.zeros((0, 3, ROWS, COLS), dtype=np.float32),
            np.zeros((0, COLS), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    states = np.stack([np.asarray(ex.state, dtype=np.float32) for ex in examples], axis=0)
    policies = np.stack([np.asarray(ex.policy_target, dtype=np.float32) for ex in examples], axis=0)
    values = np.asarray([float(ex.value_target) for ex in examples], dtype=np.float32)
    return states, policies, values


def _next_selfplay_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("selfplay_*.npz"))
    max_id = 0
    for p in existing:
        stem = p.stem  # selfplay_000001
        tail = stem.split("_")[-1]
        try:
            max_id = max(max_id, int(tail))
        except ValueError:
            continue
    return output_dir / f"selfplay_{max_id + 1:06d}.npz"


def _run_self_play_games_sequential(
    model,
    num_games,
    num_simulations,
    device="cpu",
    augment=True,
    c_puct=1.5,
    add_dirichlet_noise=True,
    use_tactical_shortcuts=False,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    all_examples: List[SelfPlayExample] = []
    game_lengths: List[int] = []
    winner_distribution = {"player1": 0, "player2": 0, "draw": 0}

    for game_idx in range(1, int(num_games) + 1):
        game_rng = np.random.default_rng(int(seed) + int(game_idx * 7919 + num_simulations * 13))
        game_examples = play_self_play_game(
            model=model,
            num_simulations=int(num_simulations),
            device=device,
            c_puct=float(c_puct),
            add_dirichlet_noise=bool(add_dirichlet_noise),
            use_tactical_shortcuts=bool(use_tactical_shortcuts),
            max_moves=42,
            rng=game_rng,
        )
        if not game_examples:
            winner_distribution["draw"] += 1
            continue

        # From initial player-1 perspective: z on first example maps winner quickly.
        z0 = float(game_examples[0].value_target)
        if z0 > 0:
            winner_distribution["player1"] += 1
        elif z0 < 0:
            winner_distribution["player2"] += 1
        else:
            winner_distribution["draw"] += 1

        game_lengths.append(len(game_examples))
        all_examples.extend(game_examples)
        if augment:
            all_examples.extend([augment_mirror(ex) for ex in game_examples])

        print(
            f"[self_play] game {game_idx}/{num_games} "
            f"len={len(game_examples)} total_examples={len(all_examples)}"
        )

    states, policies, values = _to_arrays(all_examples)
    avg_len = float(np.mean(game_lengths)) if game_lengths else 0.0

    meta = {
        "games": int(num_games),
        "examples": int(states.shape[0]),
        "simulations": int(num_simulations),
        "c_puct": float(c_puct),
        "avg_game_length": float(avg_len),
        "winner_distribution": winner_distribution,
        "augment_mirror": bool(augment),
        "dirichlet_noise": bool(add_dirichlet_noise),
        "use_tactical_shortcuts": bool(use_tactical_shortcuts),
        "mcts_target_version": MCTS_TARGET_VERSION,
        "created_at": _utc_now_iso(),
    }
    return states, policies, values, meta


def _save_self_play_npz(
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    meta: Dict[str, object],
    output_dir: str,
) -> Path:
    out_dir = Path(output_dir)
    out_path = _next_selfplay_path(out_dir)
    np.savez_compressed(
        str(out_path),
        states=states,
        policies=policies,
        values=values,
        metadata=json.dumps(meta, ensure_ascii=False),
    )
    print(
        f"[self_play] saved {out_path.resolve()} "
        f"states={states.shape} policies={policies.shape} values={values.shape}"
    )
    return out_path


def _self_play_worker(payload: Dict[str, object]):
    configure_worker_runtime(int(payload.get("worker_cpu_threads", 1) or 1))
    model, _ = _load_model_for_self_play(
        checkpoint=str(payload["checkpoint_path"]) if payload.get("checkpoint_path") else None,
        device=str(payload["device"]),
    )
    return _run_self_play_games_sequential(
        model=model,
        num_games=int(payload["num_games"]),
        num_simulations=int(payload["num_simulations"]),
        device=str(payload["device"]),
        augment=bool(payload["augment"]),
        c_puct=float(payload["c_puct"]),
        add_dirichlet_noise=bool(payload["add_dirichlet_noise"]),
        use_tactical_shortcuts=bool(payload["use_tactical_shortcuts"]),
        seed=int(payload["seed"]),
    )


def generate_self_play_games(
    model,
    num_games,
    num_simulations,
    device="cpu",
    augment=True,
    output_dir="data/selfplay",
    c_puct=1.5,
    add_dirichlet_noise=True,
    use_tactical_shortcuts=False,
    checkpoint_path: Optional[str] = None,
    workers: int = 1,
    main_cpu_threads: Optional[int] = None,
    worker_cpu_threads: int = 1,
    seed: int = 42,
) -> Path:
    """Generate self-play examples and save as NPZ."""
    configure_cpu_runtime(main_cpu_threads)
    n_workers = max(1, int(workers))
    if str(device).lower() != "cpu" and n_workers > 1:
        print("[self_play] non-cpu device detected; forcing workers=1 for stability")
        n_workers = 1
    if n_workers <= 1 or int(num_games) <= 1:
        states, policies, values, meta = _run_self_play_games_sequential(
            model=model,
            num_games=int(num_games),
            num_simulations=int(num_simulations),
            device=device,
            augment=augment,
            c_puct=c_puct,
            add_dirichlet_noise=add_dirichlet_noise,
            use_tactical_shortcuts=use_tactical_shortcuts,
            seed=int(seed),
        )
        meta["parallel_workers"] = 1
        meta["main_cpu_threads"] = int(main_cpu_threads or suggest_cpu_plan()["main_threads"])
        meta["worker_cpu_threads"] = int(worker_cpu_threads)
        return _save_self_play_npz(states, policies, values, meta, output_dir)

    if not checkpoint_path:
        raise ValueError(
            "Parallel self-play requires checkpoint_path so each worker can load the same actor model."
        )

    games_total = max(0, int(num_games))
    shard_games = [games_total // n_workers for _ in range(n_workers)]
    for i in range(games_total % n_workers):
        shard_games[i] += 1
    payloads = []
    for idx, games_i in enumerate(shard_games):
        if int(games_i) <= 0:
            continue
        payloads.append(
            {
                "checkpoint_path": checkpoint_path,
                "num_games": games_i,
                "num_simulations": int(num_simulations),
                "device": str(device),
                "augment": bool(augment),
                "c_puct": float(c_puct),
                "add_dirichlet_noise": bool(add_dirichlet_noise),
                "use_tactical_shortcuts": bool(use_tactical_shortcuts),
                "worker_cpu_threads": int(worker_cpu_threads),
                "seed": int(seed) + (idx * 1009) + int(games_i),
            }
        )

    plan = suggest_cpu_plan()
    print(
        "[self_play] parallel plan total_cpus={cpu} main_threads={main} workers={workers} worker_threads={wt} reserve={reserve}".format(
            cpu=plan["total_cpus"],
            main=(main_cpu_threads or plan["main_threads"]),
            workers=len(payloads),
            wt=max(1, int(worker_cpu_threads)),
            reserve=plan["reserve_cores"],
        )
    )

    states_parts: List[np.ndarray] = []
    policies_parts: List[np.ndarray] = []
    values_parts: List[np.ndarray] = []
    winner_distribution = {"player1": 0, "player2": 0, "draw": 0}
    total_examples = 0
    total_games = 0
    avg_lengths: List[float] = []
    with ProcessPoolExecutor(max_workers=len(payloads)) as executor:
        future_map = {
            executor.submit(_self_play_worker, payload): idx
            for idx, payload in enumerate(payloads, start=1)
        }
        for future in as_completed(future_map):
            worker_idx = future_map[future]
            states_i, policies_i, values_i, meta_i = future.result()
            states_parts.append(np.asarray(states_i, dtype=np.float32))
            policies_parts.append(np.asarray(policies_i, dtype=np.float32))
            values_parts.append(np.asarray(values_i, dtype=np.float32))
            total_examples += int(np.asarray(states_i).shape[0])
            total_games += int(meta_i.get("games", 0) or 0)
            avg_lengths.append(float(meta_i.get("avg_game_length", 0.0) or 0.0))
            for key, value in dict(meta_i.get("winner_distribution") or {}).items():
                winner_distribution[str(key)] = winner_distribution.get(str(key), 0) + int(value)
            print(
                f"[self_play] worker {worker_idx}/{len(payloads)} done "
                f"games={int(meta_i.get('games', 0) or 0)} examples={int(np.asarray(states_i).shape[0])}"
            )

    if states_parts:
        states = np.concatenate(states_parts, axis=0).astype(np.float32, copy=False)
        policies = np.concatenate(policies_parts, axis=0).astype(np.float32, copy=False)
        values = np.concatenate(values_parts, axis=0).astype(np.float32, copy=False)
    else:
        states = np.zeros((0, 3, ROWS, COLS), dtype=np.float32)
        policies = np.zeros((0, COLS), dtype=np.float32)
        values = np.zeros((0,), dtype=np.float32)

    meta = {
        "games": int(total_games),
        "examples": int(total_examples),
        "simulations": int(num_simulations),
        "c_puct": float(c_puct),
        "avg_game_length": float(np.mean(avg_lengths)) if avg_lengths else 0.0,
        "winner_distribution": winner_distribution,
        "augment_mirror": bool(augment),
        "dirichlet_noise": bool(add_dirichlet_noise),
        "use_tactical_shortcuts": bool(use_tactical_shortcuts),
        "mcts_target_version": MCTS_TARGET_VERSION,
        "parallel_workers": int(len(payloads)),
        "main_cpu_threads": int(main_cpu_threads or plan["main_threads"]),
        "worker_cpu_threads": int(worker_cpu_threads),
        "seed": int(seed),
        "created_at": _utc_now_iso(),
    }
    return _save_self_play_npz(states, policies, values, meta, output_dir)


def _load_model_for_self_play(checkpoint: Optional[str], device: str) -> Tuple[ConnectXNet, Dict]:
    if checkpoint:
        model, meta = load_checkpoint(checkpoint, device=device)
        return model, meta
    model = ConnectXNet(in_channels=3)
    model.to(device)
    return model, {}


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate AlphaZero-lite self-play data")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--no-dirichlet-noise", action="store_true")
    parser.add_argument("--use-tactical-shortcuts", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/selfplay",
        help="Directory mode output: selfplay_XXXXXX.npz",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional exact output file path (.npz). If set, generated file is copied to this path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--main-cpu-threads", type=int, default=None)
    parser.add_argument("--worker-cpu-threads", type=int, default=1)
    args = parser.parse_args()
    plan = suggest_cpu_plan()
    if args.workers is None:
        args.workers = int(plan["selfplay_workers"])
    if args.main_cpu_threads is None:
        args.main_cpu_threads = int(plan["main_threads"])

    model, ckpt_meta = _load_model_for_self_play(args.checkpoint, device=args.device)
    if args.checkpoint:
        print(f"[self_play] loaded checkpoint: {args.checkpoint}")
        if ckpt_meta:
            print(
                "[self_play] checkpoint metadata:",
                json.dumps(
                    {
                        "iteration": ckpt_meta.get("iteration"),
                        "train_steps": ckpt_meta.get("train_steps"),
                        "simulations": ckpt_meta.get("simulations"),
                    },
                    ensure_ascii=False,
                ),
            )

    generated_path = generate_self_play_games(
        model=model,
        num_games=int(args.games),
        num_simulations=int(args.simulations),
        device=str(args.device),
        augment=not args.no_augment,
        output_dir=str(args.output_dir),
        c_puct=float(args.c_puct),
        add_dirichlet_noise=not args.no_dirichlet_noise,
        use_tactical_shortcuts=bool(args.use_tactical_shortcuts),
        checkpoint_path=args.checkpoint,
        workers=int(args.workers),
        main_cpu_threads=args.main_cpu_threads,
        worker_cpu_threads=int(args.worker_cpu_threads),
        seed=int(getattr(args, "seed", 42)),
    )

    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = np.load(str(generated_path), allow_pickle=True)
        np.savez_compressed(
            str(target),
            states=np.asarray(data["states"], dtype=np.float32),
            policies=np.asarray(data["policies"], dtype=np.float32),
            values=np.asarray(data["values"], dtype=np.float32),
            metadata=str(data["metadata"]),
        )
        print(f"[self_play] copied to explicit output: {target.resolve()}")


if __name__ == "__main__":
    _main()
