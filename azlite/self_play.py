"""AlphaZero-lite self-play data generation.

Core rule:
  policy target (pi) must come from MCTS root visit counts, not direct NN output.
"""

from __future__ import annotations

import argparse
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


ROWS = 6
COLS = 7


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
) -> List[SelfPlayExample]:
    """Play one self-play game and return (state, pi, z) examples."""
    board = np.zeros((ROWS, COLS), dtype=np.int8)
    current_player = 1
    examples: List[SelfPlayExample] = []

    in_channels = int(getattr(model, "in_channels", 3))
    include_legal_channel = in_channels >= 3
    evaluator = NeuralEvaluator(model, device=device)
    rng = np.random.default_rng()

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
) -> Path:
    """Generate self-play examples and save as NPZ."""
    all_examples: List[SelfPlayExample] = []
    game_lengths: List[int] = []
    winner_distribution = {"player1": 0, "player2": 0, "draw": 0}

    for game_idx in range(1, int(num_games) + 1):
        game_examples = play_self_play_game(
            model=model,
            num_simulations=int(num_simulations),
            device=device,
            c_puct=float(c_puct),
            add_dirichlet_noise=bool(add_dirichlet_noise),
            use_tactical_shortcuts=bool(use_tactical_shortcuts),
            max_moves=42,
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
        "created_at": _utc_now_iso(),
    }

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
    args = parser.parse_args()

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

