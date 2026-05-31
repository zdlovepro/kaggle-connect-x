"""Supervised pretraining for AlphaZero-lite policy-value network.

Usage:
  python -m azlite.pretrain --data data/teacher/teacher_depth5.npz
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from azlite.board import find_immediate_block, find_immediate_win, legal_moves, obs_board_to_numpy
from azlite.model import ConnectXNet, NeuralEvaluator, predict_policy_value, save_checkpoint
from azlite.puct_mcts import run_mcts


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    data = np.load(str(npz_path), allow_pickle=True)
    states = np.asarray(data["states"], dtype=np.float32)
    policies = np.asarray(data["policies"], dtype=np.float32)
    values = np.asarray(data["values"], dtype=np.float32)

    if states.ndim != 4 or states.shape[2:] != (6, 7):
        raise ValueError(f"states shape must be (N,C,6,7), got {states.shape}")
    if policies.shape != (states.shape[0], 7):
        raise ValueError(
            f"policies shape must be (N,7) and match states N, got {policies.shape}"
        )
    if values.shape != (states.shape[0],):
        raise ValueError(f"values shape must be (N,), got {values.shape}")

    meta: Dict[str, object] = {}
    if "metadata" in data:
        raw = data["metadata"]
        try:
            if isinstance(raw, np.ndarray):
                if raw.shape == ():
                    raw = raw.item()
                elif raw.size == 1:
                    raw = raw.reshape(-1)[0]
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(raw, str):
                meta = json.loads(raw)
        except Exception:
            meta = {"raw_metadata": str(raw)}

    return states, policies, values, meta


def _soft_target_ce_loss(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    """Cross-entropy for soft policy targets."""
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target_probs * log_probs).sum(dim=1).mean()
    return loss


def _policy_only_agent(model: ConnectXNet, device: str = "cpu"):
    def _agent(observation, configuration):
        board = obs_board_to_numpy(
            observation.board,
            rows=int(configuration.rows),
            columns=int(configuration.columns),
        )
        valid = legal_moves(board)
        if not valid:
            return 0

        win_col = find_immediate_win(board, observation.mark)
        if win_col is not None:
            return int(win_col)
        opp = 2 if observation.mark == 1 else 1
        block_col = find_immediate_block(board, observation.mark, opp)
        if block_col is not None:
            return int(block_col)

        policy, _ = predict_policy_value(model, board, observation.mark, device=device)
        best = max(valid, key=lambda c: float(policy[c]))
        return int(best)

    return _agent


def _mcts_model_agent(
    model: ConnectXNet,
    device: str = "cpu",
    simulations: int = 96,
):
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
        result = run_mcts(
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
        move = int(result["move"])
        if move in valid:
            return move
        return int(valid[0])

    return _agent


@dataclass
class EvalResult:
    wins: int
    losses: int
    draws: int
    win_rate: float
    first_player_win_rate: float
    second_player_win_rate: float


def _evaluate_vs(
    agent_a,
    agent_b,
    games: int = 20,
) -> EvalResult:
    try:
        from kaggle_environments import make
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "kaggle_environments is required for evaluation. "
            "Install with `python -m pip install kaggle-environments`."
        ) from exc

    wins = losses = draws = 0
    fp_games = sp_games = 0
    fp_wins = sp_wins = 0

    half = games // 2
    if games % 2 != 0:
        half += 1

    for i in range(games):
        if i < half:
            env = make("connectx", debug=False)
            env.run([agent_a, agent_b])
            r0 = env.steps[-1][0].reward or 0
            r1 = env.steps[-1][1].reward or 0
            fp_games += 1
            if r0 == 1:
                wins += 1
                fp_wins += 1
            elif r1 == 1:
                losses += 1
            else:
                draws += 1
        else:
            env = make("connectx", debug=False)
            env.run([agent_b, agent_a])
            r0 = env.steps[-1][0].reward or 0
            r1 = env.steps[-1][1].reward or 0
            sp_games += 1
            if r1 == 1:
                wins += 1
                sp_wins += 1
            elif r0 == 1:
                losses += 1
            else:
                draws += 1

    wr = wins / max(1, games)
    fp_wr = fp_wins / max(1, fp_games)
    sp_wr = sp_wins / max(1, sp_games)
    return EvalResult(
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wr,
        first_player_win_rate=fp_wr,
        second_player_win_rate=sp_wr,
    )


def _print_eval(title: str, r: EvalResult) -> None:
    print(
        f"{title}: "
        f"{r.wins}W/{r.losses}L/{r.draws}D  "
        f"WR={r.win_rate*100:.1f}%  "
        f"FP={r.first_player_win_rate*100:.1f}%  "
        f"SP={r.second_player_win_rate*100:.1f}%"
    )


def train_supervised(
    model: ConnectXNet,
    states: np.ndarray,
    policies: np.ndarray,
    values: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    value_loss_weight: float,
) -> Tuple[ConnectXNet, torch.optim.Optimizer, Dict[str, float]]:
    x = torch.from_numpy(states).to(dtype=torch.float32)
    pi = torch.from_numpy(policies).to(dtype=torch.float32)
    v = torch.from_numpy(values).to(dtype=torch.float32)
    dataset = TensorDataset(x, pi, v)
    loader = DataLoader(dataset, batch_size=max(1, int(batch_size)), shuffle=True)

    model = model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))

    last_metrics: Dict[str, float] = {}
    for epoch in range(1, int(epochs) + 1):
        loss_sum = pi_loss_sum = v_loss_sum = 0.0
        batch_count = 0

        for xb, pib, vb in loader:
            xb = xb.to(device=device, dtype=torch.float32, non_blocking=True)
            pib = pib.to(device=device, dtype=torch.float32, non_blocking=True)
            vb = vb.to(device=device, dtype=torch.float32, non_blocking=True)

            logits, value_pred = model(xb)
            value_pred = value_pred.squeeze(1)

            pi_loss = _soft_target_ce_loss(logits, pib)
            v_loss = F.mse_loss(value_pred, vb)
            loss = pi_loss + float(value_loss_weight) * v_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item())
            pi_loss_sum += float(pi_loss.item())
            v_loss_sum += float(v_loss.item())
            batch_count += 1

        metrics = {
            "loss": loss_sum / max(1, batch_count),
            "policy_loss": pi_loss_sum / max(1, batch_count),
            "value_loss": v_loss_sum / max(1, batch_count),
        }
        last_metrics = metrics
        print(
            f"[pretrain] epoch {epoch}/{epochs}  "
            f"loss={metrics['loss']:.4f}  "
            f"pi={metrics['policy_loss']:.4f}  "
            f"v={metrics['value_loss']:.4f}"
        )

    return model, optimizer, last_metrics


def _main() -> None:
    parser = argparse.ArgumentParser(description="Supervised pretraining for ConnectX policy-value net")
    parser.add_argument("--data", type=str, required=True, help="Path to teacher dataset npz")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default="checkpoints/azlite_pretrained.pt")
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-simulations", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    _set_seed(int(args.seed))
    data_path = Path(args.data)
    states, policies, values, data_meta = _load_dataset(data_path)
    print(
        f"[pretrain] loaded data: states={states.shape}, "
        f"policies={policies.shape}, values={values.shape}"
    )
    if data_meta:
        short_meta = {
            "teacher_type": data_meta.get("teacher_type"),
            "teacher_depth": data_meta.get("teacher_depth"),
            "positions_saved": data_meta.get("positions_saved"),
            "created_at": data_meta.get("created_at"),
        }
        print("[pretrain] dataset metadata:", json.dumps(short_meta, ensure_ascii=False))

    in_channels = int(states.shape[1])
    device = str(args.device)

    model = ConnectXNet(in_channels=in_channels)
    init_model = copy.deepcopy(model)

    eval_results: Dict[str, Dict[str, float]] = {}
    if not args.skip_eval and int(args.eval_games) > 0:
        print("[pretrain] pre-train evaluation...")
        pre_policy_agent = _policy_only_agent(init_model, device=device)
        pre_mcts_agent = _mcts_model_agent(
            init_model,
            device=device,
            simulations=int(args.eval_simulations),
        )
        r = _evaluate_vs(pre_policy_agent, "random", games=int(args.eval_games))
        _print_eval("pre policy-only vs random", r)
        eval_results["pre_policy_vs_random"] = r.__dict__

        r = _evaluate_vs(pre_mcts_agent, "random", games=int(args.eval_games))
        _print_eval("pre mcts+random-net vs random", r)
        eval_results["pre_mcts_randomnet_vs_random"] = r.__dict__

    model, optimizer, train_metrics = train_supervised(
        model=model,
        states=states,
        policies=policies,
        values=values,
        device=device,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        value_loss_weight=float(args.value_loss_weight),
    )

    if not args.skip_eval and int(args.eval_games) > 0:
        print("[pretrain] post-train evaluation...")
        post_policy_agent = _policy_only_agent(model, device=device)
        post_mcts_agent = _mcts_model_agent(
            model,
            device=device,
            simulations=int(args.eval_simulations),
        )

        r = _evaluate_vs(post_policy_agent, "random", games=int(args.eval_games))
        _print_eval("post policy-only vs random", r)
        eval_results["post_policy_vs_random"] = r.__dict__

        r = _evaluate_vs(post_mcts_agent, "random", games=int(args.eval_games))
        _print_eval("post mcts+pretrained-net vs random", r)
        eval_results["post_mcts_pretrained_vs_random"] = r.__dict__

        r = _evaluate_vs(post_mcts_agent, "negamax", games=int(args.eval_games))
        _print_eval("post mcts+pretrained-net vs negamax", r)
        eval_results["post_mcts_pretrained_vs_negamax"] = r.__dict__

    output = Path(args.output)
    metadata = {
        "iteration": 0,
        "train_steps": int(args.epochs) * int(np.ceil(states.shape[0] / max(1, int(args.batch_size)))),
        "simulations": int(args.eval_simulations),
        "eval_results": eval_results,
        "data_path": str(data_path.resolve()),
        "data_metadata": data_meta,
        "train_metrics": train_metrics,
    }
    save_checkpoint(model, optimizer, output, metadata=metadata)
    print("[pretrain] checkpoint saved:", output.resolve())


if __name__ == "__main__":
    _main()
