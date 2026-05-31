"""Lightweight neural model and evaluator utilities for AlphaZero-lite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from azlite.board import legal_moves, obs_board_to_numpy, to_tensor
from azlite.puct_mcts import Evaluator


class ConnectXNet(nn.Module):
    """Small policy-value network for 6x7 ConnectX."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.in_channels = int(in_channels)

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 7, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, 7)  # logits
        self.value_head = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        h = self.backbone(h)
        policy_logits = self.policy_head(h)
        value = self.value_head(h)
        return policy_logits, value


def _ensure_board_array(board) -> np.ndarray:
    arr = np.asarray(board, dtype=np.int8)
    if arr.ndim == 1:
        if arr.size != 42:
            raise ValueError(f"flat board must have length 42, got {arr.size}")
        return obs_board_to_numpy(arr.tolist())
    if arr.ndim == 2:
        return arr
    raise ValueError(f"board must be flat(42,) or 2D, got shape={arr.shape}")


def _softmax_masked_logits(logits: np.ndarray, valid_cols: list[int]) -> np.ndarray:
    probs = np.zeros_like(logits, dtype=np.float32)
    if not valid_cols:
        return probs

    legal_logits = np.array([logits[c] for c in valid_cols], dtype=np.float64)
    legal_logits -= np.max(legal_logits)
    exp = np.exp(legal_logits)
    denom = float(exp.sum())
    if denom <= 1e-12:
        probs[valid_cols] = 1.0 / len(valid_cols)
        return probs

    soft = exp / denom
    for i, c in enumerate(valid_cols):
        probs[c] = np.float32(soft[i])
    return probs


def predict_policy_value(
    model: ConnectXNet,
    board,
    current_player: int,
    device: str = "cpu",
) -> Tuple[np.ndarray, float]:
    """Predict masked policy probs and value from current_player perspective."""
    board_arr = _ensure_board_array(board)
    include_legal_channel = bool(getattr(model, "in_channels", 3) >= 3)

    x_np = to_tensor(
        board=board_arr,
        current_player=current_player,
        include_legal_channel=include_legal_channel,
        dtype=np.float32,
    )
    x = torch.from_numpy(x_np).unsqueeze(0).to(device=device, dtype=torch.float32)

    was_training = model.training
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        policy_logits, value = model(x)
    if was_training:
        model.train()

    logits_np = policy_logits.squeeze(0).detach().cpu().numpy().astype(np.float64)
    valid_cols = legal_moves(board_arr)
    policy_probs = _softmax_masked_logits(logits_np, valid_cols)

    value_f = float(value.squeeze(0).item())
    value_f = max(-1.0, min(1.0, value_f))
    return policy_probs, value_f


class NeuralEvaluator(Evaluator):
    """Evaluator adapter so neural model can plug into run_mcts directly."""

    def __init__(self, model: ConnectXNet, device: str = "cpu"):
        self.model = model
        self.device = device
        self.model.to(device)

    def evaluate(self, board: np.ndarray, current_player: int) -> Tuple[np.ndarray, float]:
        return predict_policy_value(
            model=self.model,
            board=board,
            current_player=current_player,
            device=self.device,
        )


def _default_metadata(metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    m = dict(metadata or {})
    m.setdefault("iteration", 0)
    m.setdefault("train_steps", 0)
    m.setdefault("simulations", 0)
    m.setdefault("eval_results", {})
    m.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    return m


def save_checkpoint(
    model: ConnectXNet,
    optimizer,
    path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save model/optimizer states and training metadata."""
    meta = _default_metadata(metadata)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metadata": meta,
        "model_config": {
            "in_channels": int(getattr(model, "in_channels", 3)),
        },
    }

    path = Path(path)
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, str(path))


def load_checkpoint(path, device: str = "cpu") -> Tuple[ConnectXNet, Dict[str, Any]]:
    """Load checkpoint and return reconstructed model + metadata."""
    ckpt = torch.load(str(path), map_location=device)
    model_cfg = ckpt.get("model_config", {})
    in_channels = int(model_cfg.get("in_channels", 3))

    model = ConnectXNet(in_channels=in_channels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    metadata = _default_metadata(ckpt.get("metadata"))
    return model, metadata


__all__ = [
    "ConnectXNet",
    "NeuralEvaluator",
    "load_checkpoint",
    "predict_policy_value",
    "save_checkpoint",
]

