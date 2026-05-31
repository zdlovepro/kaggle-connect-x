"""AlphaZero-lite submission runtime (single-file friendly).

This module is used as a build template by `build_azlite_submission.py`.
The builder injects `AZLITE_EXPORT` directly so final submission has no
external file dependency.
"""

from __future__ import annotations

import base64
import math
import time
import zlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


AZLITE_EXPORT = None  # __AZLITE_EXPORT_PLACEHOLDER__


_EXPORT_CACHE: Optional[Dict[str, Any]] = None
_NP_TENSOR_CACHE: Optional[Dict[str, np.ndarray]] = None
_TORCH_STATE: Optional[Dict[str, Any]] = None


def _load_export() -> Dict[str, Any]:
    global _EXPORT_CACHE
    if _EXPORT_CACHE is not None:
        return _EXPORT_CACHE

    payload = AZLITE_EXPORT
    if payload is None:
        try:
            from agents.azlite_weights import AZLITE_EXPORT as payload  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "AZLITE export payload missing. Build final submission with "
                "`python build_azlite_submission.py --checkpoint ...`"
            ) from exc

    _EXPORT_CACHE = dict(payload)
    return _EXPORT_CACHE


def _decode_tensor(spec: Any) -> np.ndarray:
    if isinstance(spec, np.ndarray):
        return spec.astype(np.float32, copy=False)

    if isinstance(spec, Mapping) and "data" in spec and "shape" in spec:
        raw = base64.b64decode(str(spec["data"]).encode("ascii"))
        encoding = str(spec.get("encoding", "base64"))
        if "zlib" in encoding:
            raw = zlib.decompress(raw)
        arr = np.frombuffer(raw, dtype=np.dtype(spec.get("dtype", "float32")))
        arr = arr.reshape(tuple(int(v) for v in spec["shape"]))
        return arr.astype(np.float32, copy=False)

    return np.asarray(spec, dtype=np.float32)


def _get_np_tensors() -> Dict[str, np.ndarray]:
    global _NP_TENSOR_CACHE
    if _NP_TENSOR_CACHE is not None:
        return _NP_TENSOR_CACHE

    export = _load_export()
    tensors = export.get("tensors")
    if not isinstance(tensors, Mapping):
        raise RuntimeError("Invalid AZLITE export payload: missing tensors")

    decoded: Dict[str, np.ndarray] = {}
    for name, spec in tensors.items():
        decoded[str(name)] = _decode_tensor(spec)
    _NP_TENSOR_CACHE = decoded
    return decoded


def _get_runtime_cfg() -> Dict[str, Any]:
    export = _load_export()
    runtime = export.get("runtime", {})
    if not isinstance(runtime, Mapping):
        runtime = {}
    return dict(runtime)


def _dims() -> Tuple[int, int, int, int]:
    export = _load_export()
    rows = int(export.get("rows", 6))
    cols = int(export.get("columns", 7))
    inarow = int(export.get("inarow", 4))
    channels = int(export.get("channels", 3))
    return rows, cols, inarow, channels


def _ordered_cols(columns: int) -> List[int]:
    center = columns // 2
    return sorted(range(columns), key=lambda c: (abs(c - center), c))


def _obs_to_board(obs_board: Sequence[int], rows: int, columns: int) -> np.ndarray:
    arr = np.asarray(obs_board, dtype=np.int8)
    if arr.size != rows * columns:
        raise ValueError(f"obs.board length mismatch: {arr.size} vs {rows * columns}")
    return arr.reshape(rows, columns)


def _legal_moves(board: np.ndarray) -> List[int]:
    return [c for c in range(board.shape[1]) if board[0, c] == 0]


def _next_open_row(board: np.ndarray, col: int) -> Optional[int]:
    for r in range(board.shape[0] - 1, -1, -1):
        if board[r, col] == 0:
            return r
    return None


def _apply_move(board: np.ndarray, col: int, player: int) -> np.ndarray:
    r = _next_open_row(board, col)
    if r is None:
        raise ValueError(f"illegal move col={col}")
    out = board.copy()
    out[r, col] = np.int8(player)
    return out


def _check_win(board: np.ndarray, player: int, inarow: int) -> bool:
    rows, cols = board.shape

    for r in range(rows):
        for c in range(cols - inarow + 1):
            if np.all(board[r, c : c + inarow] == player):
                return True
    for r in range(rows - inarow + 1):
        for c in range(cols):
            if np.all(board[r : r + inarow, c] == player):
                return True
    for r in range(rows - inarow + 1):
        for c in range(cols - inarow + 1):
            if all(board[r + i, c + i] == player for i in range(inarow)):
                return True
    for r in range(inarow - 1, rows):
        for c in range(cols - inarow + 1):
            if all(board[r - i, c + i] == player for i in range(inarow)):
                return True
    return False


def _winner(board: np.ndarray, inarow: int) -> Optional[int]:
    if _check_win(board, 1, inarow):
        return 1
    if _check_win(board, 2, inarow):
        return 2
    return None


def _is_draw(board: np.ndarray) -> bool:
    return bool(np.all(board != 0))


def _terminal_value(board: np.ndarray, current_player: int, inarow: int) -> Optional[float]:
    w = _winner(board, inarow)
    if w is not None:
        return 1.0 if w == current_player else -1.0
    if _is_draw(board):
        return 0.0
    return None


def _immediate_win(board: np.ndarray, player: int, inarow: int) -> Optional[int]:
    for col in _ordered_cols(board.shape[1]):
        if board[0, col] != 0:
            continue
        nxt = _apply_move(board, col, player)
        if _check_win(nxt, player, inarow):
            return col
    return None


def _immediate_block(board: np.ndarray, player: int, inarow: int) -> Optional[int]:
    opp = 2 if player == 1 else 1
    del player
    for col in _ordered_cols(board.shape[1]):
        if board[0, col] != 0:
            continue
        nxt = _apply_move(board, col, opp)
        if _check_win(nxt, opp, inarow):
            return col
    return None


def _board_to_input(board: np.ndarray, current_player: int, channels: int) -> np.ndarray:
    opp = 2 if current_player == 1 else 1
    x = np.zeros((channels, board.shape[0], board.shape[1]), dtype=np.float32)
    x[0] = (board == current_player).astype(np.float32, copy=False)
    x[1] = (board == opp).astype(np.float32, copy=False)

    if channels >= 3:
        legal_plane = np.zeros_like(board, dtype=np.float32)
        for c in _legal_moves(board):
            r = _next_open_row(board, c)
            if r is not None:
                legal_plane[r, c] = 1.0
        x[2] = legal_plane
    return x


def _conv2d_same(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    k_h = int(w.shape[2])
    k_w = int(w.shape[3])
    pad_h = k_h // 2
    pad_w = k_w // 2
    x_pad = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    # patches: (C, H, W, kH, kW) -> (H, W, C, kH, kW)
    patches = np.lib.stride_tricks.sliding_window_view(
        x_pad, (k_h, k_w), axis=(1, 2)
    ).transpose(1, 2, 0, 3, 4)
    # out: (H, W, O)
    out = np.tensordot(patches, w, axes=([2, 3, 4], [1, 2, 3]))
    out = out.transpose(2, 0, 1)
    out += b.reshape(-1, 1, 1)
    return out.astype(np.float32, copy=False)


def _forward_numpy(x: np.ndarray) -> Tuple[np.ndarray, float]:
    t = _get_np_tensors()
    h = _conv2d_same(x, t["features.0.weight"], t["features.0.bias"])
    h = np.maximum(h, 0.0).astype(np.float32, copy=False)
    h = _conv2d_same(h, t["features.2.weight"], t["features.2.bias"])
    h = np.maximum(h, 0.0).astype(np.float32, copy=False)
    h = _conv2d_same(h, t["features.4.weight"], t["features.4.bias"])
    h = np.maximum(h, 0.0).astype(np.float32, copy=False)

    flat = h.reshape(-1).astype(np.float32, copy=False)
    fc = t["backbone.1.weight"] @ flat + t["backbone.1.bias"]
    fc = np.maximum(fc, 0.0).astype(np.float32, copy=False)

    policy_logits = t["policy_head.weight"] @ fc + t["policy_head.bias"]
    value = t["value_head.0.weight"] @ fc + t["value_head.0.bias"]
    value = float(np.tanh(value.reshape(-1)[0]))
    return policy_logits.astype(np.float32, copy=False), value


def _maybe_init_torch() -> Optional[Dict[str, Any]]:
    global _TORCH_STATE
    if _TORCH_STATE is not None:
        return _TORCH_STATE

    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        _TORCH_STATE = None
        return None

    np_tensors = _get_np_tensors()
    torch_tensors = {k: torch.from_numpy(v).to(dtype=torch.float32) for k, v in np_tensors.items()}
    _TORCH_STATE = {"torch": torch, "F": F, "tensors": torch_tensors}
    return _TORCH_STATE


def _forward_torch(x: np.ndarray) -> Tuple[np.ndarray, float]:
    state = _maybe_init_torch()
    if state is None:
        return _forward_numpy(x)

    torch = state["torch"]
    F = state["F"]
    t = state["tensors"]

    xt = torch.from_numpy(x).unsqueeze(0).to(dtype=torch.float32)
    h = F.relu(F.conv2d(xt, t["features.0.weight"], t["features.0.bias"], padding=1))
    h = F.relu(F.conv2d(h, t["features.2.weight"], t["features.2.bias"], padding=1))
    h = F.relu(F.conv2d(h, t["features.4.weight"], t["features.4.bias"], padding=1))
    h = h.flatten(start_dim=1)
    fc = F.relu(F.linear(h, t["backbone.1.weight"], t["backbone.1.bias"]))
    policy_logits = F.linear(fc, t["policy_head.weight"], t["policy_head.bias"])
    value = torch.tanh(F.linear(fc, t["value_head.0.weight"], t["value_head.0.bias"]))

    p = policy_logits.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    v = float(value.squeeze(0).item())
    return p, max(-1.0, min(1.0, v))


def _masked_softmax(logits: np.ndarray, legal: Sequence[int]) -> np.ndarray:
    probs = np.zeros(logits.shape[0], dtype=np.float32)
    if not legal:
        return probs

    legal_logits = np.asarray([logits[c] for c in legal], dtype=np.float64)
    legal_logits -= np.max(legal_logits)
    exp = np.exp(legal_logits)
    denom = float(exp.sum())
    if denom <= 1e-12:
        probs[list(legal)] = 1.0 / len(legal)
        return probs
    soft = exp / denom
    for i, col in enumerate(legal):
        probs[col] = np.float32(soft[i])
    return probs


def predict_policy_value_from_board(board: np.ndarray | Sequence[int], current_player: int) -> Tuple[np.ndarray, float]:
    rows, cols, _inarow, channels = _dims()
    arr = np.asarray(board, dtype=np.int8)
    if arr.ndim == 1:
        arr = arr.reshape(rows, cols)

    legal = _legal_moves(arr)
    if not legal:
        return np.zeros(cols, dtype=np.float32), 0.0

    x = _board_to_input(arr, int(current_player), channels)
    runtime = _get_runtime_cfg()
    mode = str(runtime.get("inference_mode", _load_export().get("runtime_mode", "numpy"))).lower()

    if mode == "torch":
        logits, value = _forward_torch(x)
    else:
        logits, value = _forward_numpy(x)

    policy = _masked_softmax(logits, legal)
    return policy, max(-1.0, min(1.0, float(value)))


class _Node:
    __slots__ = (
        "board",
        "player",
        "prior",
        "parent",
        "action_from_parent",
        "children",
        "visit_count",
        "value_sum",
        "expanded",
    )

    def __init__(
        self,
        board: np.ndarray,
        player: int,
        prior: float = 0.0,
        parent: Optional["_Node"] = None,
        action_from_parent: Optional[int] = None,
    ) -> None:
        self.board = board
        self.player = int(player)
        self.prior = float(prior)
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: Dict[int, _Node] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.expanded = False

    @property
    def q(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum / self.visit_count)


def _select_child(node: _Node, c_puct: float, columns: int) -> Tuple[int, _Node]:
    best_action = -1
    best_score = -1e18
    best_child: Optional[_Node] = None
    sqrt_n = math.sqrt(max(1, node.visit_count))

    for col in _ordered_cols(columns):
        child = node.children.get(col)
        if child is None:
            continue
        score = child.q + c_puct * child.prior * sqrt_n / (1.0 + child.visit_count)
        if score > best_score:
            best_score = score
            best_action = col
            best_child = child

    if best_child is None:
        action, child = next(iter(node.children.items()))
        return int(action), child
    return best_action, best_child


def _expand(node: _Node, inarow: int, columns: int) -> float:
    tv = _terminal_value(node.board, node.player, inarow)
    if tv is not None:
        node.expanded = True
        return float(tv)

    legal = _legal_moves(node.board)
    if not legal:
        node.expanded = True
        return 0.0

    policy, value = predict_policy_value_from_board(node.board, node.player)
    opp = 2 if node.player == 1 else 1
    for col in legal:
        nxt = _apply_move(node.board, col, node.player)
        node.children[col] = _Node(
            board=nxt,
            player=opp,
            prior=float(policy[col]),
            parent=node,
            action_from_parent=col,
        )
    node.expanded = True
    return float(max(-1.0, min(1.0, value)))


def _backprop(path: Sequence[_Node], leaf_value: float) -> None:
    val = float(leaf_value)
    for node in reversed(path):
        node.visit_count += 1
        node.value_sum += val
        val = -val


def _run_mcts(
    board: np.ndarray,
    current_player: int,
    inarow: int,
    max_simulations: int,
    c_puct: float,
    deadline_s: float,
) -> Tuple[int, np.ndarray, float]:
    cols = board.shape[1]
    root = _Node(board=board.copy(), player=int(current_player), prior=1.0)
    sims = 0

    while sims < max(1, int(max_simulations)):
        if time.time() >= deadline_s:
            break
        node = root
        path = [node]

        while node.expanded and node.children:
            if time.time() >= deadline_s:
                break
            _, node = _select_child(node, float(c_puct), cols)
            path.append(node)
        if time.time() >= deadline_s:
            break

        leaf_val = _expand(node, inarow, cols)
        _backprop(path, leaf_val)
        sims += 1

    visits = np.zeros(cols, dtype=np.float32)
    for c, child in root.children.items():
        visits[c] = float(child.visit_count)

    legal = _legal_moves(board)
    if not legal:
        return 0, visits, 0.0

    best = max(legal, key=lambda c: (visits[c], -abs(c - cols // 2), -c))
    root_value = root.q if root.visit_count > 0 else 0.0
    return int(best), visits, float(root_value)


def _fallback_move(board: np.ndarray) -> int:
    legal = _legal_moves(board)
    if not legal:
        return 0
    ordered = _ordered_cols(board.shape[1])
    for c in ordered:
        if c in legal:
            return int(c)
    return int(legal[0])


def _simulations_by_phase(piece_count: int, runtime: Mapping[str, Any]) -> int:
    s_open = int(runtime.get("simulations_opening", 28))
    s_mid = int(runtime.get("simulations_midgame", 46))
    s_end = int(runtime.get("simulations_endgame", 72))
    s_min = int(runtime.get("simulations_min", 12))
    s_max = int(runtime.get("simulations_max", 100))

    if piece_count < 10:
        sims = s_open
    elif piece_count < 26:
        sims = s_mid
    else:
        sims = s_end
    sims = max(s_min, min(sims, s_max))
    return int(sims)


def agent(observation: Any, configuration: Any) -> int:
    rows, cols, inarow, _channels = _dims()
    rows = int(getattr(configuration, "rows", rows))
    cols = int(getattr(configuration, "columns", cols))
    inarow = int(getattr(configuration, "inarow", inarow))

    board = _obs_to_board(observation.board, rows, cols)
    current_player = int(observation.mark)

    legal = _legal_moves(board)
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])

    win_col = _immediate_win(board, current_player, inarow)
    if win_col is not None:
        return int(win_col)

    block_col = _immediate_block(board, current_player, inarow)
    if block_col is not None:
        return int(block_col)

    runtime = _get_runtime_cfg()
    c_puct = float(runtime.get("c_puct", 1.5))
    base_budget = float(runtime.get("time_budget_sec", 1.80))
    max_budget = float(runtime.get("time_budget_max_sec", 1.94))
    piece_count = int(np.count_nonzero(board))
    max_sims = _simulations_by_phase(piece_count, runtime)

    overage = float(getattr(observation, "remainingOverageTime", 0.0) or 0.0)
    bonus = min(0.12, max(0.0, overage) * 0.005)
    budget_sec = min(max_budget, base_budget + bonus)

    start = time.time()
    deadline = start + budget_sec

    try:
        move, visits, _root_value = _run_mcts(
            board=board,
            current_player=current_player,
            inarow=inarow,
            max_simulations=max_sims,
            c_puct=c_puct,
            deadline_s=deadline,
        )
        if move in legal:
            return int(move)

        if visits.size == cols:
            best = max(legal, key=lambda c: (visits[c], -abs(c - cols // 2), -c))
            return int(best)
    except Exception:
        pass

    return _fallback_move(board)
