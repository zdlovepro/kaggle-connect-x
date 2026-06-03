"""PUCT MCTS with pluggable evaluator for AlphaZero-lite pipeline.

This module keeps compatibility with current MCTS-lite direction while
preparing a clean evaluator interface for future policy-value networks.
"""

from __future__ import annotations

import argparse
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from azlite.board import (
    apply_move,
    center_first_order,
    find_immediate_block,
    find_immediate_win,
    legal_moves,
    obs_board_to_numpy,
    terminal_value,
)


EPS = 1e-12


@dataclass
class MCTSNode:
    """PUCT MCTS node.

    value_sum is always stored from *this node's current_player perspective*.
    """

    board: np.ndarray
    current_player: int
    parent: Optional["MCTSNode"] = None
    prior: float = 0.0
    action_from_parent: Optional[int] = None
    children: Dict[int, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0
    is_expanded: bool = False

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return float(self.value_sum / self.visit_count)


class Evaluator(ABC):
    """Abstract evaluator: (board, current_player) -> (policy_probs, value)."""

    @abstractmethod
    def evaluate(self, board: np.ndarray, current_player: int) -> Tuple[np.ndarray, float]:
        raise NotImplementedError


def _ensure_board(board: np.ndarray | Sequence[int] | Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(board, dtype=np.int8)
    if arr.ndim == 1:
        # Kaggle observation.board flat list
        if arr.size != 42:
            raise ValueError(f"flat board must have length 42, got {arr.size}")
        return obs_board_to_numpy(arr.tolist())
    if arr.ndim == 2:
        return arr.astype(np.int8, copy=False)
    raise ValueError(f"board must be flat(42,) or 2D, got shape={arr.shape}")


def _normalize_policy_for_board(policy: np.ndarray, board: np.ndarray) -> np.ndarray:
    cols = board.shape[1]
    p = np.asarray(policy, dtype=np.float64).reshape(-1)
    if p.size != cols:
        raise ValueError(f"policy_probs must have shape=({cols},), got ({p.size},)")

    p = np.maximum(p, 0.0)
    mask = np.zeros(cols, dtype=np.float64)
    valid = legal_moves(board)
    if not valid:
        return mask

    mask[valid] = 1.0
    p *= mask
    s = float(p.sum())
    if s <= EPS:
        p = mask / float(mask.sum())
    else:
        p /= s
    return p.astype(np.float32, copy=False)


class UniformEvaluator(Evaluator):
    """Uniform policy over legal moves, zero value."""

    def evaluate(self, board: np.ndarray, current_player: int) -> Tuple[np.ndarray, float]:
        del current_player
        cols = board.shape[1]
        raw = np.ones(cols, dtype=np.float32)
        policy = _normalize_policy_for_board(raw, board)
        return policy, 0.0


def _count_open_windows(board: np.ndarray, player: int, inarow: int = 4) -> Tuple[int, int]:
    """Return (#3-in-row-with-1-empty, #2-in-row-with-2-empty) for `player`."""
    rows, cols = board.shape
    opp = 2 if player == 1 else 1
    threes = 0
    twos = 0

    def _scan_window(window: np.ndarray):
        nonlocal threes, twos
        p = int(np.count_nonzero(window == player))
        o = int(np.count_nonzero(window == opp))
        e = int(np.count_nonzero(window == 0))
        if o > 0:
            return
        if p == 3 and e == 1:
            threes += 1
        elif p == 2 and e == 2:
            twos += 1

    # Horizontal
    for r in range(rows):
        for c in range(cols - inarow + 1):
            _scan_window(board[r, c : c + inarow])
    # Vertical
    for r in range(rows - inarow + 1):
        for c in range(cols):
            _scan_window(board[r : r + inarow, c])
    # Main diagonal (\)
    for r in range(rows - inarow + 1):
        for c in range(cols - inarow + 1):
            w = np.array([board[r + i, c + i] for i in range(inarow)], dtype=np.int8)
            _scan_window(w)
    # Anti diagonal (/)
    for r in range(inarow - 1, rows):
        for c in range(cols - inarow + 1):
            w = np.array([board[r - i, c + i] for i in range(inarow)], dtype=np.int8)
            _scan_window(w)

    return threes, twos


class HeuristicEvaluator(Evaluator):
    """Heuristic evaluator compatible with future policy-value interface.

    Policy:
      - immediate win boost
      - immediate block boost
      - center preference
      - avoid moves that allow opponent immediate win

    Value:
      - terminal exact value
      - tactical immediate win/loss signal
      - open-window heuristic squashed to [-1, 1] by tanh
    """

    def evaluate(self, board: np.ndarray, current_player: int) -> Tuple[np.ndarray, float]:
        board = _ensure_board(board)
        cols = board.shape[1]
        opp = 2 if current_player == 1 else 1
        valid = legal_moves(board)
        raw_policy = np.zeros(cols, dtype=np.float32)

        if not valid:
            return raw_policy, 0.0

        immediate_win = find_immediate_win(board, current_player)
        immediate_block = find_immediate_block(board, current_player, opp)
        center = cols // 2

        for col in valid:
            score = 1.0 + float(cols - abs(col - center))
            if immediate_win is not None and col == immediate_win:
                score += 200.0
            if immediate_block is not None and col == immediate_block:
                score += 120.0

            nxt = apply_move(board, col, current_player)
            # Penalize moves that give opponent a direct tactical reply.
            if find_immediate_win(nxt, opp) is not None:
                score *= 0.1
            raw_policy[col] = score

        policy = _normalize_policy_for_board(raw_policy, board)

        tv = terminal_value(board, current_player)
        if tv is not None:
            return policy, float(tv)

        if immediate_win is not None:
            return policy, 0.95
        if find_immediate_win(board, opp) is not None:
            return policy, -0.95

        my3, my2 = _count_open_windows(board, current_player)
        op3, op2 = _count_open_windows(board, opp)
        center_my = int(np.count_nonzero(board[:, center] == current_player))
        center_op = int(np.count_nonzero(board[:, center] == opp))

        raw_value = (
            1.8 * (my3 - op3)
            + 0.7 * (my2 - op2)
            + 0.4 * (center_my - center_op)
        )
        value = float(np.tanh(raw_value / 6.0))
        return policy, max(-1.0, min(1.0, value))


def _select_child(node: MCTSNode, c_puct: float) -> Tuple[int, MCTSNode]:
    parent_n = max(1, node.visit_count)
    sqrt_parent = math.sqrt(parent_n)
    best_action = -1
    best_score = -float("inf")
    best_child: Optional[MCTSNode] = None

    for action in center_first_order(node.board.shape[1]):
        child = node.children.get(action)
        if child is None:
            continue
        # child.q_value is stored from child.current_player perspective.
        # Since child.current_player is the opponent of node.current_player,
        # negate it to score this move from the parent/node perspective.
        q = -child.q_value  # 0 for unvisited nodes
        u = c_puct * float(child.prior) * sqrt_parent / (1.0 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action
            best_child = child

    if best_child is None:
        # Should not happen for expanded non-terminal nodes, but keep safe fallback.
        action, child = next(iter(node.children.items()))
        return action, child
    return best_action, best_child


def _expand_and_evaluate(node: MCTSNode, evaluator: Evaluator) -> float:
    """Expand leaf node and return value from node.current_player perspective."""
    tv = terminal_value(node.board, node.current_player)
    if tv is not None:
        node.is_expanded = True
        return float(tv)

    policy, value = evaluator.evaluate(node.board, node.current_player)
    policy = _normalize_policy_for_board(policy, node.board)
    value = float(max(-1.0, min(1.0, value)))

    legal = legal_moves(node.board)
    opp = 2 if node.current_player == 1 else 1
    for action in legal:
        next_board = apply_move(node.board, action, node.current_player)
        node.children[action] = MCTSNode(
            board=next_board,
            current_player=opp,
            parent=node,
            prior=float(policy[action]),
            action_from_parent=action,
        )
    node.is_expanded = True
    return value


def _backpropagate(search_path: Sequence[MCTSNode], leaf_value: float) -> None:
    """Backpropagate leaf value along path.

    Why flip sign each level:
      `leaf_value` is always from the *leaf node current_player* perspective.
      Parent nodes are the opponent perspective of their children, so the same
      position value must change sign when moving one ply upward.
    """
    value = float(leaf_value)
    for node in reversed(search_path):
        node.visit_count += 1
        node.value_sum += value
        value = -value


def _apply_dirichlet_noise_to_root(
    root: MCTSNode,
    alpha: float,
    frac: float,
) -> None:
    legal = sorted(root.children.keys())
    if not legal:
        return
    noise = np.random.dirichlet([alpha] * len(legal))
    for i, action in enumerate(legal):
        child = root.children[action]
        child.prior = (1.0 - frac) * float(child.prior) + frac * float(noise[i])


def _build_policy_from_visits(
    visit_counts: np.ndarray,
    legal: Sequence[int],
    temperature: float,
) -> np.ndarray:
    policy = np.zeros_like(visit_counts, dtype=np.float32)
    if not legal:
        return policy

    if temperature <= 1e-8:
        max_visits = np.max(visit_counts)
        best = [a for a in legal if visit_counts[a] == max_visits]
        chosen = min(best, key=lambda a: (abs(a - (visit_counts.size // 2)), a))
        policy[chosen] = 1.0
        return policy

    scaled = np.zeros_like(visit_counts, dtype=np.float64)
    scaled[legal] = np.power(visit_counts[legal], 1.0 / temperature)
    s = float(scaled.sum())
    if s <= EPS:
        policy[legal] = 1.0 / len(legal)
        return policy
    policy = (scaled / s).astype(np.float32)
    return policy


def run_mcts(
    board,
    current_player,
    evaluator,
    num_simulations=100,
    c_puct=1.5,
    temperature=1.0,
    add_dirichlet_noise=False,
    dirichlet_alpha=0.3,
    dirichlet_frac=0.25,
    use_tactical_shortcuts=True,
    return_root=False,
):
    """Run PUCT MCTS and return move + policy target + visit stats."""
    board = _ensure_board(board)
    cols = board.shape[1]
    if cols != 7:
        raise ValueError(f"current implementation expects 7 columns, got {cols}")

    if not isinstance(evaluator, Evaluator):
        raise TypeError("evaluator must implement Evaluator interface")

    legal = legal_moves(board)
    visit_counts = np.zeros(cols, dtype=np.int32)

    # Tactical shortcuts for battle/submission path.
    if use_tactical_shortcuts:
        opp = 2 if current_player == 1 else 1
        win_col = find_immediate_win(board, current_player)
        if win_col is not None:
            visit_counts[win_col] = 1
            policy_target = np.zeros(cols, dtype=np.float32)
            policy_target[win_col] = 1.0
            out = {
                "move": int(win_col),
                "policy_target": policy_target,
                "visit_counts": visit_counts.astype(np.float32),
                "root_value": 1.0,
            }
            if return_root:
                out["root"] = None
            return out

        block_col = find_immediate_block(board, current_player, opp)
        if block_col is not None:
            visit_counts[block_col] = 1
            policy_target = np.zeros(cols, dtype=np.float32)
            policy_target[block_col] = 1.0
            out = {
                "move": int(block_col),
                "policy_target": policy_target,
                "visit_counts": visit_counts.astype(np.float32),
                "root_value": 0.0,
            }
            if return_root:
                out["root"] = None
            return out

    root = MCTSNode(board=board.copy(), current_player=int(current_player), prior=1.0)

    if add_dirichlet_noise:
        # Expand root once so priors exist, then perturb priors only at root.
        _ = _expand_and_evaluate(root, evaluator)
        _apply_dirichlet_noise_to_root(root, dirichlet_alpha, dirichlet_frac)

    sims = max(0, int(num_simulations))
    for _ in range(sims):
        node = root
        search_path = [node]

        while node.is_expanded and node.children:
            _, node = _select_child(node, c_puct=float(c_puct))
            search_path.append(node)

        leaf_value = _expand_and_evaluate(node, evaluator)
        _backpropagate(search_path, leaf_value)

    for action, child in root.children.items():
        visit_counts[action] = child.visit_count

    legal = legal_moves(board)
    policy_target = _build_policy_from_visits(
        visit_counts=visit_counts.astype(np.float32),
        legal=legal,
        temperature=float(temperature),
    )
    move = int(np.argmax(policy_target)) if legal else 0

    result = {
        "move": move,
        "policy_target": policy_target.astype(np.float32),
        "visit_counts": visit_counts.astype(np.float32),
        "root_value": float(root.q_value),
    }
    if return_root:
        result["root"] = root
    return result


def _smoke() -> None:
    board = np.zeros((6, 7), dtype=np.int8)
    evaluator = HeuristicEvaluator()
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=evaluator,
        num_simulations=64,
        c_puct=1.5,
        temperature=1.0,
        add_dirichlet_noise=False,
        use_tactical_shortcuts=True,
        return_root=False,
    )
    print("[SMOKE] move:", result["move"])
    print("[SMOKE] policy_target:", np.round(result["policy_target"], 4))
    print("[SMOKE] visit_counts:", result["visit_counts"].astype(int).tolist())
    print("[SMOKE] root_value:", round(float(result["root_value"]), 4))


def _main() -> None:
    parser = argparse.ArgumentParser(description="PUCT MCTS smoke runner")
    parser.add_argument("--smoke", action="store_true", help="Run quick smoke test")
    args = parser.parse_args()
    if args.smoke:
        _smoke()
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
