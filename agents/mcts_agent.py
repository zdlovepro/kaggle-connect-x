"""
MCTS (Monte Carlo Tree Search) Agent — Bitboard 版

基于 UCB1 选择策略的蒙特卡洛树搜索，使用 bitboard 表示棋盘。

特性:
  - UCB1 树选择策略
  - Bitboard 位运算 → 赢棋检测 O(1)，节点复制仅需两个 uint64
  - 启发式 Rollout (80% 启发式, 20% 随机)
  - 时间感知的模拟控制
"""

import math
import random
import time
from typing import Any, List, Optional

import numpy as np

from azlite.board import (
    center_first_order,
    find_immediate_block,
    find_immediate_win,
    legal_moves as az_legal_moves,
    obs_board_to_numpy,
    ordered_legal_moves,
)
from connectx.bitboard import (
    COLS, ROWS, INAROW, ROW_STRIDE,
    has_won, list_to_bitboards, get_heights_from_list,
    valid_cols, drop_pos,
)
from .base import BaseAgent


# ── MCTS Node ────────────────────────────────────────────────

class MCTSNode:
    """MCTS 树节点 — bitboard 状态"""

    __slots__ = ('b1', 'b2', 'heights', 'mark', 'parent', 'action',
                 'children', 'visits', 'value', 'unvisited')

    def __init__(self, b1, b2, heights, mark, parent=None, action=None,
                 unvisited=None):
        self.b1 = b1
        self.b2 = b2
        self.heights = heights
        self.mark = mark          # 1 or 2
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.unvisited = unvisited or []

    def is_fully_expanded(self):
        return len(self.unvisited) == 0


# ── MCTS Agent ───────────────────────────────────────────────

class MCTSAgent(BaseAgent):
    """MCTS Agent — UCB1 + Bitboard"""

    def __init__(self, name="mcts", c_param=1.414, time_budget_ms=1900.0,
                 heuristic_rollout_prob=0.8):
        super().__init__(name=name)
        self.c_param = c_param
        self.time_budget_ms = time_budget_ms
        self.heuristic_rollout_prob = heuristic_rollout_prob
        self.simulations = 0

    def select_action(self, observation, configuration):
        board_list = list(observation.board)
        mark = observation.mark
        board_np = obs_board_to_numpy(board_list)
        b1, b2 = list_to_bitboards(board_list)
        heights = get_heights_from_list(board_list)

        valid = az_legal_moves(board_np)
        if not valid:
            self.stats["moves_made"] += 1
            return 0

        opp = 2 if mark == 1 else 1
        win_col = find_immediate_win(board_np, mark)
        if win_col is not None:
            self.stats["moves_made"] += 1
            return win_col

        block_col = find_immediate_block(board_np, mark, opp)
        if block_col is not None:
            self.stats["moves_made"] += 1
            return block_col

        root = MCTSNode(
            b1=b1, b2=b2, heights=heights.copy(), mark=mark,
            unvisited=ordered_legal_moves(board_np),
        )
        return self._search(root)

    def _search(self, root):
        self.simulations = 0
        start = time.perf_counter()

        while True:
            if (time.perf_counter() - start) * 1000 > self.time_budget_ms:
                break

            node = self._select(root)

            if not node.is_fully_expanded():
                node = self._expand(node)

            result = self._simulate(node)
            self._backpropagate(node, result)

            self.simulations += 1

        if not root.children:
            fallback = self._order_moves(valid_cols(root.heights))
            return fallback[0] if fallback else 0

        best_action = max(
            root.children.items(),
            key=lambda kv: (kv[1].visits,
                            kv[1].value / max(1, kv[1].visits))
        )[0]

        self.stats["nodes_visited"] = self.simulations
        self.stats["moves_made"] += 1
        return best_action

    def _select(self, node):
        while node.children and node.unvisited == []:
            node = self._best_child(node)
        return node

    def _best_child(self, node):
        log_parent = math.log(max(1, node.visits))
        best = None
        best_ucb = -float('inf')
        for child in node.children.values():
            ucb = (float('inf') if child.visits == 0
                   else child.value / child.visits
                        + self.c_param * math.sqrt(log_parent / child.visits))
            if ucb > best_ucb:
                best_ucb = ucb
                best = child
        return best if best else next(iter(node.children.values()))

    def _expand(self, node):
        if not node.unvisited:
            return node

        col = node.unvisited.pop(0)
        pos = drop_pos(node.heights, col)
        mask = np.uint64(1) << np.uint64(pos)

        new_heights = node.heights.copy()
        new_heights[col] += 1

        if node.mark == 1:
            new_b1 = node.b1 | mask
            new_b2 = node.b2
        else:
            new_b1 = node.b1
            new_b2 = node.b2 | mask

        new_mark = 2 if node.mark == 1 else 1
        new_valid = valid_cols(new_heights)

        child = MCTSNode(
            b1=new_b1, b2=new_b2, heights=new_heights,
            mark=new_mark, parent=node, action=col,
            unvisited=self._order_moves(new_valid),
        )
        node.children[col] = child
        return child

    def _simulate(self, node):
        b1 = np.uint64(int(node.b1))
        b2 = np.uint64(int(node.b2))
        heights = node.heights.copy()
        mark = node.mark
        original_mark = mark
        steps = 0

        while True:
            valid = valid_cols(heights)
            if not valid:
                return 0.0

            b_self = b1 if mark == 1 else b2
            b_opp = b2 if mark == 1 else b1

            for col in valid:
                pos = drop_pos(heights, col)
                if has_won(b_self | (np.uint64(1) << np.uint64(pos))):
                    return 1.0 if mark == original_mark else 0.0

            if random.random() < self.heuristic_rollout_prob:
                col = self._heuristic_move(b_self, b_opp, heights, valid)
            else:
                col = random.choice(valid)

            pos = drop_pos(heights, col)
            mask = np.uint64(1) << np.uint64(pos)
            if mark == 1:
                b1 |= mask
            else:
                b2 |= mask
            heights[col] += 1
            mark = 2 if mark == 1 else 1

            steps += 1
            if steps > COLS * ROWS:
                return 0.0

    def _backpropagate(self, node, value):
        while node is not None:
            node.visits += 1
            node.value += value
            value = 1.0 - value
            node = node.parent

    def _heuristic_move(self, b_self, b_opp, heights, valid):
        for col in valid:
            pos = drop_pos(heights, col)
            if has_won(b_self | (np.uint64(1) << np.uint64(pos))):
                return col

        for col in valid:
            pos = drop_pos(heights, col)
            if has_won(b_opp | (np.uint64(1) << np.uint64(pos))):
                return col

        candidates = sorted(valid, key=lambda c: abs(c - COLS // 2))
        if random.random() < 0.7:
            return candidates[0]
        return random.choice(candidates[:min(3, len(candidates))])

    def _order_moves(self, valid):
        valid_set = set(valid)
        return [c for c in center_first_order(COLS) if c in valid_set]


# ── Factory ──────────────────────────────────────────────────

_MCTS_AGENT = None


def create_mcts_agent(**kwargs):
    global _MCTS_AGENT
    if _MCTS_AGENT is None:
        _MCTS_AGENT = MCTSAgent(**kwargs)
    return _MCTS_AGENT


def mcts_agent_fn(observation, configuration):
    return create_mcts_agent().select_action(observation, configuration)
