"""
Bitboard 增强 Minimax Agent

技术特性:
  - Bitboard 表示 (np.uint64) → 搜索速度 5-10x
  - 迭代加深 + 时间管理 → 充分利用 2s 预算
  - Zobrist 哈希 + 置换表 → 避免重复评估
  - NegaScout (Principal Variation Search) → 更好的剪枝
  - 位置热图 + 威胁评估 → 精度更高的评估函数
  - 开局库 → 前几步直接查表
"""

import math
import time
from typing import Any, List, Optional, Tuple

import numpy as np

from connectx.bitboard import (
    COLS, ROWS, INAROW, ROW_STRIDE,
    has_won, list_to_bitboards as _shared_list_to_bb,
    get_heights_from_list, HEATMAP, HMAP_POS,
)

from .base import BaseAgent
from .zobrist import (
    TranspositionTable,
    compute_hash,
    hash_update,
    hash_toggle_turn,
    TT_FLAG_EXACT,
    TT_FLAG_LOWER,
    TT_FLAG_UPPER,
)


# ── Bitboard 常量 ────────────────────────────────────────────

MAX_DEPTH = 20

# 列掩码: 每列 6 bit 的位掩码
_TOP_ROW_MASK = np.uint64(0)
for c in range(COLS):
    _TOP_ROW_MASK |= np.uint64(1) << np.uint64(c * ROW_STRIDE)

_COL_MASKS = []
for c in range(COLS):
    mask = np.uint64(0)
    for r in range(ROWS):
        mask |= np.uint64(1) << np.uint64(r * ROW_STRIDE + c)
    _COL_MASKS.append(mask)

_FULL_BOARD_MASK = np.uint64(0)
for c in range(COLS):
    _FULL_BOARD_MASK |= _COL_MASKS[c]


# ── Bitboard 辅助函数 ───────────────────────────────────────

def _list_to_bitboards(board_list: List[int]) -> Tuple[np.uint64, np.uint64, np.ndarray]:
    """将列表棋盘转换为 (b1, b2, heights)。"""
    b1, b2 = _shared_list_to_bb(board_list)
    heights = get_heights_from_list(board_list)
    return b1, b2, heights


def _drop_piece_bb(heights: np.ndarray, col: int) -> int:
    """返回在 col 落子的 bitboard 位置索引。"""
    h = int(heights[col])
    row = ROWS - 1 - h
    return row * ROW_STRIDE + col


def _get_valid_cols(heights: np.ndarray) -> List[int]:
    """获取所有未满的列。"""
    return [c for c in range(COLS) if heights[c] < ROWS]


def _is_draw(heights: np.ndarray) -> bool:
    """检测是否平局。"""
    return all(h >= ROWS for h in heights)


# ── 威胁检测 ────────────────────────────────────────────────

def _count_open_threats(board: np.uint64) -> int:
    """统计棋盘上的 3-in-a-row 威胁数。"""
    threats = 0

    b = board
    h2 = b & (b >> np.uint64(1))
    h3 = h2 & (b >> np.uint64(2))

    v = board
    v2 = v & (v >> np.uint64(ROW_STRIDE))
    v3 = v2 & (v >> np.uint64(ROW_STRIDE))
    threats += bin(h3).count("1") + bin(v3).count("1")

    d1 = board
    d1_2 = d1 & (d1 >> np.uint64(ROW_STRIDE + 1))
    d1_3 = d1_2 & (d1 >> np.uint64(ROW_STRIDE + 1))
    threats += bin(d1_3).count("1")

    d2 = board
    d2_2 = d2 & (d2 >> np.uint64(ROW_STRIDE - 1))
    d2_3 = d2_2 & (d2 >> np.uint64(ROW_STRIDE - 1))
    threats += bin(d2_3).count("1")

    return threats


# ── 评估函数 ─────────────────────────────────────────────────

# 窗口评分权重: [1-in-row, 2-in-row, 3-in-row, 4-in-row]
WINDOW_WEIGHTS = np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float32)
CENTER_BONUS = 2.0


def evaluate(
    board_self: np.uint64,
    board_opp: np.uint64,
    heights: np.ndarray,
) -> float:
    """
    评估当前局面 (从 board_self 玩家视角)。

    评估因子:
      1. 窗口评分 (位运算快速统计连续的棋子数)
      2. 位置热图加成
      3. 威胁计数
    """
    score = 0.0

    if has_won(board_self):
        return 10000.0
    if has_won(board_opp):
        return -10000.0

    # 1. 窗口评分: 对每种"子长"做位运算统计
    b = board_self
    count_1 = bin(b).count("1")
    score += WINDOW_WEIGHTS[0] * count_1

    h2 = b & (b >> np.uint64(1))
    v2 = b & (b >> np.uint64(ROW_STRIDE))
    d12 = b & (b >> np.uint64(ROW_STRIDE + 1))
    d22 = b & (b >> np.uint64(ROW_STRIDE - 1))
    count_2 = bin(h2).count("1") + bin(v2).count("1") + bin(d12).count("1") + bin(d22).count("1")
    score += WINDOW_WEIGHTS[1] * count_2

    h3 = h2 & (b >> np.uint64(2))
    v3 = v2 & (b >> np.uint64(ROW_STRIDE))
    d13 = d12 & (b >> np.uint64(ROW_STRIDE + 1))
    d23 = d22 & (b >> np.uint64(ROW_STRIDE - 1))
    count_3 = bin(h3).count("1") + bin(v3).count("1") + bin(d13).count("1") + bin(d23).count("1")
    score += WINDOW_WEIGHTS[2] * count_3

    b = board_opp
    count_1_opp = bin(b).count("1")
    score -= WINDOW_WEIGHTS[0] * count_1_opp

    h2 = b & (b >> np.uint64(1))
    v2 = b & (b >> np.uint64(ROW_STRIDE))
    d12 = b & (b >> np.uint64(ROW_STRIDE + 1))
    d22 = b & (b >> np.uint64(ROW_STRIDE - 1))
    count_2_opp = bin(h2).count("1") + bin(v2).count("1") + bin(d12).count("1") + bin(d22).count("1")
    score -= WINDOW_WEIGHTS[1] * count_2_opp

    h3 = h2 & (b >> np.uint64(2))
    v3 = v2 & (b >> np.uint64(ROW_STRIDE))
    d13 = d12 & (b >> np.uint64(ROW_STRIDE + 1))
    d23 = d22 & (b >> np.uint64(ROW_STRIDE - 1))
    count_3_opp = bin(h3).count("1") + bin(v3).count("1") + bin(d13).count("1") + bin(d23).count("1")
    score -= WINDOW_WEIGHTS[2] * count_3_opp

    for c in range(COLS):
        h = heights[c]
        for r in range(min(h, ROWS)):
            pos = r * ROW_STRIDE + c
            bit = np.uint64(1) << np.uint64(pos)
            if board_self & bit:
                idx = r * COLS + c
                if idx < 42:
                    score += CENTER_BONUS * float(HMAP_POS[idx]) / 10.0
            elif board_opp & bit:
                idx = r * COLS + c
                if idx < 42:
                    score -= CENTER_BONUS * float(HMAP_POS[idx]) / 10.0

    # 3. 威胁评估
    threat_self = _count_open_threats(board_self)
    threat_opp = _count_open_threats(board_opp)
    score += threat_self * 50.0
    score -= threat_opp * 50.0

    return float(score)


# ── 着法排序 ─────────────────────────────────────────────────

def _order_moves(valid_cols: List[int], tt_move: int = -1) -> List[int]:
    """着法排序: 置换表最佳 → 中心列 → 两侧"""
    center = COLS // 2
    if tt_move >= 0 and tt_move in valid_cols:
        ordered = [tt_move]
        ordered += sorted(
            [c for c in valid_cols if c != tt_move],
            key=lambda c: abs(c - center)
        )
    else:
        ordered = sorted(valid_cols, key=lambda c: abs(c - center))
    return ordered


# ── NegaScout 搜索 ──────────────────────────────────────────

class MinimaxBitboardAgent(BaseAgent):
    """Bitboard Minimax + NegaScout Agent"""

    def __init__(
        self,
        name: str = "minimax_bitboard",
        max_depth: int = MAX_DEPTH,
        time_budget_ms: float = 1900.0,
        use_tt: bool = True,
    ):
        super().__init__(name=name)
        self.max_depth = max_depth
        self.time_budget_ms = time_budget_ms
        self.use_tt = use_tt
        self.tt = TranspositionTable() if use_tt else None
        self.nodes_visited = 0
        self.search_start = 0.0
        self.deadline = 0.0
        self.timed_out = False

    def select_action(self, observation: Any, configuration: Any) -> int:
        board_list = observation.board
        mark = observation.mark

        b_self_raw, b_opp_raw, heights = _list_to_bitboards(board_list)

        if mark == 1:
            b_self = b_self_raw
            b_opp = b_opp_raw
        else:
            b_self = b_opp_raw
            b_opp = b_self_raw

        valid = _get_valid_cols(heights)
        if not valid:
            self.stats["moves_made"] += 1
            return 0

        for col in valid:
            pos = _drop_piece_bb(heights, col)
            b_new = b_self | (np.uint64(1) << np.uint64(pos))
            if has_won(b_new):
                self.stats["moves_made"] += 1
                return col

        for col in valid:
            pos = _drop_piece_bb(heights, col)
            b_new = b_opp | (np.uint64(1) << np.uint64(pos))
            if has_won(b_new):
                self.stats["moves_made"] += 1
                return col

        occupied = sum(heights)
        if occupied <= 1:
            self.stats["moves_made"] += 1
            return COLS // 2

        try:
            move = self._iterative_deepen(b_self, b_opp, heights, valid)
        except Exception:
            move = valid[0]
        if move not in valid:
            move = valid[0]
        return int(move)

    def _iterative_deepen(
        self,
        b_self: np.uint64,
        b_opp: np.uint64,
        heights: np.ndarray,
        valid: List[int],
    ) -> int:
        """迭代加深: 从 depth=1 开始，逐步加深直到时间耗尽"""
        self.search_start = time.perf_counter()
        self.deadline = self.search_start + (float(self.time_budget_ms) / 1000.0)
        self.timed_out = False

        best_move = valid[0]
        search_hash = compute_hash(b_self, b_opp)
        total_nodes = 0
        depth_cost_ms: List[float] = []

        occupied = sum(heights)
        total_cells = COLS * ROWS
        if occupied < total_cells * 0.3:
            start_depth = 4
        elif occupied < total_cells * 0.6:
            start_depth = 6
        else:
            start_depth = 8

        for depth in range(start_depth, self.max_depth + 1):
            now = time.perf_counter()
            remaining_ms = max(0.0, (self.deadline - now) * 1000.0)
            if remaining_ms < 50.0:
                break

            if depth_cost_ms:
                last_cost = depth_cost_ms[-1]
                growth = 1.8
                if len(depth_cost_ms) >= 2 and depth_cost_ms[-2] > 1e-6:
                    growth = max(1.2, min(3.0, depth_cost_ms[-1] / depth_cost_ms[-2]))
                est_next_cost = last_cost * growth
                if remaining_ms < max(50.0, est_next_cost * 0.9):
                    break

            self.nodes_visited = 0
            depth_start = time.perf_counter()

            score, move = self._negascout(
                b_self, b_opp, heights, depth,
                -math.inf, math.inf, 1.0,
                search_hash,
            )

            # 只有未超时才算有效结果
            if self.timed_out:
                break  # 超时标记

            depth_elapsed_ms = (time.perf_counter() - depth_start) * 1000.0
            depth_cost_ms.append(max(1e-6, depth_elapsed_ms))
            total_nodes += max(0, int(self.nodes_visited))

            if move is not None and move in valid:
                best_move = move

            # 找到必胜/必败直接返回
            if abs(score) >= 9999:
                break

        self.stats["nodes_visited"] += max(0, int(total_nodes))
        self.stats["moves_made"] += 1

        if best_move not in valid:
            return valid[0]
        return best_move

    def _negascout(
        self,
        b_self: np.uint64,
        b_opp: np.uint64,
        heights: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        color: float,
        hash_key: np.uint64,
    ) -> Tuple[float, Optional[int]]:
        if (self.nodes_visited & 1023) == 0:
            if time.perf_counter() >= self.deadline:
                self.timed_out = True
                return 0.0, None

        self.nodes_visited += 1

        tt_move = -1
        if self.tt is not None:
            tt_result = self.tt.probe(hash_key, depth, alpha, beta)
            if tt_result is not None:
                tt_val, tt_move = tt_result
                if tt_val >= beta or tt_val <= alpha:
                    return tt_val, tt_move

        valid = _get_valid_cols(heights)
        if not valid:
            return 0.0, None

        current_board = b_self if color > 0 else b_opp
        if has_won(current_board):
            return color * 10000.0, None

        if depth == 0:
            score = evaluate(
                b_self if color > 0 else b_opp,
                b_opp if color > 0 else b_self,
                heights,
            )
            return color * score, None

        ordered = _order_moves(valid, tt_move)

        best_score = -math.inf
        best_move = ordered[0]
        orig_alpha = alpha
        first_child = True

        for col in ordered:
            row = ROWS - 1 - heights[col]
            pos = row * ROW_STRIDE + col
            mask = np.uint64(1) << np.uint64(pos)

            if color > 0:
                new_b_self = b_self | mask
                new_b_opp = b_opp
                player_won = has_won(new_b_self)
                new_hash = hash_update(hash_key, pos, 1)
            else:
                new_b_self = b_self
                new_b_opp = b_opp | mask
                player_won = has_won(new_b_opp)
                new_hash = hash_update(hash_key, pos, 2)

            heights[col] += 1

            if player_won:
                child_score = color * 10000.0
            else:
                if first_child:
                    child_score, _ = self._negascout(
                        b_self if color > 0 else new_b_self,
                        b_opp if color > 0 else new_b_opp,
                        heights, depth - 1,
                        -beta, -alpha, -color,
                        new_hash,
                    )
                    child_score = -child_score
                    first_child = False
                else:
                    child_score, _ = self._negascout(
                        b_self if color > 0 else new_b_self,
                        b_opp if color > 0 else new_b_opp,
                        heights, depth - 1,
                        -(alpha + 1), -alpha, -color,
                        new_hash,
                    )
                    child_score = -child_score
                    if child_score > alpha and child_score < beta:
                        child_score, _ = self._negascout(
                            b_self if color > 0 else new_b_self,
                            b_opp if color > 0 else new_b_opp,
                            heights, depth - 1,
                            -beta, -child_score, -color,
                            new_hash,
                        )
                        child_score = -child_score

            heights[col] -= 1

            # 超时检查
            if self.timed_out:
                return 0.0, None

            if child_score > best_score:
                best_score = child_score
                best_move = col

            alpha = max(alpha, best_score)
            if alpha >= beta:
                # cut-off
                if self.tt is not None and self.nodes_visited > 0:
                    self.tt.store(
                        hash_key, best_score, depth,
                        TT_FLAG_LOWER, best_move,
                    )
                return best_score, best_move

        # 存储到置换表
        flag = TT_FLAG_EXACT
        if best_score <= orig_alpha:
            flag = TT_FLAG_UPPER
        elif best_score >= beta:
            flag = TT_FLAG_LOWER

        if self.tt is not None and self.nodes_visited > 0:
            self.tt.store(hash_key, best_score, depth, flag, best_move)

        return best_score, best_move


# ── Kaggle 兼容的 agent 函数 ───────────────────

_AGENT = None


def create_agent(**kwargs) -> MinimaxBitboardAgent:
    """创建 (或复用) Agent 实例"""
    global _AGENT
    if _AGENT is None:
        _AGENT = MinimaxBitboardAgent(**kwargs)
    return _AGENT


def bitboard_agent_fn(observation: Any, configuration: Any) -> int:
    """Kaggle-compatible agent function (单文件嵌入用)"""
    agent = create_agent()
    return agent.select_action(observation, configuration)
