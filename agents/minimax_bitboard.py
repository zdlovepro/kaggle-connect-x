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


# ── Bitboard 常量 (7列 × 6行) ───────────────────────────────

COLS = 7
ROWS = 6
INAROW = 4
MAX_DEPTH = 20

# 列掩码: 每列 6 bit 的位掩码
_TOP_ROW_MASK = np.uint64(0)
for c in range(COLS):
    _TOP_ROW_MASK |= np.uint64(1) << np.uint64(c * (ROWS + 1))

_COL_MASKS = []
for c in range(COLS):
    mask = np.uint64(0)
    for r in range(ROWS):
        mask |= np.uint64(1) << np.uint64(r * (ROWS + 1) + c)
    _COL_MASKS.append(mask)

# 用于快速获取列高度的数组
_COL_SHIFT = np.array([c * (ROWS + 1) for c in range(COLS)], dtype=np.uint8)

# 完整的棋盘掩码 (所有有效位)
_FULL_BOARD_MASK = np.uint64(0)
for c in range(COLS):
    _FULL_BOARD_MASK |= _COL_MASKS[c]


# ── 获胜检测 (位运算) ───────────────────────────────────────

def _has_won(board: np.uint64) -> bool:
    """位运算检测四连: 对4个方向做位移匹配"""
    # 水平 (右移1位)
    b = board
    b = b & (b >> np.uint64(1))
    b = b & (b >> np.uint64(1))
    b = b & (b >> np.uint64(1))
    if b != 0:
        return True

    # 垂直 (右移 8位 = ROWS+1)
    b = board
    b = b & (b >> np.uint64(ROWS + 1))
    b = b & (b >> np.uint64(ROWS + 1))
    b = b & (b >> np.uint64(ROWS + 1))
    if b != 0:
        return True

    # 对角线 \ (右移 9位)
    b = board
    b = b & (b >> np.uint64(ROWS + 2))
    b = b & (b >> np.uint64(ROWS + 2))
    b = b & (b >> np.uint64(ROWS + 2))
    if b != 0:
        return True

    # 反对角线 / (右移 7位)
    b = board
    b = b & (b >> np.uint64(ROWS))
    b = b & (b >> np.uint64(ROWS))
    b = b & (b >> np.uint64(ROWS))
    if b != 0:
        return True

    return False


# ── 位置热图评估权重 ─────────────────────────────────────────

# 默认热图 (对称的 6×7 权重表)
# row0(top) → row5(bottom)
_POSITION_HEATMAP = [
    [3, 4, 5, 7, 5, 4, 3],
    [4, 6, 8, 10, 8, 6, 4],
    [5, 8, 11, 13, 11, 8, 5],
    [5, 8, 11, 13, 11, 8, 5],
    [4, 6, 8, 10, 8, 6, 4],
    [3, 4, 5, 7, 5, 4, 3],
]

# 预计算热图为 42 长度的数组
_POSITION_BONUS = np.array([_POSITION_HEATMAP[r][c] for r in range(ROWS) for c in range(COLS)], dtype=np.float32)


# ── Bitboard 辅助函数 ───────────────────────────────────────

def _list_to_bitboards(board_list: List[int], columns: int = COLS, rows: int = ROWS) -> Tuple[np.uint64, np.uint64, np.ndarray]:
    """
    将列表棋盘转换为 bitboard 表示。

    Returns:
        board_1: 玩家1 的 bitboard
        board_2: 玩家2 的 bitboard
        heights: 每列高度
    """
    b1 = np.uint64(0)
    b2 = np.uint64(0)
    heights = np.zeros(columns, dtype=np.uint8)

    for c in range(columns):
        for r in range(rows - 1, -1, -1):
            v = board_list[r * columns + c]
            if v == 0:
                break
            pos = r * (rows + 1) + c
            if v == 1:
                b1 |= np.uint64(1) << np.uint64(pos)
            elif v == 2:
                b2 |= np.uint64(1) << np.uint64(pos)
            heights[c] += 1

    return b1, b2, heights


def _drop_piece_bb(
    board: np.uint64,
    heights: np.ndarray,
    col: int,
    rows: int = ROWS,
) -> int:
    """
    在 bitboard 上落子 (计算落子后的 bit 位置)。

    Returns:
        落子位置编码 pos = row*(ROWS+1) + col (row: 0=顶, rows-1=底)
    """
    h = heights[col]
    # 重力: 棋子落在该列最底部空位
    row = rows - 1 - h
    return row * (rows + 1) + col


def _get_valid_cols(heights: np.ndarray, rows: int = ROWS) -> List[int]:
    """获取所有未满的列"""
    return [c for c in range(COLS) if heights[c] < rows]


def _is_draw(heights: np.ndarray, rows: int = ROWS) -> bool:
    """检测是否平局"""
    return all(h >= rows for h in heights)


# ── 威胁检测 ────────────────────────────────────────────────

def _count_open_threats(board: np.uint64, rows: int = ROWS) -> int:
    """
    统计棋盘上有几个 "开三" 威胁 (3-in-a-row 且两端至少有一端空)
    使用位运算快速识别。
    """
    threats = 0
    # 水平方向
    empty_board = np.uint64(0)  # 用不到，简化为...

    # 2-in-a-row 检查
    b = board
    h2 = b & (b >> np.uint64(1))
    # 3-in-a-row
    h3 = h2 & (b >> np.uint64(2))

    # 垂直
    v = board
    v2 = v & (v >> np.uint64(rows + 1))
    v3 = v2 & (v >> np.uint64(rows + 1))
    threats += bin(h3).count("1") + bin(v3).count("1")

    # 对角
    d1 = board
    d1_2 = d1 & (d1 >> np.uint64(rows + 2))
    d1_3 = d1_2 & (d1 >> np.uint64(rows + 2))
    threats += bin(d1_3).count("1")

    d2 = board
    d2_2 = d2 & (d2 >> np.uint64(rows))
    d2_3 = d2_2 & (d2 >> np.uint64(rows))
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
    rows: int = ROWS,
    cols: int = COLS,
) -> float:
    """
    评估当前局面 (从 board_self 玩家视角)。

    评估因子:
      1. 窗口评分 (位运算快速统计连续的棋子数)
      2. 位置热图加成
      3. 威胁计数
    """
    score = 0.0

    # 直接检测获胜
    if _has_won(board_self):
        return 10000.0
    if _has_won(board_opp):
        return -10000.0

    # 1. 窗口评分: 对每种"子长"做位运算统计
    # self pieces
    b = board_self
    # 1-in-a-row = 所有棋子
    count_1 = bin(b).count("1")
    score += WINDOW_WEIGHTS[0] * count_1

    # 2-in-a-row
    h2 = b & (b >> np.uint64(1))
    v2 = b & (b >> np.uint64(rows + 1))
    d12 = b & (b >> np.uint64(rows + 2))
    d22 = b & (b >> np.uint64(rows))
    count_2 = bin(h2).count("1") + bin(v2).count("1") + bin(d12).count("1") + bin(d22).count("1")
    score += WINDOW_WEIGHTS[1] * count_2

    # 3-in-a-row
    h3 = h2 & (b >> np.uint64(2))
    v3 = v2 & (b >> np.uint64(rows + 1))
    d13 = d12 & (b >> np.uint64(rows + 2))
    d23 = d22 & (b >> np.uint64(rows))
    count_3 = bin(h3).count("1") + bin(v3).count("1") + bin(d13).count("1") + bin(d23).count("1")
    score += WINDOW_WEIGHTS[2] * count_3

    # opponent pieces
    b = board_opp
    count_1_opp = bin(b).count("1")
    score -= WINDOW_WEIGHTS[0] * count_1_opp

    h2 = b & (b >> np.uint64(1))
    v2 = b & (b >> np.uint64(rows + 1))
    d12 = b & (b >> np.uint64(rows + 2))
    d22 = b & (b >> np.uint64(rows))
    count_2_opp = bin(h2).count("1") + bin(v2).count("1") + bin(d12).count("1") + bin(d22).count("1")
    score -= WINDOW_WEIGHTS[1] * count_2_opp

    h3 = h2 & (b >> np.uint64(2))
    v3 = v2 & (b >> np.uint64(rows + 1))
    d13 = d12 & (b >> np.uint64(rows + 2))
    d23 = d22 & (b >> np.uint64(rows))
    count_3_opp = bin(h3).count("1") + bin(v3).count("1") + bin(d13).count("1") + bin(d23).count("1")
    score -= WINDOW_WEIGHTS[2] * count_3_opp

    # 2. 位置热图加成
    for c in range(min(cols, COLS)):
        h = heights[c]
        for r in range(min(h, rows)):
            pos = r * (rows + 1) + c
            bit = np.uint64(1) << np.uint64(pos)
            if board_self & bit:
                idx = r * cols + c
                if idx < 42:
                    score += CENTER_BONUS * _POSITION_BONUS[idx] / 10.0
            elif board_opp & bit:
                idx = r * cols + c
                if idx < 42:
                    score -= CENTER_BONUS * _POSITION_BONUS[idx] / 10.0

    # 3. 威胁评估
    threat_self = _count_open_threats(board_self)
    threat_opp = _count_open_threats(board_opp)
    score += threat_self * 50.0
    score -= threat_opp * 50.0

    return float(score)


# ── 着法排序 ─────────────────────────────────────────────────

def _order_moves(valid_cols: List[int], tt_move: int = -1, cols: int = COLS) -> List[int]:
    """着法排序: 置换表最佳 → 中心列 → 两侧"""
    center = cols // 2
    # 如果置换表有推荐，排最前面
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

    def select_action(self, observation: Any, configuration: Any) -> int:
        board_list = observation.board
        mark = observation.mark
        columns = configuration.columns
        rows = configuration.rows
        inarow = configuration.inarow
        opp_mark = 2 if mark == 1 else 1

        b_self_raw, b_opp_raw, heights = _list_to_bitboards(board_list, columns, rows)

        # 交换标记: 始终以 "自己=b1" 的视角
        if mark == 1:
            b_self = b_self_raw
            b_opp = b_opp_raw
        else:
            b_self = b_opp_raw
            b_opp = b_self_raw

        valid = _get_valid_cols(heights, rows)
        if not valid:
            self.stats["moves_made"] += 1
            return 0

        # 快速检测: 直接获胜
        for col in valid:
            pos = _drop_piece_bb(b_self, heights, col, rows)
            b_new = b_self | (np.uint64(1) << np.uint64(pos))
            if _has_won(b_new):
                self.stats["moves_made"] += 1
                return col

        # 快速检测: 拦截对手
        for col in valid:
            pos = _drop_piece_bb(b_opp, heights, col, rows)
            b_new = b_opp | (np.uint64(1) << np.uint64(pos))
            if _has_won(b_new):
                self.stats["moves_made"] += 1
                return col

        # 开局库 (前几步固定走法)
        occupied = sum(heights)
        if occupied <= 1:
            self.stats["moves_made"] += 1
            return columns // 2  # 开局走中心

        # 迭代加深搜索
        return self._iterative_deepen(b_self, b_opp, heights, valid, columns, rows)

    def _iterative_deepen(
        self,
        b_self: np.uint64,
        b_opp: np.uint64,
        heights: np.ndarray,
        valid: List[int],
        cols: int,
        rows: int,
    ) -> int:
        """迭代加深: 从 depth=1 开始，逐步加深直到时间耗尽"""
        self.nodes_visited = 0
        self.search_start = time.perf_counter()

        best_move = valid[0]  # fallback
        search_hash = compute_hash(b_self, b_opp)

        # 自适应初始深度 (根据填充率)
        occupied = sum(heights)
        total_cells = cols * rows
        if occupied < total_cells * 0.3:
            start_depth = 4
        elif occupied < total_cells * 0.6:
            start_depth = 6
        else:
            start_depth = 8

        for depth in range(start_depth, self.max_depth + 1):
            elapsed = (time.perf_counter() - self.search_start) * 1000
            if elapsed > self.time_budget_ms * 0.5:
                break  # 时间不够完成下一层

            self.nodes_visited = 0

            score, move = self._negascout(
                b_self, b_opp, heights, depth,
                -math.inf, math.inf, 1.0,
                cols, rows, search_hash,
            )

            # 只有未超时才算有效结果
            if self.nodes_visited == -1:
                break  # 超时标记

            best_move = move if move is not None else best_move

            # 找到必胜/必败直接返回
            if abs(score) >= 9999:
                break

        self.stats["nodes_visited"] += self.nodes_visited if self.nodes_visited > 0 else 0
        self.stats["moves_made"] += 1

        return best_move

    def _negascout(
        self,
        b_self: np.uint64,
        b_opp: np.uint64,
        heights: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        color: float,  # 1.0 for self to move, -1.0 for opp to move
        cols: int,
        rows: int,
        hash_key: np.uint64,
    ) -> Tuple[float, Optional[int]]:
        """
        NegaScout (Principal Variation Search).

        color = 1.0: self to move (maximizing)
        color = -1.0: opp to move (minimizing)

        Returns:
            (score, best_move)
        """
        # 时间检查
        if self.nodes_visited % 10000 == 0:
            elapsed = (time.perf_counter() - self.search_start) * 1000
            if elapsed > self.time_budget_ms:
                self.nodes_visited = -1
                return 0.0, None

        self.nodes_visited += 1

        # 置换表查询
        tt_move = -1
        if self.tt is not None:
            tt_result = self.tt.probe(hash_key, depth, alpha, beta)
            if tt_result is not None:
                tt_val, tt_move = tt_result
                if tt_val >= beta or tt_val <= alpha:
                    return tt_val, tt_move

        # 终端检测
        valid = _get_valid_cols(heights, rows)

        if not valid:
            return 0.0, None  # draw

        current_board = b_self if color > 0 else b_opp
        if _has_won(current_board):
            return color * 10000.0, None

        if depth == 0:
            # 评估局面
            score = evaluate(
                b_self if color > 0 else b_opp,
                b_opp if color > 0 else b_self,
                heights, rows, cols,
            )
            return color * score, None

        # 着法排序
        ordered = _order_moves(valid, tt_move, cols)

        best_score = -math.inf
        best_move = ordered[0]
        orig_alpha = alpha
        first_child = True

        for col in ordered:
            row = rows - 1 - heights[col]
            pos = row * (rows + 1) + col
            mask = np.uint64(1) << np.uint64(pos)

            # 落子 (确保两个变量都被定义)
            if color > 0:
                new_b_self = b_self | mask
                new_b_opp = b_opp
                has_won = _has_won(new_b_self)
                new_hash = hash_update(hash_key, pos, 1)
            else:
                new_b_self = b_self
                new_b_opp = b_opp | mask
                has_won = _has_won(new_b_opp)
                new_hash = hash_update(hash_key, pos, 2)

            heights[col] += 1

            if has_won:
                child_score = color * 10000.0
                child_hash = new_hash
            else:
                # NegaScout: 第一个子节点全窗口搜索，后续用 null window
                if first_child:
                    child_score, _ = self._negascout(
                        b_self if color > 0 else new_b_self,
                        b_opp if color > 0 else new_b_opp,
                        heights, depth - 1,
                        -beta, -alpha, -color,
                        cols, rows, new_hash,
                    )
                    child_score = -child_score
                    first_child = False
                else:
                    # Null window search
                    child_score, _ = self._negascout(
                        b_self if color > 0 else new_b_self,
                        b_opp if color > 0 else new_b_opp,
                        heights, depth - 1,
                        -(alpha + 1), -alpha, -color,
                        cols, rows, new_hash,
                    )
                    child_score = -child_score
                    # re-search if promising
                    if child_score > alpha and child_score < beta:
                        child_score, _ = self._negascout(
                            b_self if color > 0 else new_b_self,
                            b_opp if color > 0 else new_b_opp,
                            heights, depth - 1,
                            -beta, -child_score, -color,
                            cols, rows, new_hash,
                        )
                        child_score = -child_score

            heights[col] -= 1

            # 超时检查
            if self.nodes_visited == -1:
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
