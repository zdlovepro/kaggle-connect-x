"""共享 Bitboard 模块 — 所有 agent 和训练模块统一使用。

棋盘编码: np.uint64, ROW_STRIDE = 7 (6行 + 1哨兵位/列)
位置索引: pos = row * 7 + col  (row 0=顶, row 5=底)
"""

from typing import List, Tuple

import numpy as np

# ── 常量 ─────────────────────────────────────────────────────

COLS = 7
ROWS = 6
INAROW = 4
ROW_STRIDE = ROWS + 1   # 7 bits per column

# 预计算边缘列掩码 (防止位移检测时的跨列误报)
_COL0_MASK = np.uint64(0)
_COL6_MASK = np.uint64(0)
for r in range(ROWS):
    _COL0_MASK |= np.uint64(1) << np.uint64(r * ROW_STRIDE)
    _COL6_MASK |= np.uint64(1) << np.uint64(r * ROW_STRIDE + (COLS - 1))

# Public aliases so training/evaluation code can share one bitboard source.
COL0_MASK = _COL0_MASK
COL6_MASK = _COL6_MASK

# ── 位置热图 ────────────────────────────────────────────────

HEATMAP = np.array([
    [3, 4, 5, 7, 5, 4, 3],
    [4, 6, 8, 10, 8, 6, 4],
    [5, 8, 11, 13, 11, 8, 5],
    [5, 8, 11, 13, 11, 8, 5],
    [4, 6, 8, 10, 8, 6, 4],
    [3, 4, 5, 7, 5, 4, 3],
], dtype=np.float32)

HMAP_POS = np.zeros(ROWS * COLS, dtype=np.float32)
for r in range(ROWS):
    for c in range(COLS):
        HMAP_POS[r * ROW_STRIDE + c] = HEATMAP[r][c]


# ── 赢棋检测 ─────────────────────────────────────────────────

def has_won(board: np.uint64) -> bool:
    """位运算四连检测: 3 轮 (b & b>>shift) 即可判定。O(1) 时间。"""
    # 水平 (col6→col0 防止跨行回绕)
    b = board & ~_COL6_MASK
    b = b & (b >> np.uint64(1))
    b = b & (b >> np.uint64(1))
    b = b & (b >> np.uint64(1))
    if b:
        return True
    # 垂直
    b = board
    b = b & (b >> np.uint64(ROW_STRIDE))
    b = b & (b >> np.uint64(ROW_STRIDE))
    b = b & (b >> np.uint64(ROW_STRIDE))
    if b:
        return True
    # 对角线 \
    b = board & ~_COL6_MASK
    b = b & (b >> np.uint64(ROW_STRIDE + 1))
    b = b & (b >> np.uint64(ROW_STRIDE + 1))
    b = b & (b >> np.uint64(ROW_STRIDE + 1))
    if b:
        return True
    # 反对角线 /
    b = board & ~_COL0_MASK
    b = b & (b >> np.uint64(ROW_STRIDE - 1))
    b = b & (b >> np.uint64(ROW_STRIDE - 1))
    b = b & (b >> np.uint64(ROW_STRIDE - 1))
    if b:
        return True
    return False


# ── 棋盘转换 ─────────────────────────────────────────────────

def list_to_bitboards(board_list: List[int]) -> Tuple[np.uint64, np.uint64]:
    """将 42 元素的 flat list 棋盘转换为 (b1, b2) bitboard。"""
    b1 = np.uint64(0)
    b2 = np.uint64(0)
    for r in range(ROWS):
        for c in range(COLS):
            v = board_list[r * COLS + c]
            if v == 0:
                continue
            pos = r * ROW_STRIDE + c
            if v == 1:
                b1 |= np.uint64(1) << np.uint64(pos)
            elif v == 2:
                b2 |= np.uint64(1) << np.uint64(pos)
    return b1, b2


# ── 高度与合法性 ─────────────────────────────────────────────

def get_heights_from_list(board_list: List[int]) -> np.ndarray:
    """从 flat list 棋盘计算每列高度 (比从 bitboard 计算更快)。"""
    heights = np.zeros(COLS, dtype=np.int32)
    for c in range(COLS):
        if board_list[c] == 0:
            h = 0
            for r in range(ROWS - 1, -1, -1):
                if board_list[r * COLS + c] != 0:
                    h += 1
                else:
                    break
            heights[c] = h
        else:
            heights[c] = ROWS
    return heights


def get_heights(b1: np.uint64, b2: np.uint64) -> np.ndarray:
    """从 bitboard 计算每列高度 (适用于无 list board 的场景)。"""
    heights = np.zeros(COLS, dtype=np.int32)
    board = b1 | b2
    for c in range(COLS):
        h = 0
        for r in range(ROWS - 1, -1, -1):
            if board & (np.uint64(1) << np.uint64(r * ROW_STRIDE + c)):
                h += 1
            else:
                break
        heights[c] = h
    return heights


def valid_cols(heights: np.ndarray) -> List[int]:
    """返回所有未满列的索引列表。"""
    return [c for c in range(COLS) if heights[c] < ROWS]


def drop_pos(heights: np.ndarray, col: int) -> int:
    """返回在 col 列落子的 bitboard 位置索引。"""
    h = int(heights[col])
    row = ROWS - 1 - h
    return row * ROW_STRIDE + col
