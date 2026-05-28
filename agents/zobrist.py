"""
Zobrist 哈希 + 置换表

统一实现 — 与 submission.py 内联版本保持一致的索引方案:
  位置编码: pos = row * 7 + col  (0-41, 共 42 个有效位置)
  Zobrist 表: (42, 2) → player-1 索引

置换表:
  大小 2^20 = 1M 条目，深度优先替换策略。
  存储 (key, value, depth, flag, best_move)。
"""

import numpy as np

# ── Zobrist 随机数表 ────────────────────────────────────────
# 42 个有效 bitboard 位置 (row * 7 + col), 每个位置 2 个玩家

_rng = np.random.RandomState(42)
ZOBRIST_TABLE = _rng.randint(0, 2**63, size=(42, 2), dtype=np.uint64)

ZOBRIST_TURN = _rng.randint(0, 2**63, dtype=np.uint64)


def compute_hash(board_1: np.uint64, board_2: np.uint64,
                 turn_player: int = 0) -> np.uint64:
    """从 bitboard 计算完整 Zobrist 哈希。"""
    h = np.uint64(0)
    for pos in range(42):
        mask = np.uint64(1) << np.uint64(pos)
        if board_1 & mask:
            h ^= ZOBRIST_TABLE[pos][0]
        elif board_2 & mask:
            h ^= ZOBRIST_TABLE[pos][1]
    if turn_player:
        h ^= ZOBRIST_TURN
    return h


def hash_update(h: np.uint64, pos: int, player: int) -> np.uint64:
    """增量更新哈希 (落子时调用)。player 为 1 或 2。"""
    return h ^ ZOBRIST_TABLE[pos][player - 1]


def hash_toggle_turn(h: np.uint64) -> np.uint64:
    """切换行动方标记。"""
    return h ^ ZOBRIST_TURN


# ── 置换表 ──────────────────────────────────────────────────

TT_FLAG_EXACT = 0
TT_FLAG_LOWER = 1
TT_FLAG_UPPER = 2


class TranspositionTable:
    """Alpha-Beta 置换表 — 深度优先替换策略。"""

    SIZE = 2 ** 20  # ~1M entries

    def __init__(self, size: int = None):
        if size is None:
            size = self.SIZE
        self.size = size
        self.keys = np.zeros(size, dtype=np.uint64)
        self.values = np.zeros(size, dtype=np.float32)
        self.depths = np.zeros(size, dtype=np.uint8)
        self.flags = np.zeros(size, dtype=np.uint8)
        self.best_moves = np.zeros(size, dtype=np.int8)

    def _index(self, key: np.uint64) -> int:
        return int(key % np.uint64(self.size))

    def store(self, key: np.uint64, value: float, depth: int,
              flag: int, best_move: int = -1):
        idx = self._index(key)
        if depth >= self.depths[idx]:
            self.keys[idx] = key
            self.values[idx] = np.float32(value)
            self.depths[idx] = np.uint8(min(depth, 255))
            self.flags[idx] = np.uint8(flag)
            if best_move >= 0:
                self.best_moves[idx] = np.int8(best_move)

    def probe(self, key: np.uint64, depth: int, alpha: float, beta: float):
        idx = self._index(key)
        if self.keys[idx] != key or self.depths[idx] < depth:
            return None

        val = float(self.values[idx])
        flag = int(self.flags[idx])
        best_move = int(self.best_moves[idx]) if self.best_moves[idx] >= 0 else -1

        if flag == TT_FLAG_EXACT:
            return (val, best_move)
        elif flag == TT_FLAG_LOWER and val >= beta:
            return (beta, best_move)
        elif flag == TT_FLAG_UPPER and val <= alpha:
            return (alpha, best_move)
        return None

    def get_best_move(self, key: np.uint64) -> int:
        idx = self._index(key)
        if self.keys[idx] == key:
            move = int(self.best_moves[idx])
            if move >= 0:
                return move
        return -1

    def clear(self):
        self.keys.fill(0)
        self.values.fill(0)
        self.depths.fill(0)
        self.flags.fill(0)
        self.best_moves.fill(-1)
