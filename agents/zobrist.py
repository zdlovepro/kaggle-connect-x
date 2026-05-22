"""
Zobrist 哈希 + 置换表

用于:
  - 快速对阵局面做哈希 (避免重复评估)
  - 置换表存取 (alpha-beta 剪枝中的缓存)

Zobrist 哈希:
  预生成 42*3 个随机 uint64 值，对应每个格子的 3 种状态(0/1/2)。
  通过 XOR 操作增量更新哈希值，计算开销极低。

置换表:
  大小约 2^20 = 1M 条目，深度优先替换策略。
  存储 (key, value, depth, flag)，其中 flag 表示:
    0 = exact value
    1 = lower bound
    2 = upper bound
"""

import numpy as np

# ── Zobrist 随机数表 ────────────────────────────────────────
# 使用位运算位置编码 (pos = row * (ROWS+1) + col), 最大 pos = 5*8+6 = 46
# 扩展到 56 个条目以覆盖所有合法位运算位置
_rng = np.random.RandomState(42)
ZOBRIST_TABLE = _rng.randint(0, 2**63, size=(56, 3), dtype=np.uint64)

# 换手标记 (用于区分同局面下已方/对手行动)
ZOBRIST_TURN = _rng.randint(0, 2**63, dtype=np.uint64)


def compute_hash(board_1: np.uint64, board_2: np.uint64, turn_player: int = 0) -> np.uint64:
    """
    从 bitboard 计算完整 Zobrist 哈希。

    Args:
        board_1: 玩家1 的 bitboard
        board_2: 玩家2 的 bitboard
        turn_player: 当前行动玩家 (0=无, 1=玩家1, 2=玩家2)

    Returns:
        uint64 哈希值
    """
    h = np.uint64(0)
    # 遍历所有有效位运算位置 (最大 pos = 5*8+6 = 46)
    for pos in range(46):
        mask = np.uint64(1) << np.uint64(pos)
        if board_1 & mask:
            h ^= ZOBRIST_TABLE[pos][1]
        elif board_2 & mask:
            h ^= ZOBRIST_TABLE[pos][2]
    if turn_player:
        h ^= ZOBRIST_TURN
    return h


def hash_update(h: np.uint64, pos: int, player: int) -> np.uint64:
    """
    增量更新 Zobrist 哈希 (落子时调用)。

    Args:
        h: 当前哈希值
        pos: 落子位置 (0-41)
        player: 落子玩家 (1 或 2)

    Returns:
        更新后的哈希值
    """
    return h ^ ZOBRIST_TABLE[pos][player]


def hash_toggle_turn(h: np.uint64) -> np.uint64:
    """切换行动方标记"""
    return h ^ ZOBRIST_TURN


# ── 置换表 (Transposition Table) ────────────────────────────

TT_FLAG_EXACT = 0
TT_FLAG_LOWER = 1
TT_FLAG_UPPER = 2


class TranspositionTable:
    """Alpha-Beta 置换表

    大小固定，深度优先替换策略。
    用 numpy 数组存储以获得更好的缓存局部性。
    """

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

    def store(
        self,
        key: np.uint64,
        value: float,
        depth: int,
        flag: int,
        best_move: int = -1,
    ):
        """
        存储一个评估结果。

        Args:
            key: Zobrist 哈希
            value: 评估值
            depth: 搜索深度
            flag: TT_FLAG_EXACT / TT_FLAG_LOWER / TT_FLAG_UPPER
            best_move: 该局面下的最优动作 (-1 表示无)
        """
        idx = self._index(key)
        # 深度优先替换: 只有当新深度 >= 已有深度时才替换
        if depth >= self.depths[idx]:
            self.keys[idx] = key
            self.values[idx] = np.float32(value)
            self.depths[idx] = np.uint8(min(depth, 255))
            self.flags[idx] = np.uint8(flag)
            if best_move >= 0:
                self.best_moves[idx] = np.int8(best_move)

    def probe(self, key: np.uint64, depth: int, alpha: float, beta: float):
        """
        查询置换表。

        Returns:
            - (value, best_move) 如果命中且可用
            - None 如果未命中
        """
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
        """仅获取最佳动作 (用于迭代加深时的着法排序)"""
        idx = self._index(key)
        if self.keys[idx] == key:
            move = int(self.best_moves[idx])
            if move >= 0:
                return move
        return -1

    def hit_rate(self) -> float:
        """查询命中率需要在搜索过程中统计，这里返回估算"""
        return 0.0

    def clear(self):
        """清空置换表"""
        self.keys.fill(0)
        self.values.fill(0)
        self.depths.fill(0)
        self.flags.fill(0)
        self.best_moves.fill(-1)
