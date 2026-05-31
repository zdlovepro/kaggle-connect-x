"""
ConnectX Agent — Optimized Bitboard NegaScout + Enhanced Evaluation

Algorithm:
  Bitboard (np.uint64) NegaScout (PVS) + Zobrist Transposition Table
  + Killer Heuristic + Odd/Even Threat Analysis + Opening Book

Key improvements over baseline Minimax:
  - Bitboard:  5-10x faster board operations
  - NegaScout: ~30% more pruning vs plain Alpha-Beta
  - ZobristTT: 1M-entry depth-prioritized transposition table
  - Killers:   2 killer slots per depth for better move ordering
  - Odd/Even:  Connect-4-specific threat row analysis
  - Book:      12-ply opening book from known theory lines
  - Time:      Smart iterative deepening with adaptive depth

Time complexity: O(b^(d/2)) with effective pruning
"""

import math
import time

import numpy as np

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

COLS = 7
ROWS = 6
INAROW = 4
ROW_STRIDE = ROWS + 1   # 7 bits per column (6 rows + 1 sentinel)
MAX_DEPTH = 20
TIME_BUDGET_MS = 1900.0
TT_SIZE = 1 << 20        # 1,048,576 entries
KILLER_SLOTS = 2

# Dynamic time manager (used by iterative deepening)
TIME_SAFE_BUFFER_MIN_MS = 80.0
TIME_SAFE_BUFFER_MAX_MS = 130.0
TIME_OVERAGE_FRACTION = 0.08
TIME_OVERAGE_MAX_BONUS_MS = 400.0
TIME_GROWTH_MIN = 1.2
TIME_GROWTH_MAX = 3.5

# Pre-computed bit masks for edge columns (prevent wrap-around in bit shifts)
_COL0_MASK = np.uint64(0)
_COL6_MASK = np.uint64(0)
for r in range(ROWS):
    _COL0_MASK |= np.uint64(1) << np.uint64(r * ROW_STRIDE)
    _COL6_MASK |= np.uint64(1) << np.uint64(r * ROW_STRIDE + (COLS - 1))

# Mirror normalization tables for TT key canonicalization.
_MIRROR_COL = tuple(COLS - 1 - c for c in range(COLS))
_MIRROR_POS = tuple(
    (pos // ROW_STRIDE) * ROW_STRIDE + _MIRROR_COL[pos % ROW_STRIDE]
    for pos in range(ROWS * COLS)
)

# ═══════════════════════════════════════════════════════════════
#  Zobrist Hashing
# ═══════════════════════════════════════════════════════════════

_rng = np.random.RandomState(42)
_ZOBRIST = _rng.randint(0, 2**63, size=(ROWS * COLS, 2), dtype=np.uint64)
_ZOBRIST_TURN = _rng.randint(0, 2**63, dtype=np.uint64)


def _zobrist_init(b1, b2):
    """Compute full Zobrist hash from bitboards."""
    h = np.uint64(0)
    for pos in range(ROWS * COLS):
        mask = np.uint64(1) << np.uint64(pos)
        if b1 & mask:
            h ^= _ZOBRIST[pos][0]
        elif b2 & mask:
            h ^= _ZOBRIST[pos][1]
    return h


def _zobrist_init_mirror(b1, b2):
    """Compute Zobrist hash for the left-right mirrored board."""
    h = np.uint64(0)
    for pos in range(ROWS * COLS):
        mask = np.uint64(1) << np.uint64(pos)
        mirror_pos = _MIRROR_POS[pos]
        if b1 & mask:
            h ^= _ZOBRIST[mirror_pos][0]
        elif b2 & mask:
            h ^= _ZOBRIST[mirror_pos][1]
    return h


def _zobrist_update(h, pos, player):
    """Incremental hash update: XOR in piece at pos for player (1 or 2)."""
    return h ^ _ZOBRIST[pos][player - 1]


def _canonical_tt_key(hash_key, mirror_hash_key):
    """Use min(raw_hash, mirror_hash) as canonical TT key."""
    if mirror_hash_key < hash_key:
        return mirror_hash_key, True
    return hash_key, False


def _mirror_col(col):
    return _MIRROR_COL[col]


def _tt_move_to_local(tt_move, tt_is_mirrored):
    """Map canonical TT move back to local board orientation."""
    if tt_move < 0:
        return -1
    if tt_is_mirrored:
        return _mirror_col(tt_move)
    return tt_move


def _local_move_to_tt(local_move, tt_is_mirrored):
    """Map local move into canonical TT orientation."""
    if local_move < 0:
        return -1
    if tt_is_mirrored:
        return _mirror_col(local_move)
    return local_move


# ═══════════════════════════════════════════════════════════════
#  Transposition Table
# ═══════════════════════════════════════════════════════════════

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Persistent globals (reused across agent calls)
_tt_keys = np.zeros(TT_SIZE, dtype=np.uint64)
_tt_values = np.zeros(TT_SIZE, dtype=np.float32)
_tt_depths = np.zeros(TT_SIZE, dtype=np.uint8)
_tt_flags = np.zeros(TT_SIZE, dtype=np.uint8)
_tt_moves = np.full(TT_SIZE, -1, dtype=np.int8)


def _tt_store(key, value, depth, flag, best_move=-1):
    idx = int(key % np.uint64(TT_SIZE))
    if depth >= _tt_depths[idx]:
        _tt_keys[idx] = key
        _tt_values[idx] = np.float32(value)
        _tt_depths[idx] = np.uint8(min(depth, 255))
        _tt_flags[idx] = np.uint8(flag)
        if best_move >= 0:
            _tt_moves[idx] = np.int8(best_move)


def _tt_probe(key, depth, alpha, beta):
    """Return (value, best_move) on useful hit, else None."""
    idx = int(key % np.uint64(TT_SIZE))
    if _tt_keys[idx] != key or _tt_depths[idx] < depth:
        return None
    val = float(_tt_values[idx])
    flag = int(_tt_flags[idx])
    bm = int(_tt_moves[idx]) if _tt_moves[idx] >= 0 else -1
    if flag == TT_EXACT:
        return (val, bm)
    if flag == TT_LOWER and val >= beta:
        return (beta, bm)
    if flag == TT_UPPER and val <= alpha:
        return (alpha, bm)
    return None


def _tt_get_move(key):
    """Get best move from TT for ordering (depth-agnostic)."""
    idx = int(key % np.uint64(TT_SIZE))
    if _tt_keys[idx] == key and _tt_moves[idx] >= 0:
        return int(_tt_moves[idx])
    return -1


# ═══════════════════════════════════════════════════════════════
#  Killer Move Table
# ═══════════════════════════════════════════════════════════════

_killers = np.full((MAX_DEPTH + 1, KILLER_SLOTS), -1, dtype=np.int8)
_history = np.zeros((2, COLS), dtype=np.int32)


def _killer_store(depth, col):
    if _killers[depth][0] != col:
        _killers[depth][1] = _killers[depth][0]
        _killers[depth][0] = col


def _is_killer(depth, col):
    return col == _killers[depth][0] or col == _killers[depth][1]


def _history_player_index(color):
    return 0 if color > 0 else 1


def _history_store(color, col, depth):
    """History heuristic update on fail-high cutoffs."""
    idx = _history_player_index(color)
    bonus = max(1, depth * depth)
    new_val = int(_history[idx][col]) + bonus
    if new_val > 1_000_000:
        _history[idx] = (_history[idx] // 2).astype(np.int32)
        new_val = int(_history[idx][col]) + bonus
    _history[idx][col] = np.int32(new_val)


def _history_score(color, col):
    idx = _history_player_index(color)
    return int(_history[idx][col])


# ═══════════════════════════════════════════════════════════════
#  Bitboard Operations
# ═══════════════════════════════════════════════════════════════

def _has_won(board):
    """
    Bit-shift win detection: 3 iterations of (b & b>>shift) finds 4-in-a-row.
    Edge columns are masked to prevent wrap-around false positives.
    """
    # Horizontal (shift=1): mask col6 → col0 wrap
    b = board & ~_COL6_MASK
    b = b & (b >> np.uint64(1))
    b = b & (b >> np.uint64(1))
    b = b & (b >> np.uint64(1))
    if b:
        return True
    # Vertical (shift=ROW_STRIDE=7): no wrap possible
    b = board
    b = b & (b >> np.uint64(ROW_STRIDE))
    b = b & (b >> np.uint64(ROW_STRIDE))
    b = b & (b >> np.uint64(ROW_STRIDE))
    if b:
        return True
    # Diagonal \ (shift=8): mask col6
    b = board & ~_COL6_MASK
    b = b & (b >> np.uint64(ROW_STRIDE + 1))
    b = b & (b >> np.uint64(ROW_STRIDE + 1))
    b = b & (b >> np.uint64(ROW_STRIDE + 1))
    if b:
        return True
    # Diagonal / (shift=6): mask col0
    b = board & ~_COL0_MASK
    b = b & (b >> np.uint64(ROW_STRIDE - 1))
    b = b & (b >> np.uint64(ROW_STRIDE - 1))
    b = b & (b >> np.uint64(ROW_STRIDE - 1))
    if b:
        return True
    return False


def _get_heights_from_list(board_list):
    """Compute column heights from the flat board list (faster than from bitboards)."""
    heights = np.zeros(COLS, dtype=np.uint8)
    for c in range(COLS):
        if board_list[c] == 0:  # column not full
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


def _valid_cols(heights):
    return [c for c in range(COLS) if heights[c] < ROWS]


def _drop_pos(heights, col):
    """Return bit position for dropping a piece in column col."""
    h = heights[col]
    row = ROWS - 1 - h
    return row * ROW_STRIDE + col


def _list_to_bitboards(board_list):
    """Convert flat list board to (b1, b2) bitboards."""
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


# ═══════════════════════════════════════════════════════════════
#  Evaluation Function
# ═══════════════════════════════════════════════════════════════

# === BUILD_CONFIG_START ===
# Generated by evaluate/build_submission.py. Source: defaults, D:\205zd\Desktop\Kaggle\agents\weights.py, D:\205zd\Desktop\Kaggle\agents\td_weights.py
_W_SCORE = 10
_W_THREAT = 100
_W_HMAP = 0.3
_W_ODD_EVEN = 80
_HEATMAP = np.array([
    [3, 4, 5, 7, 5, 4, 3],
    [4, 6, 8, 10, 8, 6, 4],
    [5, 8, 11, 13, 11, 8, 5],
    [5, 8, 11, 13, 11, 8, 5],
    [4, 6, 8, 10, 8, 6, 4],
    [3, 4, 5, 7, 5, 4, 3],
], dtype=np.float32)
# Format: [bias, w_2self, w_3self, w_2opp, w_3opp,
#          w_hmap_self, w_hmap_opp,
#          w_odd_self, w_even_self, w_odd_opp, w_even_opp]
_TD_WEIGHTS = [0, 0.06, 0.6, -0.06, -0.6, 0.0018, -0.0018, 0.06, 0.015, -0.06, -0.015]
# === BUILD_CONFIG_END ===

# Pre-computed heatmap value at each bitboard position (indexed by pos = r*7+c)
_HMAP_POS = np.zeros(ROWS * COLS, dtype=np.float32)
for r in range(ROWS):
    for c in range(COLS):
        _HMAP_POS[r * ROW_STRIDE + c] = _HEATMAP[r][c]


def _eval_window_count(b_self, b_opp):
    """
    Count 2-in-a-row and 3-in-a-row patterns via bitwise shift-and-accumulate.
    (4-in-a-row is caught by terminal check in search, never reaches leaf eval.
     1-in-a-row is handled by position heatmap.)
    """
    score = 0.0

    for b, sign in [(b_self, 1.0), (b_opp, -1.0)]:
        b_nor = b & ~_COL6_MASK   # prevent col6→col0 wrap
        b_nol = b & ~_COL0_MASK   # prevent col0→col6 wrap

        # 2-in-a-row
        h2 = b_nor & (b >> np.uint64(1))
        v2 = b & (b >> np.uint64(ROW_STRIDE))
        d12 = b_nor & (b >> np.uint64(ROW_STRIDE + 1))
        d22 = b_nol & (b >> np.uint64(ROW_STRIDE - 1))
        c2 = h2.bit_count() + v2.bit_count() + d12.bit_count() + d22.bit_count()
        score += sign * _W_SCORE * c2

        # 3-in-a-row
        h3 = h2 & (b >> np.uint64(2))
        v3 = v2 & (b >> np.uint64(ROW_STRIDE))
        d13 = d12 & (b >> np.uint64(ROW_STRIDE + 2))
        d23 = d22 & (b >> np.uint64(ROW_STRIDE - 2))
        c3 = h3.bit_count() + v3.bit_count() + d13.bit_count() + d23.bit_count()
        score += sign * _W_THREAT * c3

    return score


def _eval_odd_even_threats(b_self, b_opp, heights):
    """
    Odd/Even threat analysis — the critical Connect 4 endgame concept.

    A threat completing on an odd row (0-indexed from bottom) is more
    valuable because the player who creates it gets the final move in
    that column. This bonus scales up as the game progresses.

    Row parity from bottom: row_5(odd), row_4(even), row_3(odd),
    row_2(even), row_1(odd), row_0(even).
    """
    filled = sum(heights)
    # Scale: 0 at start, 1.0 at endgame
    phase = filled / (COLS * ROWS)
    scale = _W_ODD_EVEN * phase  # gradually increase threat importance

    bonus = 0.0
    for c in range(COLS):
        h = int(heights[c])
        if h < 3:
            continue
        complete_row = ROWS - 1 - h
        bottom_parity = h % 2  # 1=odd, 0=even

        all_self = True
        all_opp = True
        for r in range(complete_row + 1, complete_row + 4):
            mask = np.uint64(1) << np.uint64(r * ROW_STRIDE + c)
            if not (b_self & mask):
                all_self = False
            if not (b_opp & mask):
                all_opp = False

        if all_self:
            if bottom_parity == 1:
                bonus += 2.0 * scale + 10.0
            else:
                bonus += 0.5 * scale + 3.0
        elif all_opp:
            if bottom_parity == 1:
                bonus -= 2.0 * scale + 10.0
            else:
                bonus -= 0.5 * scale + 3.0

    return bonus


def _eval_position_heatmap(b_self, b_opp):
    """Iterate over only the set bits for O(pieces) instead of O(42)."""
    bonus = 0.0
    # Self pieces
    b = b_self
    while b:
        lsb = int(b & (~b + np.uint64(1)))  # lowest set bit
        pos = (lsb.bit_length() - 1)
        if pos < len(_HMAP_POS):
            bonus += _HMAP_POS[pos] * _W_HMAP
        b ^= lsb
    # Opponent pieces
    b = b_opp
    while b:
        lsb = int(b & (~b + np.uint64(1)))
        pos = (lsb.bit_length() - 1)
        if pos < len(_HMAP_POS):
            bonus -= _HMAP_POS[pos] * _W_HMAP
        b ^= lsb
    return bonus


def evaluate(b_self, b_opp, heights):
    """Full board evaluation from b_self's perspective."""
    if _has_won(b_self):
        return 100000.0
    if _has_won(b_opp):
        return -100000.0

    if _TD_WEIGHTS is not None:
        return _evaluate_learned(b_self, b_opp, heights)

    score = _eval_window_count(b_self, b_opp)
    score += _eval_odd_even_threats(b_self, b_opp, heights)
    score += _eval_position_heatmap(b_self, b_opp)
    return float(score)


def _evaluate_learned(b_self, b_opp, heights):
    """Learned evaluation using TD-optimized weights."""
    w = _TD_WEIGHTS
    raw = w[0]  # bias

    # Window counts (features 1-4)
    for idx, b in enumerate([b_self, b_opp]):
        base = 1 if idx == 0 else 3
        b_nor = b & ~_COL6_MASK
        b_nol = b & ~_COL0_MASK

        h2 = b_nor & (b >> np.uint64(1))
        v2 = b & (b >> np.uint64(ROW_STRIDE))
        d12 = b_nor & (b >> np.uint64(ROW_STRIDE + 1))
        d22 = b_nol & (b >> np.uint64(ROW_STRIDE - 1))
        c2 = h2.bit_count() + v2.bit_count() + d12.bit_count() + d22.bit_count()
        raw += w[base] * float(c2)

        h3 = h2 & (b >> np.uint64(2))
        v3 = v2 & (b >> np.uint64(ROW_STRIDE))
        d13 = d12 & (b >> np.uint64(ROW_STRIDE + 2))
        d23 = d22 & (b >> np.uint64(ROW_STRIDE - 2))
        c3 = h3.bit_count() + v3.bit_count() + d13.bit_count() + d23.bit_count()
        raw += w[base + 1] * float(c3)

    # Heatmap (features 5-6)
    for idx, b in enumerate([b_self, b_opp]):
        fi = 5 if idx == 0 else 6
        bb = b
        hsum = 0.0
        while bb:
            lsb = int(bb & (~bb + np.uint64(1)))
            pos = (lsb.bit_length() - 1)
            if pos < len(_HMAP_POS):
                hsum += _HMAP_POS[pos]
            bb ^= lsb
        raw += w[fi] * hsum

    # Odd/Even threats (features 7-10)
    filled = sum(int(heights[c]) for c in range(COLS))
    phase = filled / (COLS * ROWS)
    scale = phase
    for c in range(COLS):
        h = int(heights[c])
        if h < 3:
            continue
        complete_row = ROWS - 1 - h
        bottom_parity = h % 2
        all_self = True
        all_opp = True
        for r in range(complete_row + 1, complete_row + 4):
            mask = np.uint64(1) << np.uint64(r * ROW_STRIDE + c)
            if not (b_self & mask):
                all_self = False
            if not (b_opp & mask):
                all_opp = False
        if all_self:
            if bottom_parity == 1:
                raw += w[7] * (2.0 * scale + 0.125)
            else:
                raw += w[8] * (0.5 * scale + 0.0375)
        elif all_opp:
            if bottom_parity == 1:
                raw += w[9] * (2.0 * scale + 0.125)
            else:
                raw += w[10] * (0.5 * scale + 0.0375)

    return float(math.tanh(raw) * 100000.0)


# ═══════════════════════════════════════════════════════════════
#  Move Ordering
# ═══════════════════════════════════════════════════════════════

def _order_moves(valid, tt_move=-1, depth=0, b_self=None, b_opp=None, heights=None, color=1.0):
    """
    Score and sort moves:
      Immediate win > Forced block > TT best > Killer > History > Center.
    Higher score = searched first.
    """
    center = COLS // 2
    tactical = {}

    if b_self is not None and b_opp is not None and heights is not None:
        current = b_self if color > 0 else b_opp
        opponent = b_opp if color > 0 else b_self
        for col in valid:
            pos = _drop_pos(heights, col)
            mask = np.uint64(1) << np.uint64(pos)
            is_win = _has_won(current | mask)
            is_block = _has_won(opponent | mask)
            tactical[col] = (is_win, is_block)

    def _score(col):
        is_win, is_block = tactical.get(col, (False, False))
        if is_win:
            tier = 5
        elif is_block:
            tier = 4
        elif col == tt_move:
            tier = 3
        elif _is_killer(depth, col):
            tier = 2
        else:
            tier = 1

        history = _history_score(color, col)
        center_bias = 100 - abs(col - center)
        return (tier, history, center_bias)

    return sorted(valid, key=_score, reverse=True)


# ═══════════════════════════════════════════════════════════════
#  NegaScout Search
# ═══════════════════════════════════════════════════════════════

_nodes = 0
_search_start = 0.0
_timed_out = False
_search_soft_budget_ms = TIME_BUDGET_MS * 0.9
_search_hard_budget_ms = TIME_BUDGET_MS


def _calc_time_windows(fill_pct, remaining_overage_time=None):
    """Compute soft/hard budgets for this move."""
    overage_bonus = 0.0
    if remaining_overage_time is not None:
        overage_bonus = max(0.0, float(remaining_overage_time)) * 1000.0 * TIME_OVERAGE_FRACTION
        overage_bonus = min(overage_bonus, TIME_OVERAGE_MAX_BONUS_MS)

    hard_budget = TIME_BUDGET_MS + overage_bonus
    safety_buffer = TIME_SAFE_BUFFER_MIN_MS + (1.0 - fill_pct) * (TIME_SAFE_BUFFER_MAX_MS - TIME_SAFE_BUFFER_MIN_MS)
    soft_budget = max(150.0, hard_budget - safety_buffer)
    return soft_budget, hard_budget


def _estimate_next_depth_cost_ms(last_depth_ms, prev_depth_ms, fill_pct):
    """Predict the next depth cost from recent depths."""
    if last_depth_ms is None:
        if fill_pct < 0.25:
            return 45.0
        if fill_pct < 0.5:
            return 60.0
        if fill_pct < 0.75:
            return 90.0
        return 130.0

    growth = 1.8 - 0.5 * fill_pct
    if prev_depth_ms is not None and prev_depth_ms > 1e-6:
        growth = last_depth_ms / prev_depth_ms
    growth = min(max(growth, TIME_GROWTH_MIN), TIME_GROWTH_MAX)
    return last_depth_ms * growth


def _can_start_next_depth(elapsed_ms, last_depth_ms, prev_depth_ms, fill_pct, soft_budget_ms):
    """Guardrail: only start next depth if predicted completion is still safe."""
    predicted = _estimate_next_depth_cost_ms(last_depth_ms, prev_depth_ms, fill_pct)
    return (elapsed_ms + predicted) <= soft_budget_ms


def _negascout(b_self, b_opp, heights, depth, alpha, beta, color, hash_key, mirror_hash_key):
    """
    NegaScout (PVS) with TT and killer heuristic.

    color = +1: current player is max (self)
    color = -1: current player is min (opponent)
    """
    global _nodes, _timed_out, _search_hard_budget_ms
    _nodes += 1
    if _nodes % 2048 == 0:
        if (time.perf_counter() - _search_start) * 1000 > _search_hard_budget_ms:
            _timed_out = True
            return 0.0, None

    # ── TT probe ──
    tt_key, tt_key_is_mirrored = _canonical_tt_key(hash_key, mirror_hash_key)
    tt_result = _tt_probe(tt_key, depth, alpha, beta)
    tt_move = -1
    if tt_result is not None:
        tt_val, tt_best = tt_result
        return tt_val, _tt_move_to_local(tt_best, tt_key_is_mirrored)
    tt_move = _tt_move_to_local(_tt_get_move(tt_key), tt_key_is_mirrored)

    valid = _valid_cols(heights)
    if not valid:
        return 0.0, None  # draw
    if tt_move not in valid:
        tt_move = -1

    # ── Terminal: someone won on the previous move ──
    current_board = b_self if color > 0 else b_opp
    if _has_won(current_board):
        return color * 100000.0, None

    # ── Leaf node ──
    if depth == 0:
        score = evaluate(b_self, b_opp, heights)
        return color * score, None

    # ── Move ordering ──
    ordered = _order_moves(valid, tt_move, depth, b_self, b_opp, heights, color)

    best_score = -math.inf
    best_col = ordered[0]
    orig_alpha = alpha
    first_child = True

    for col in ordered:
        pos = _drop_pos(heights, col)
        mirror_pos = _MIRROR_POS[pos]
        mask = np.uint64(1) << np.uint64(pos)

        if color > 0:
            new_self, new_opp = b_self | mask, b_opp
            won = _has_won(new_self)
            new_hash = _zobrist_update(hash_key, pos, 1)
            new_mirror_hash = _zobrist_update(mirror_hash_key, mirror_pos, 1)
        else:
            new_self, new_opp = b_self, b_opp | mask
            won = _has_won(new_opp)
            new_hash = _zobrist_update(hash_key, pos, 2)
            new_mirror_hash = _zobrist_update(mirror_hash_key, mirror_pos, 2)

        heights[col] += 1

        if won:
            child_score = color * 100000.0
        elif first_child:
            child_score, _ = _negascout(
                new_self, new_opp, heights, depth - 1,
                -beta, -alpha, -color, new_hash, new_mirror_hash,
            )
            child_score = -child_score
            first_child = False
        else:
            # Null-window scout
            child_score, _ = _negascout(
                new_self, new_opp, heights, depth - 1,
                -(alpha + 1), -alpha, -color, new_hash, new_mirror_hash,
            )
            child_score = -child_score
            # Re-search full window if promising
            if child_score > alpha and child_score < beta:
                child_score, _ = _negascout(
                    new_self, new_opp, heights, depth - 1,
                    -beta, -child_score, -color, new_hash, new_mirror_hash,
                )
                child_score = -child_score

        heights[col] -= 1

        if _timed_out:
            return 0.0, None

        if child_score > best_score:
            best_score = child_score
            best_col = col

        alpha = max(alpha, best_score)
        if alpha >= beta:
            _killer_store(depth, col)
            _history_store(color, col, depth)
            tt_col = _local_move_to_tt(col, tt_key_is_mirrored)
            _tt_store(tt_key, best_score, depth, TT_LOWER, tt_col)
            return best_score, col

    flag = TT_EXACT
    if best_score <= orig_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    tt_best = _local_move_to_tt(best_col, tt_key_is_mirrored)
    _tt_store(tt_key, best_score, depth, flag, tt_best)

    return best_score, best_col


# ═══════════════════════════════════════════════════════════════
#  Opening Book
# ═══════════════════════════════════════════════════════════════

_BOOK_MAX_PLY = 12
_BOOK_LINE_STRINGS = (
    # 3-4 opening and common expert counters (trimmed to first 8-12 plies).
    "D1 d2 D3 d4 D5 e1 E2 e3 E4 b1 B2 b3",
    "D1 d2 D3 b1 B2 b3 B4 f1 F2 f3 F4 b5",
    "D1 d2 D3 c1 C2 c3 C4 g1 G2 g3 E1 e2",
    "D1 e1 A1 d2 D3 d4 B1 c1 C2 c3 E2 c4",
    "D1 e1 B1 e2 B2 b3 D2 e3 E4 a1 D3 d4",
    "D1 e1 B1 b2 A1 c1 C2 c3 C4 e2 C5 a2",
    "D1 e1 B1 e2 A1 c1 B2 b3 C2 c3 D2 a2",
    # Classic textbook expert examples (helps stabilize common practical lines).
    "D1 d2 D3 c1 C2 c3 D4 d5 G1 e1 G2 g3",
    "D1 e1 A1 c1 E2 e3 C2 c3 C4 f1 B2 a1",
)


def _book_parse_cols(line_str):
    cols = []
    for tok in line_str.split():
        col = ord(tok[0].lower()) - ord("a")
        if 0 <= col < COLS:
            cols.append(col)
    return cols


def _book_mirror_board(board):
    mirrored = [0] * (ROWS * COLS)
    for r in range(ROWS):
        base = r * COLS
        for c in range(COLS):
            mirrored[base + _MIRROR_COL[c]] = board[base + c]
    return mirrored


def _book_drop(board, heights, col, mark):
    if col < 0 or col >= COLS or heights[col] >= ROWS:
        return False
    row = ROWS - 1 - heights[col]
    board[row * COLS + col] = mark
    heights[col] += 1
    return True


def _book_register_line(book, cols):
    board = [0] * (ROWS * COLS)
    heights = [0] * COLS
    to_move = 1
    for col in cols:
        key = (tuple(board), to_move)
        if key not in book:
            book[key] = col

        mirrored_board = _book_mirror_board(board)
        mirrored_col = _mirror_col(col)
        mirrored_key = (tuple(mirrored_board), to_move)
        if mirrored_key not in book:
            book[mirrored_key] = mirrored_col

        if not _book_drop(board, heights, col, to_move):
            break
        to_move = 2 if to_move == 1 else 1


def _build_opening_book():
    book = {}
    for line in _BOOK_LINE_STRINGS:
        _book_register_line(book, _book_parse_cols(line))
    return book


_OPENING_BOOK = _build_opening_book()


def _book_move(board_list, valid, mark):
    piece_count = sum(1 for v in board_list if v != 0)
    if piece_count > _BOOK_MAX_PLY:
        return None

    # Primary: explicit expert lines + mirrored branches.
    key = (tuple(board_list), mark)
    move = _OPENING_BOOK.get(key)
    if move is not None and move in valid:
        return move

    # Fallback: conservative center control in very early game.
    if piece_count == 0 and mark == 1:
        return 3 if 3 in valid else valid[0]
    if piece_count <= 4 and 3 in valid:
        return 3
    if piece_count <= 8:
        if 2 in valid or 4 in valid:
            return 2 if 2 in valid else 4

    return None


# ═══════════════════════════════════════════════════════════════
#  Iterative Deepening Driver
# ═══════════════════════════════════════════════════════════════

def _search(b_self, b_opp, heights, hash_key, mirror_hash_key, remaining_overage_time=None):
    """Iterative deepening with adaptive start depth and time management."""
    global _nodes, _search_start, _timed_out
    global _search_soft_budget_ms, _search_hard_budget_ms

    valid = _valid_cols(heights)
    if len(valid) == 1:
        return valid[0]

    _search_start = time.perf_counter()
    _nodes = 0
    _timed_out = False

    best_col = valid[0]

    # Adaptive start depth — conservative early, aggressive late
    filled = sum(heights)
    fill_pct = filled / (COLS * ROWS)
    if fill_pct < 0.25:
        start_depth = 4
    elif fill_pct < 0.5:
        start_depth = 6
    elif fill_pct < 0.75:
        start_depth = 8
    else:
        start_depth = 10  # endgame: search deep

    _search_soft_budget_ms, _search_hard_budget_ms = _calc_time_windows(
        fill_pct, remaining_overage_time
    )
    prev_depth_ms = None
    last_depth_ms = None

    for depth in range(start_depth, MAX_DEPTH + 1):
        elapsed = (time.perf_counter() - _search_start) * 1000
        if not _can_start_next_depth(
            elapsed, last_depth_ms, prev_depth_ms, fill_pct, _search_soft_budget_ms
        ):
            break

        depth_start = time.perf_counter()
        score, move = _negascout(
            b_self, b_opp, heights, depth,
            -math.inf, math.inf, 1.0,
            hash_key, mirror_hash_key,
        )
        depth_ms = (time.perf_counter() - depth_start) * 1000
        prev_depth_ms = last_depth_ms
        last_depth_ms = max(1e-3, depth_ms)

        if _timed_out:
            break

        if move is not None:
            best_col = move

        if abs(score) >= 99999:
            break  # forced win/loss

        elapsed = (time.perf_counter() - _search_start) * 1000
        if elapsed >= _search_soft_budget_ms:
            break

    return best_col


# ═══════════════════════════════════════════════════════════════
#  Agent Entry Point (MUST be the LAST callable in this file)
# ═══════════════════════════════════════════════════════════════

def agent(observation, configuration):
    """
    ConnectX Agent — Bitboard NegaScout + Enhanced Evaluation.

    Strategy:
      1. Immediate win
      2. Block opponent win
      3. Opening book (~12 ply)
      4. Iterative deepening NegaScout search
    """
    board_list = observation.board
    mark = observation.mark
    columns = configuration.columns
    rows = configuration.rows
    inarow = configuration.inarow
    opponent_mark = 2 if mark == 1 else 1

    # Get valid moves
    valid = [c for c in range(columns) if board_list[c] == 0]
    if not valid:
        return 0
    if len(valid) == 1:
        return valid[0]

    # List → Bitboard conversion
    b1, b2 = _list_to_bitboards(board_list)
    if mark == 1:
        b_self, b_opp = b1, b2
    else:
        b_self, b_opp = b2, b1

    heights = _get_heights_from_list(board_list)
    ordered_root = _order_moves(valid, -1, 0, b_self, b_opp, heights, 1.0)

    # ── 1. Immediate win ──
    for col in ordered_root:
        pos = _drop_pos(heights, col)
        new_board = b_self | (np.uint64(1) << np.uint64(pos))
        if _has_won(new_board):
            return col

    # ── 2. Block opponent ──
    for col in ordered_root:
        pos = _drop_pos(heights, col)
        new_board = b_opp | (np.uint64(1) << np.uint64(pos))
        if _has_won(new_board):
            return col

    # ── 3. Opening book ──
    book = _book_move(board_list, valid, mark)
    if book is not None and book in valid:
        return book

    # ── 4. Search ──
    hash_key = _zobrist_init(b_self, b_opp)
    mirror_hash_key = _zobrist_init_mirror(b_self, b_opp)
    overage = getattr(observation, "remainingOverageTime", None)
    return _search(b_self, b_opp, heights, hash_key, mirror_hash_key, overage)
