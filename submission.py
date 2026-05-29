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
  - Book:      8-ply opening book from known theory
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

# Pre-computed bit masks for edge columns (prevent wrap-around in bit shifts)
_COL0_MASK = np.uint64(0)
_COL6_MASK = np.uint64(0)
for r in range(ROWS):
    _COL0_MASK |= np.uint64(1) << np.uint64(r * ROW_STRIDE)
    _COL6_MASK |= np.uint64(1) << np.uint64(r * ROW_STRIDE + (COLS - 1))

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


def _zobrist_update(h, pos, player):
    """Incremental hash update: XOR in piece at pos for player (1 or 2)."""
    return h ^ _ZOBRIST[pos][player - 1]


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


def _killer_store(depth, col):
    if _killers[depth][0] != col:
        _killers[depth][1] = _killers[depth][0]
        _killers[depth][0] = col


def _is_killer(depth, col):
    return col == _killers[depth][0] or col == _killers[depth][1]


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

def _order_moves(valid, tt_move=-1, depth=0):
    """
    Score and sort moves: TT best > Killer > Center > Edges.
    Higher score = searched first.
    """
    center = COLS // 2

    def _score(col):
        if col == tt_move:
            return 1000000
        if _is_killer(depth, col):
            return 100000
        return 100 - abs(col - center)

    return sorted(valid, key=_score, reverse=True)


# ═══════════════════════════════════════════════════════════════
#  NegaScout Search
# ═══════════════════════════════════════════════════════════════

_nodes = 0
_search_start = 0.0
_timed_out = False


def _negascout(b_self, b_opp, heights, depth, alpha, beta, color, hash_key):
    """
    NegaScout (PVS) with TT and killer heuristic.

    color = +1: current player is max (self)
    color = -1: current player is min (opponent)
    """
    global _nodes, _timed_out
    _nodes += 1
    if _nodes % 2048 == 0:
        if (time.perf_counter() - _search_start) * 1000 > TIME_BUDGET_MS:
            _timed_out = True
            return 0.0, None

    # ── TT probe ──
    tt_result = _tt_probe(hash_key, depth, alpha, beta)
    tt_move = -1
    if tt_result is not None:
        return tt_result
    tt_move = _tt_get_move(hash_key)

    valid = _valid_cols(heights)
    if not valid:
        return 0.0, None  # draw

    # ── Terminal: someone won on the previous move ──
    current_board = b_self if color > 0 else b_opp
    if _has_won(current_board):
        return color * 100000.0, None

    # ── Leaf node ──
    if depth == 0:
        score = evaluate(b_self, b_opp, heights)
        return color * score, None

    # ── Move ordering ──
    ordered = _order_moves(valid, tt_move, depth)

    best_score = -math.inf
    best_col = ordered[0]
    orig_alpha = alpha
    first_child = True

    for col in ordered:
        pos = _drop_pos(heights, col)
        mask = np.uint64(1) << np.uint64(pos)

        if color > 0:
            new_self, new_opp = b_self | mask, b_opp
            won = _has_won(new_self)
            new_hash = _zobrist_update(hash_key, pos, 1)
        else:
            new_self, new_opp = b_self, b_opp | mask
            won = _has_won(new_opp)
            new_hash = _zobrist_update(hash_key, pos, 2)

        heights[col] += 1

        if won:
            child_score = color * 100000.0
        elif first_child:
            child_score, _ = _negascout(
                new_self, new_opp, heights, depth - 1,
                -beta, -alpha, -color, new_hash,
            )
            child_score = -child_score
            first_child = False
        else:
            # Null-window scout
            child_score, _ = _negascout(
                new_self, new_opp, heights, depth - 1,
                -(alpha + 1), -alpha, -color, new_hash,
            )
            child_score = -child_score
            # Re-search full window if promising
            if child_score > alpha and child_score < beta:
                child_score, _ = _negascout(
                    new_self, new_opp, heights, depth - 1,
                    -beta, -child_score, -color, new_hash,
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
            _tt_store(hash_key, best_score, depth, TT_LOWER, col)
            return best_score, col

    flag = TT_EXACT
    if best_score <= orig_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER
    _tt_store(hash_key, best_score, depth, flag, best_col)

    return best_score, best_col


# ═══════════════════════════════════════════════════════════════
#  Opening Book
# ═══════════════════════════════════════════════════════════════

def _book_move(board_list, valid, mark):
    """
    Hardcoded opening book (~8 ply) from Connect 4 theory.

    Theory:
      - P1's strongest opening: D1 (col 3).
      - P2's only drawing response to D1: D2 (col 3).
      - P1's 3rd move after D1-D2: D3 (col 3) maintains advantage.
      - If P1 goes C1 (col 2), P2 should take center or play D2.
    """
    piece_count = sum(1 for v in board_list if v != 0)
    if piece_count >= 8:
        return None  # Out of book range

    if mark == 1:
        if piece_count == 0:
            return 3  # D1: always open center
        if piece_count == 2:
            # Find opponent's first move (bottom-most piece)
            opp_col = None
            for c in range(COLS):
                if board_list[5 * COLS + c] != 0:
                    opp_col = c
                    break
            if opp_col == 3:
                return 3 if 3 in valid else (2 if 2 in valid else 4)
            elif opp_col in (2, 4):
                return 3 if 3 in valid else (COLS // 2)
            else:
                return 3
        if piece_count == 4:
            return 3 if 3 in valid else (2 if 2 in valid else 4)

    if mark == 2:
        if piece_count == 1:
            return 3  # Respond to center with center
        if piece_count == 3:
            return 3 if 3 in valid else (2 if 2 in valid else 4)

    return None


# ═══════════════════════════════════════════════════════════════
#  Iterative Deepening Driver
# ═══════════════════════════════════════════════════════════════

def _search(b_self, b_opp, heights, hash_key):
    """Iterative deepening with adaptive start depth and time management."""
    global _nodes, _search_start, _timed_out

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

    for depth in range(start_depth, MAX_DEPTH + 1):
        elapsed = (time.perf_counter() - _search_start) * 1000
        if elapsed > TIME_BUDGET_MS * 0.4:
            break

        score, move = _negascout(
            b_self, b_opp, heights, depth,
            -math.inf, math.inf, 1.0,
            hash_key,
        )

        if _timed_out:
            break

        if move is not None:
            best_col = move

        if abs(score) >= 99999:
            break  # forced win/loss

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
      3. Opening book (~8 ply)
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

    # ── 1. Immediate win ──
    for col in valid:
        pos = _drop_pos(heights, col)
        new_board = b_self | (np.uint64(1) << np.uint64(pos))
        if _has_won(new_board):
            return col

    # ── 2. Block opponent ──
    for col in valid:
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
    return _search(b_self, b_opp, heights, hash_key)
