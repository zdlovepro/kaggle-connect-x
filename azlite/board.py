"""Unified ConnectX board/rules utilities for AlphaZero-lite pipeline.

Default config follows Kaggle ConnectX:
  rows=6, columns=7, inarow=4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from connectx import bitboard as _shared_bitboard
except Exception:  # pragma: no cover - optional fallback for isolated use
    _shared_bitboard = None


DEFAULT_ROWS = 6
DEFAULT_COLUMNS = 7
DEFAULT_INAROW = 4


@dataclass(frozen=True)
class BoardConfig:
    rows: int = DEFAULT_ROWS
    columns: int = DEFAULT_COLUMNS
    inarow: int = DEFAULT_INAROW


DEFAULT_CONFIG = BoardConfig()


def _as_board_array(board: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    arr = np.asarray(board, dtype=np.int8)
    if arr.ndim != 2:
        raise ValueError(f"board must be 2D, got shape={arr.shape}")
    return arr


def obs_board_to_numpy(
    obs_board: Sequence[int],
    rows: int = DEFAULT_ROWS,
    columns: int = DEFAULT_COLUMNS,
    dtype: np.dtype = np.int8,
) -> np.ndarray:
    """Convert Kaggle flat `observation.board` list into shape=(rows, columns)."""
    expected = rows * columns
    if len(obs_board) != expected:
        raise ValueError(f"obs_board length must be {expected}, got {len(obs_board)}")
    return np.asarray(obs_board, dtype=dtype).reshape(rows, columns)


def numpy_to_obs_board(board: np.ndarray | Sequence[Sequence[int]]) -> List[int]:
    """Convert shape=(rows, columns) board to Kaggle flat list."""
    arr = _as_board_array(board)
    return arr.reshape(-1).astype(np.int8).tolist()


def board_to_bitboards(
    board: np.ndarray | Sequence[Sequence[int]],
    rows: int = DEFAULT_ROWS,
    columns: int = DEFAULT_COLUMNS,
) -> Tuple[np.uint64, np.uint64]:
    """Convert board to two bitboards (player1, player2).

    Reuses `connectx.bitboard.list_to_bitboards` in default 6x7 config.
    """
    arr = _as_board_array(board)
    if arr.shape != (rows, columns):
        raise ValueError(f"board shape mismatch: expected {(rows, columns)}, got {arr.shape}")

    if (
        _shared_bitboard is not None
        and rows == _shared_bitboard.ROWS
        and columns == _shared_bitboard.COLS
    ):
        return _shared_bitboard.list_to_bitboards(numpy_to_obs_board(arr))

    row_stride = rows + 1
    max_bits = row_stride * columns
    if max_bits > 64:
        raise ValueError(
            f"bitboard overflow: need {max_bits} bits (>64) for rows={rows}, cols={columns}"
        )

    b1 = np.uint64(0)
    b2 = np.uint64(0)
    for r in range(rows):
        for c in range(columns):
            v = int(arr[r, c])
            if v == 0:
                continue
            pos = r * row_stride + c
            bit = np.uint64(1) << np.uint64(pos)
            if v == 1:
                b1 |= bit
            elif v == 2:
                b2 |= bit
    return b1, b2


def legal_moves(board: np.ndarray | Sequence[Sequence[int]]) -> List[int]:
    arr = _as_board_array(board)
    return [c for c in range(arr.shape[1]) if arr[0, c] == 0]


def is_legal_move(board: np.ndarray | Sequence[Sequence[int]], col: int) -> bool:
    arr = _as_board_array(board)
    if col < 0 or col >= arr.shape[1]:
        return False
    return arr[0, col] == 0


def get_next_open_row(
    board: np.ndarray | Sequence[Sequence[int]],
    col: int,
) -> Optional[int]:
    arr = _as_board_array(board)
    if col < 0 or col >= arr.shape[1]:
        return None
    for r in range(arr.shape[0] - 1, -1, -1):
        if arr[r, col] == 0:
            return r
    return None


def apply_move(
    board: np.ndarray | Sequence[Sequence[int]],
    col: int,
    player: int,
) -> np.ndarray:
    arr = _as_board_array(board)
    if player not in (1, 2):
        raise ValueError(f"player must be 1 or 2, got {player}")
    row = get_next_open_row(arr, col)
    if row is None:
        raise ValueError(f"illegal move: column {col} is full or out of range")
    new_board = arr.copy()
    new_board[row, col] = np.int8(player)
    return new_board


def check_win(
    board: np.ndarray | Sequence[Sequence[int]],
    player: int,
    inarow: int = DEFAULT_INAROW,
) -> bool:
    arr = _as_board_array(board)
    rows, cols = arr.shape

    # Horizontal
    for r in range(rows):
        for c in range(cols - inarow + 1):
            if np.all(arr[r, c : c + inarow] == player):
                return True

    # Vertical
    for r in range(rows - inarow + 1):
        for c in range(cols):
            if np.all(arr[r : r + inarow, c] == player):
                return True

    # Main diagonal (\)
    for r in range(rows - inarow + 1):
        for c in range(cols - inarow + 1):
            if all(arr[r + i, c + i] == player for i in range(inarow)):
                return True

    # Anti diagonal (/)
    for r in range(inarow - 1, rows):
        for c in range(cols - inarow + 1):
            if all(arr[r - i, c + i] == player for i in range(inarow)):
                return True

    return False


def is_draw(board: np.ndarray | Sequence[Sequence[int]]) -> bool:
    arr = _as_board_array(board)
    return np.all(arr != 0)


def get_winner(
    board: np.ndarray | Sequence[Sequence[int]],
    inarow: int = DEFAULT_INAROW,
) -> Optional[int]:
    if check_win(board, 1, inarow=inarow):
        return 1
    if check_win(board, 2, inarow=inarow):
        return 2
    return None


def terminal_value(
    board: np.ndarray | Sequence[Sequence[int]],
    current_player: int,
    inarow: int = DEFAULT_INAROW,
) -> Optional[float]:
    """Terminal value from *current-to-move* perspective.

    Returns:
      +1.0 if current player already has a winning position
      -1.0 if opponent already has a winning position
       0.0 if draw
      None if game not finished
    """
    winner = get_winner(board, inarow=inarow)
    if winner is not None:
        return 1.0 if winner == current_player else -1.0
    if is_draw(board):
        return 0.0
    return None


def center_first_order(columns: int = DEFAULT_COLUMNS) -> List[int]:
    center = columns // 2
    return sorted(range(columns), key=lambda c: (abs(c - center), c))


def ordered_legal_moves(board: np.ndarray | Sequence[Sequence[int]]) -> List[int]:
    arr = _as_board_array(board)
    valid = set(legal_moves(arr))
    return [c for c in center_first_order(arr.shape[1]) if c in valid]


def find_immediate_win(
    board: np.ndarray | Sequence[Sequence[int]],
    player: int,
    inarow: int = DEFAULT_INAROW,
) -> Optional[int]:
    arr = _as_board_array(board)
    for col in ordered_legal_moves(arr):
        nxt = apply_move(arr, col, player)
        if check_win(nxt, player, inarow=inarow):
            return col
    return None


def find_immediate_block(
    board: np.ndarray | Sequence[Sequence[int]],
    player: int,
    opponent: int,
    inarow: int = DEFAULT_INAROW,
) -> Optional[int]:
    del player  # intent kept for API symmetry with caller side-to-move
    arr = _as_board_array(board)
    for col in ordered_legal_moves(arr):
        nxt = apply_move(arr, col, opponent)
        if check_win(nxt, opponent, inarow=inarow):
            return col
    return None


def board_to_current_player_tensor(
    board: np.ndarray | Sequence[Sequence[int]],
    current_player: int,
    include_legal_channel: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Encode board into current-player perspective channels.

    Channel layout:
      0: current player's stones
      1: opponent stones
      2: (optional) current legal drop positions mask (one-hot per legal column)
    """
    arr = _as_board_array(board)
    if current_player not in (1, 2):
        raise ValueError(f"current_player must be 1 or 2, got {current_player}")
    opp = 2 if current_player == 1 else 1

    channels = 3 if include_legal_channel else 2
    out = np.zeros((channels, arr.shape[0], arr.shape[1]), dtype=dtype)
    out[0] = (arr == current_player).astype(dtype, copy=False)
    out[1] = (arr == opp).astype(dtype, copy=False)

    if include_legal_channel:
        legal_mask = np.zeros_like(arr, dtype=dtype)
        for c in legal_moves(arr):
            r = get_next_open_row(arr, c)
            if r is not None:
                legal_mask[r, c] = 1.0
        out[2] = legal_mask

    return out


def to_tensor(
    board: np.ndarray | Sequence[Sequence[int]],
    current_player: int,
    include_legal_channel: bool = True,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Alias for board->tensor encoding used by model/evaluator pipeline."""
    return board_to_current_player_tensor(
        board=board,
        current_player=current_player,
        include_legal_channel=include_legal_channel,
        dtype=dtype,
    )


__all__ = [
    "BoardConfig",
    "DEFAULT_COLUMNS",
    "DEFAULT_CONFIG",
    "DEFAULT_INAROW",
    "DEFAULT_ROWS",
    "apply_move",
    "board_to_bitboards",
    "board_to_current_player_tensor",
    "center_first_order",
    "check_win",
    "find_immediate_block",
    "find_immediate_win",
    "get_next_open_row",
    "get_winner",
    "is_draw",
    "is_legal_move",
    "legal_moves",
    "numpy_to_obs_board",
    "obs_board_to_numpy",
    "ordered_legal_moves",
    "to_tensor",
    "terminal_value",
]
