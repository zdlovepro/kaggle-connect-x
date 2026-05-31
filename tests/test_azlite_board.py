import numpy as np

from azlite.board import (
    apply_move,
    board_to_bitboards,
    board_to_current_player_tensor,
    check_win,
    find_immediate_block,
    find_immediate_win,
    get_winner,
    is_draw,
    is_legal_move,
    legal_moves,
    numpy_to_obs_board,
    obs_board_to_numpy,
    ordered_legal_moves,
    terminal_value,
)


def _empty_board():
    return np.zeros((6, 7), dtype=np.int8)


def test_legal_moves_detection():
    board = _empty_board()
    assert legal_moves(board) == [0, 1, 2, 3, 4, 5, 6]


def test_full_column_detection():
    board = _empty_board()
    for i in range(6):
        board = apply_move(board, 3, 1 if i % 2 == 0 else 2)
    assert 3 not in legal_moves(board)
    assert is_legal_move(board, 3) is False
    assert is_legal_move(board, 2) is True


def test_apply_move_lowest_open_row():
    board = _empty_board()
    board = apply_move(board, 4, 1)
    assert board[5, 4] == 1
    board = apply_move(board, 4, 2)
    assert board[4, 4] == 2


def test_horizontal_four_in_a_row():
    board = _empty_board()
    board[5, 0:4] = 1
    assert check_win(board, 1) is True


def test_vertical_four_in_a_row():
    board = _empty_board()
    board[2:6, 5] = 2
    assert check_win(board, 2) is True


def test_main_diagonal_four_in_a_row():
    board = _empty_board()
    # (2,0) -> (5,3)  "\"
    board[2, 0] = 1
    board[3, 1] = 1
    board[4, 2] = 1
    board[5, 3] = 1
    assert check_win(board, 1) is True


def test_anti_diagonal_four_in_a_row():
    board = _empty_board()
    # (5,0) -> (2,3)  "/"
    board[5, 0] = 2
    board[4, 1] = 2
    board[3, 2] = 2
    board[2, 3] = 2
    assert check_win(board, 2) is True


def test_draw_detection():
    board = np.array(
        [
            [1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1],
            [1, 1, 2, 2, 1, 1, 2],
            [2, 2, 1, 1, 2, 2, 1],
        ],
        dtype=np.int8,
    )
    assert is_draw(board) is True
    assert get_winner(board) is None
    assert terminal_value(board, current_player=1) == 0.0


def test_current_player_perspective_tensor():
    board = _empty_board()
    board[5, 3] = 1
    board[5, 2] = 2

    x = board_to_current_player_tensor(board, current_player=2)
    assert x.shape == (3, 6, 7)
    assert x[0, 5, 2] == 1.0  # current player stones
    assert x[1, 5, 3] == 1.0  # opponent stones
    assert x[2, 4, 2] == 1.0  # next legal drop in col 2
    assert x[2, 4, 3] == 1.0  # next legal drop in col 3
    assert int(x[2].sum()) == len(legal_moves(board))


def test_find_immediate_win():
    board = _empty_board()
    board[5, 0] = 1
    board[5, 1] = 1
    board[5, 2] = 1
    assert find_immediate_win(board, player=1) == 3


def test_find_immediate_block():
    board = _empty_board()
    board[5, 0] = 2
    board[5, 1] = 2
    board[5, 2] = 2
    assert find_immediate_block(board, player=1, opponent=2) == 3


def test_center_first_ordering():
    board = _empty_board()
    assert ordered_legal_moves(board) == [3, 2, 4, 1, 5, 0, 6]


def test_terminal_value_current_player_perspective():
    board = _empty_board()
    board[5, 0:4] = 1
    assert terminal_value(board, current_player=1) == 1.0
    assert terminal_value(board, current_player=2) == -1.0
    assert terminal_value(_empty_board(), current_player=1) is None


def test_obs_list_and_bitboard_conversion():
    board = _empty_board()
    board[5, 3] = 1
    board[4, 3] = 2

    obs_list = numpy_to_obs_board(board)
    restored = obs_board_to_numpy(obs_list)
    assert np.array_equal(board, restored)

    b1, b2 = board_to_bitboards(board)
    assert int(b1) != 0
    assert int(b2) != 0
