import numpy as np

from azlite.board import apply_move, legal_moves
from azlite.puct_mcts import (
    Evaluator,
    HeuristicEvaluator,
    MCTSNode,
    UniformEvaluator,
    _select_child,
    run_mcts,
)


def _empty_board():
    return np.zeros((6, 7), dtype=np.int8)


class ConstantEvaluator(Evaluator):
    def __init__(self, value: float = 0.5):
        self.value = float(value)

    def evaluate(self, board: np.ndarray, current_player: int):
        del current_player
        cols = board.shape[1]
        p = np.zeros(cols, dtype=np.float32)
        valid = legal_moves(board)
        if valid:
            p[valid] = 1.0 / len(valid)
        return p, self.value


def test_uniform_returns_legal_move():
    board = _empty_board()
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=32,
        use_tactical_shortcuts=False,
    )
    assert result["move"] in legal_moves(board)


def test_policy_target_shape():
    board = _empty_board()
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=16,
        use_tactical_shortcuts=False,
    )
    assert result["policy_target"].shape == (7,)
    assert result["visit_counts"].shape == (7,)


def test_illegal_column_visit_count_is_zero():
    board = _empty_board()
    for i in range(6):
        board = apply_move(board, 3, 1 if i % 2 == 0 else 2)

    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=24,
        use_tactical_shortcuts=False,
    )
    assert result["visit_counts"][3] == 0
    assert result["policy_target"][3] == 0


def test_full_column_not_selected():
    board = _empty_board()
    for i in range(6):
        board = apply_move(board, 6, 1 if i % 2 == 0 else 2)

    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=20,
        use_tactical_shortcuts=False,
    )
    assert result["move"] != 6


def test_immediate_win_shortcut():
    board = _empty_board()
    board[5, 0] = 1
    board[5, 1] = 1
    board[5, 2] = 1
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=50,
        use_tactical_shortcuts=True,
    )
    assert result["move"] == 3


def test_immediate_block_shortcut():
    board = _empty_board()
    board[5, 0] = 2
    board[5, 1] = 2
    board[5, 2] = 2
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=50,
        use_tactical_shortcuts=True,
    )
    assert result["move"] == 3


def test_terminal_root_value_correct():
    board = _empty_board()
    board[5, 0:4] = 1
    result_self = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=8,
        use_tactical_shortcuts=False,
    )
    result_opp = run_mcts(
        board=board,
        current_player=2,
        evaluator=UniformEvaluator(),
        num_simulations=8,
        use_tactical_shortcuts=False,
    )
    assert abs(result_self["root_value"] - 1.0) < 1e-6
    assert abs(result_opp["root_value"] + 1.0) < 1e-6


def test_backprop_value_sign_flip():
    board = _empty_board()
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=ConstantEvaluator(0.5),
        num_simulations=2,
        use_tactical_shortcuts=False,
        return_root=True,
    )
    root = result["root"]
    assert root is not None
    assert root.visit_count == 2
    # sim1 contributes +0.5 at root, sim2 visits child then root gets -0.5
    assert abs(root.value_sum) < 1e-6
    visited_children = [ch for ch in root.children.values() if ch.visit_count > 0]
    assert len(visited_children) >= 1
    assert visited_children[0].value_sum > 0


def test_selection_uses_parent_perspective_for_child_q():
    board = _empty_board()
    root = MCTSNode(board=board, current_player=1, visit_count=16, is_expanded=True)

    # Child current_player is the opponent of root.current_player.
    # Negative child.q_value means the child player dislikes the position,
    # so the root player should prefer it.
    good_for_root = MCTSNode(board=board.copy(), current_player=2, parent=root, prior=0.5)
    good_for_root.visit_count = 8
    good_for_root.value_sum = -6.0  # q = -0.75 from child/opponent perspective

    bad_for_root = MCTSNode(board=board.copy(), current_player=2, parent=root, prior=0.5)
    bad_for_root.visit_count = 8
    bad_for_root.value_sum = 6.0  # q = +0.75 from child/opponent perspective

    root.children[3] = good_for_root
    root.children[2] = bad_for_root

    action, child = _select_child(root, c_puct=0.0)
    assert action == 3
    assert child is good_for_root


def test_root_visit_count_matches_simulations():
    board = _empty_board()
    num_sim = 30
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=num_sim,
        use_tactical_shortcuts=False,
        return_root=True,
    )
    assert result["root"].visit_count == num_sim


def test_temperature_zero_picks_max_visit():
    board = _empty_board()
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=HeuristicEvaluator(),
        num_simulations=40,
        temperature=0.0,
        use_tactical_shortcuts=False,
    )
    counts = result["visit_counts"]
    assert counts[result["move"]] == np.max(counts)
    assert abs(float(result["policy_target"].sum()) - 1.0) < 1e-6


def test_temperature_one_returns_normalized_distribution():
    board = _empty_board()
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=UniformEvaluator(),
        num_simulations=40,
        temperature=1.0,
        use_tactical_shortcuts=False,
    )
    pi = result["policy_target"]
    assert pi.shape == (7,)
    assert np.all(pi >= 0)
    assert abs(float(pi.sum()) - 1.0) < 1e-6
