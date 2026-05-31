import tempfile

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from azlite.board import apply_move, legal_moves
from azlite.model import (
    ConnectXNet,
    NeuralEvaluator,
    load_checkpoint,
    predict_policy_value,
    save_checkpoint,
)
from azlite.puct_mcts import run_mcts


def _empty_board():
    return np.zeros((6, 7), dtype=np.int8)


def test_forward_shapes():
    model = ConnectXNet(in_channels=3)
    x = torch.randn(4, 3, 6, 7)
    policy_logits, value = model(x)
    assert policy_logits.shape == (4, 7)
    assert value.shape == (4, 1)


def test_value_range_in_forward():
    model = ConnectXNet(in_channels=3)
    x = torch.randn(8, 3, 6, 7)
    _, value = model(x)
    assert torch.all(value <= 1.0 + 1e-6)
    assert torch.all(value >= -1.0 - 1e-6)


def test_predict_policy_value_shape_and_mask():
    model = ConnectXNet(in_channels=3)
    board = _empty_board()
    # Fill column 3 completely
    for i in range(6):
        board = apply_move(board, 3, 1 if i % 2 == 0 else 2)

    policy_probs, value = predict_policy_value(model, board, current_player=1, device="cpu")
    assert policy_probs.shape == (7,)
    assert isinstance(value, float)
    assert -1.0 <= value <= 1.0
    assert policy_probs[3] == 0.0
    valid = legal_moves(board)
    assert np.isclose(policy_probs[valid].sum(), 1.0, atol=1e-6)


def test_neural_evaluator_with_run_mcts_returns_legal_move():
    model = ConnectXNet(in_channels=3)
    evaluator = NeuralEvaluator(model, device="cpu")
    board = _empty_board()
    result = run_mcts(
        board=board,
        current_player=1,
        evaluator=evaluator,
        num_simulations=20,
        use_tactical_shortcuts=False,
    )
    assert result["move"] in legal_moves(board)
    assert result["policy_target"].shape == (7,)


def test_checkpoint_save_load_predict_consistency():
    torch.manual_seed(123)
    model = ConnectXNet(in_channels=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    board = _empty_board()
    board = apply_move(board, 3, 1)
    board = apply_move(board, 2, 2)

    p1, v1 = predict_policy_value(model, board, current_player=1, device="cpu")

    with tempfile.TemporaryDirectory(prefix="azlite_model_ckpt_") as tmp:
        ckpt_path = f"{tmp}/model.pt"
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            path=ckpt_path,
            metadata={
                "iteration": 7,
                "train_steps": 128,
                "simulations": 4096,
                "eval_results": {"wr_vs_negamax": 0.55},
            },
        )
        loaded_model, metadata = load_checkpoint(ckpt_path, device="cpu")
        p2, v2 = predict_policy_value(loaded_model, board, current_player=1, device="cpu")

    assert np.allclose(p1, p2, atol=1e-7)
    assert abs(v1 - v2) <= 1e-7
    assert metadata["iteration"] == 7
    assert metadata["train_steps"] == 128
    assert metadata["simulations"] == 4096
    assert "eval_results" in metadata
    assert "created_at" in metadata
