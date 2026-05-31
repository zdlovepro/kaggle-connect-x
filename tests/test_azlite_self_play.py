import numpy as np
import pytest
import tempfile

torch = pytest.importorskip("torch")

from azlite.board import legal_moves
from azlite.model import ConnectXNet
from azlite.self_play import augment_mirror, generate_self_play_games, play_self_play_game


def test_can_generate_one_complete_self_play_game():
    model = ConnectXNet(in_channels=3)
    examples = play_self_play_game(
        model=model,
        num_simulations=12,
        device="cpu",
        c_puct=1.5,
        add_dirichlet_noise=True,
        use_tactical_shortcuts=False,
        max_moves=42,
    )
    assert len(examples) > 0
    assert len(examples) <= 42


def test_policy_target_shape_and_legal_distribution():
    model = ConnectXNet(in_channels=3)
    examples = play_self_play_game(
        model=model,
        num_simulations=10,
        device="cpu",
        add_dirichlet_noise=True,
        use_tactical_shortcuts=False,
        max_moves=42,
    )
    assert examples
    for ex in examples:
        assert ex.policy_target.shape == (7,)
        assert ex.visit_counts.shape == (7,)
        legal = legal_moves(ex.board_before_move)
        assert np.isclose(float(ex.policy_target[list(legal)].sum()), 1.0, atol=1e-6)
        for c in range(7):
            if c not in legal:
                assert ex.policy_target[c] == 0.0


def test_value_targets_are_only_minus1_zero_plus1():
    model = ConnectXNet(in_channels=3)
    examples = play_self_play_game(
        model=model,
        num_simulations=10,
        device="cpu",
        add_dirichlet_noise=False,
        use_tactical_shortcuts=False,
        max_moves=42,
    )
    assert examples
    values = {float(ex.value_target) for ex in examples}
    assert values.issubset({-1.0, 0.0, 1.0})


def test_mirror_augmentation_reverses_policy_and_visits():
    model = ConnectXNet(in_channels=3)
    examples = play_self_play_game(
        model=model,
        num_simulations=8,
        device="cpu",
        add_dirichlet_noise=False,
        use_tactical_shortcuts=False,
        max_moves=42,
    )
    ex = examples[0]
    mirrored = augment_mirror(ex)
    assert np.allclose(mirrored.policy_target, ex.policy_target[::-1], atol=1e-6)
    assert np.allclose(mirrored.visit_counts, ex.visit_counts[::-1], atol=1e-6)
    assert mirrored.selected_move == 6 - ex.selected_move
    assert mirrored.value_target == ex.value_target


def test_current_player_perspective_encoding_is_correct():
    model = ConnectXNet(in_channels=3)
    examples = play_self_play_game(
        model=model,
        num_simulations=10,
        device="cpu",
        add_dirichlet_noise=False,
        use_tactical_shortcuts=False,
        max_moves=42,
    )
    assert examples
    # Pick a non-empty position if possible, otherwise first one.
    ex = next((e for e in examples if int(np.count_nonzero(e.board_before_move)) > 0), examples[0])
    board = ex.board_before_move
    cur = ex.current_player
    opp = 2 if cur == 1 else 1
    assert ex.state.shape[1:] == (6, 7)
    assert np.array_equal(ex.state[0], (board == cur).astype(np.float32))
    assert np.array_equal(ex.state[1], (board == opp).astype(np.float32))


def test_generate_self_play_games_saves_npz():
    model = ConnectXNet(in_channels=3)
    with tempfile.TemporaryDirectory(prefix="azlite_selfplay_") as tmp:
        out_path = generate_self_play_games(
            model=model,
            num_games=1,
            num_simulations=8,
            device="cpu",
            augment=True,
            output_dir=tmp,
        )
        data = np.load(str(out_path), allow_pickle=True)
        assert "states" in data
        assert "policies" in data
        assert "values" in data
        assert "metadata" in data
        assert data["states"].ndim == 4
        assert data["policies"].shape[1] == 7
        assert data["values"].ndim == 1
