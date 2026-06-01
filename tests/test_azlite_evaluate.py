import time

from azlite.evaluate import (
    UnifiedAgent,
    _warn_random_overfit,
    play_match,
    should_promote_candidate,
)


def _legal_leftmost_agent() -> UnifiedAgent:
    def _fn(observation, configuration):
        board = observation.board
        for c in range(int(configuration.columns)):
            if board[c] == 0:
                return c
        return 0

    return UnifiedAgent(name="leftmost", fn=_fn)


def _always_illegal_agent() -> UnifiedAgent:
    def _fn(observation, configuration):
        del observation, configuration
        return 99

    return UnifiedAgent(name="illegal", fn=_fn)


def test_play_match_records_basic_metrics():
    a = _legal_leftmost_agent()
    b = _legal_leftmost_agent()
    result = play_match(a, b, num_games=4, swap_sides=True, seed=123)

    assert result["num_games"] == 4
    assert result["wins"] + result["losses"] + result["draws"] == 4
    assert result["illegal_actions"]["agent_a"] == 0
    assert result["timeouts"]["agent_a"] == 0
    assert result["errors"]["agent_a"] == 0
    assert result["step_count"]["agent_a"] > 0
    assert result["step_time_sum_sec"]["agent_a"] >= 0.0
    assert result["p95_step_time_sec"]["agent_a"] >= 0.0
    assert "first_player_win_rate" in result
    assert "second_player_win_rate" in result
    assert "candidate_timeouts" in result
    assert "opponent_timeouts" in result
    assert "candidate_timeout_rate" in result
    assert "opponent_timeout_rate" in result
    assert "reliable" in result
    assert "unreliable_reasons" in result
    assert "invalid_actions" in result


def test_play_match_detects_illegal_action():
    illegal = _always_illegal_agent()
    legal = _legal_leftmost_agent()
    result = play_match(illegal, legal, num_games=2, swap_sides=True, seed=7)
    assert result["illegal_actions"]["agent_a"] > 0
    assert result["losses"] >= 1


def test_warn_random_overfit_message():
    candidate_results = {
        "random": {"win_rate": 0.90},
        "negamax": {"win_rate": 0.50},
    }
    warning = _warn_random_overfit(candidate_results)
    assert warning is not None
    assert "random is not a reliable gating metric" in warning


def test_play_match_flags_unreliable_when_opponent_times_out():
    legal = _legal_leftmost_agent()

    def _slow_fn(observation, configuration):
        del observation, configuration
        time.sleep(0.01)
        return 0

    slow = UnifiedAgent(name="slow", fn=_slow_fn)
    result = play_match(
        legal,
        slow,
        num_games=2,
        swap_sides=True,
        seed=11,
        act_timeout_sec=0.001,
        max_opponent_timeout_rate=0.10,
        max_candidate_timeout_rate=0.10,
        timeout_result_policy="fail_eval",
    )
    assert result["opponent_timeouts"] >= 1
    assert result["reliable"] is False
    assert result["opponent_timeout_rate"] > 0.10
    assert any("opponent_timeout_rate_exceeded" in x for x in result["unreliable_reasons"])


def test_should_promote_candidate_enforces_rules():
    candidate_results = {
        "random": {
            "win_rate": 0.99,
            "num_games": 10,
            "illegal_actions": {"agent_a": 0, "agent_b": 0},
            "timeouts": {"agent_a": 0, "agent_b": 0},
            "errors": {"agent_a": 0, "agent_b": 0},
            "avg_steps": 20.0,
            "step_count": {"agent_a": 200},
            "step_time_sum_sec": {"agent_a": 20.0},
            "p95_step_time_sec": {"agent_a": 0.20},
            "reliable": True,
        },
        "negamax": {
            "win_rate": 0.60,
            "num_games": 10,
            "illegal_actions": {"agent_a": 0, "agent_b": 0},
            "timeouts": {"agent_a": 0, "agent_b": 0},
            "errors": {"agent_a": 0, "agent_b": 0},
            "avg_steps": 20.0,
            "step_count": {"agent_a": 200},
            "step_time_sum_sec": {"agent_a": 20.0},
            "p95_step_time_sec": {"agent_a": 0.25},
            "reliable": True,
        },
        "previous_best": {
            "win_rate": 0.56,
            "num_games": 10,
            "illegal_actions": {"agent_a": 0, "agent_b": 0},
            "timeouts": {"agent_a": 0, "agent_b": 0},
            "errors": {"agent_a": 0, "agent_b": 0},
            "avg_steps": 22.0,
            "step_count": {"agent_a": 220},
            "step_time_sum_sec": {"agent_a": 22.0},
            "p95_step_time_sec": {"agent_a": 0.30},
            "reliable": True,
        },
    }
    previous_best_results = {
        "negamax": {
            "win_rate": 0.58,
            "num_games": 10,
            "illegal_actions": {"agent_a": 0, "agent_b": 0},
            "timeouts": {"agent_a": 0, "agent_b": 0},
            "errors": {"agent_a": 0, "agent_b": 0},
            "avg_steps": 20.0,
            "step_count": {"agent_a": 200},
            "step_time_sum_sec": {"agent_a": 20.0},
            "p95_step_time_sec": {"agent_a": 0.20},
        }
    }
    passed, reasons = should_promote_candidate(candidate_results, previous_best_results)
    assert passed is True
    assert any("all gating checks passed" in r for r in reasons)


def test_should_promote_candidate_rejects_unreliable_timeout_matchup():
    candidate_results = {
        "negamax": {
            "win_rate": 0.95,
            "num_games": 10,
            "illegal_actions": {"agent_a": 0, "agent_b": 0},
            "timeouts": {"agent_a": 0, "agent_b": 0},
            "errors": {"agent_a": 0, "agent_b": 0},
            "avg_steps": 20.0,
            "step_count": {"agent_a": 200},
            "step_time_sum_sec": {"agent_a": 20.0},
            "p95_step_time_sec": {"agent_a": 0.25},
            "reliable": False,
            "unreliable_reasons": ["opponent_timeout_rate_exceeded: 0.800 > 0.050"],
        }
    }
    passed, reasons = should_promote_candidate(candidate_results, previous_best_results=None)
    assert passed is False
    assert any("opponent_timeout_rate_exceeded" in r for r in reasons)
