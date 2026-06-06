import numpy as np

from azlite.teacher_data import DEFAULT_SOURCE_MIX, build_teacher_dataset, parse_source_mix


def test_teacher_smoke_independent_of_root_submission_file():
    states, policies, values, metadata = build_teacher_dataset(
        positions=200,
        teacher="auto",
        teacher_depth=2,
        policy_mode="soft",
        policy_temperature=1.0,
        value_mode="teacher",
        value_scale=0.0,
        source_mix=parse_source_mix(DEFAULT_SOURCE_MIX),
        include_legal_channel=True,
        seed=42,
        mcts_sims=8,
        mcts_time_ms=20.0,
        negamax_time_ms=40.0,
        rollout_policy="heuristic",
        max_state_repeats=1,
        source_sampling_mode="quota",
        max_source_stall_games=32,
        workers=2,
        main_cpu_threads=2,
        worker_cpu_threads=1,
    )

    assert states.shape == (200, 3, 6, 7)
    assert policies.shape == (200, 7)
    assert values.shape == (200,)
    assert np.allclose(np.sum(policies, axis=1), 1.0, atol=1e-5)
    assert metadata["teacher_type"] == "auto"
    assert metadata["teacher_backend"] == "strong"
    assert not metadata["teacher_fallback_reasons"]
