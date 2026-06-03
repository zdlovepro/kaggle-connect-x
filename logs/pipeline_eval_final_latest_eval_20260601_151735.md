# Pipeline Evaluation

- Label: `final_latest_eval`
- Evaluator backend: `local_turn_engine_v2`
- Eval profile: `kaggle_like`
- Candidate timeout ms: `2000`
- Opponent timeout ms: `2000`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`
- Previous best: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\archived_best\previous_best_snapshot_iter0002.pt`
- previous_best_available: `True`
- side_bias_warning: `True`
- unstable_previous_best_eval: `True`

## Matrix
| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | Timeout(A) | Timeout(B) | AvgStepMs(A) | P95StepMs(A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 19/1/0 | 95.0% | 100.0% | 90.0% | 0 | 0 | 0 | 107.8 | 228.9 |
| negamax | 49/1/0 | 98.0% | 100.0% | 96.0% | 0 | 0 | 49 | 103.8 | 135.4 |
| mcts_lite | 50/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0 | 50 | 108.7 | 133.4 |
| previous_best | 25/25/0 | 50.0% | 100.0% | 0.0% | 0 | 0 | 0 | 84.6 | 142.4 |

## Gating
- Passed: `False`
- Reliable: `False`
- Composite: `0.5482`
- negamax: opponent_timeout_rate 0.980 > 0.050; win_rate excluded from gating
- mcts_lite: opponent_timeout_rate 1.000 > 0.050; win_rate excluded from gating
- negamax eval unreliable; cannot use for gating
- mcts_lite eval unreliable; cannot use for gating
- vs previous_best win rate below 55% (0.500 < 0.550)
- previous_best side bias too high (1.000 > 0.400); unstable, require more games
- vs previous_best appears 50/50 by total WR but has strong first/second-player bias
- negamax: opponent_timeout_rate_exceeded: 0.980 > 0.050; timeout_result_policy=fail_eval
- mcts_lite: opponent_timeout_rate_exceeded: 1.000 > 0.050; timeout_result_policy=fail_eval
- previous_best matchup unstable due severe first/second-player side bias; require more games.

## Reliability Warnings
- [eval][warning] opponent=negamax timeout_rate=98.0%, win_rate is unreliable and excluded from gating.
- [eval][warning] opponent=mcts_lite timeout_rate=100.0%, win_rate is unreliable and excluded from gating.

## Side Bias Warning
- previous_best side bias high (1.000)

## Unreliable Reasons
- negamax: opponent_timeout_rate_exceeded: 0.980 > 0.050
- negamax: timeout_result_policy=fail_eval
- mcts_lite: opponent_timeout_rate_exceeded: 1.000 > 0.050
- mcts_lite: timeout_result_policy=fail_eval