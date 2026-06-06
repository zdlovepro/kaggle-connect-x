# Pipeline Evaluation

- Label: `final_best_eval_strong_local`
- Evaluator backend: `local_turn_engine_v2`
- Eval profile: `strong_local`
- Candidate timeout ms: `2000`
- Opponent timeout ms: `5000`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt`
- Previous best: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\archived_best\previous_best_snapshot_iter0030.pt`
- previous_best_available: `True`
- side_bias_warning: `True`
- unstable_previous_best_eval: `True`

## Matrix
| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | TO_GAME(A/B) | TO_MOVE(A/B) | AvgStepMs(A) | P95StepMs(A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 30/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0/0 | 0/0 | 361.5 | 485.7 |
| negamax | 78/0/2 | 97.5% | 100.0% | 95.0% | 0 | 0/0 | 0/0 | 323.0 | 483.9 |
| previous_best | 50/50/0 | 50.0% | 0.0% | 100.0% | 0 | 0/0 | 0/0 | 212.3 | 368.9 |

## Gating
- Passed: `False`
- Reliable: `True`
- Composite: `0.8064`
- mcts_lite evaluation skipped/missing (diagnostic-only this run)
- vs previous_best win rate below 55% (0.500 < 0.550)
- vs previous_best appears 50/50 by total WR but has strong first/second-player bias; ignored due extreme side bias
- previous_best matchup unstable due severe first/second-player side bias; require more games.

## Side Bias Warning
- previous_best side bias high (1.000)

## Note
- strong_local is diagnostic only and should not be treated as Kaggle-equivalent.