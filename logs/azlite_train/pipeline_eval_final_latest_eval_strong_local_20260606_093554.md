# Pipeline Evaluation

- Label: `final_latest_eval_strong_local`
- Evaluator backend: `local_turn_engine_v2`
- Eval profile: `strong_local`
- Candidate timeout ms: `2000`
- Opponent timeout ms: `5000`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`
- Previous best: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\archived_best\previous_best_snapshot_iter0030.pt`
- previous_best_available: `True`
- side_bias_warning: `False`
- unstable_previous_best_eval: `False`

## Matrix
| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | TO_GAME(A/B) | TO_MOVE(A/B) | AvgStepMs(A) | P95StepMs(A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 30/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0/0 | 0/0 | 361.2 | 488.9 |
| negamax | 77/3/0 | 96.2% | 97.5% | 95.0% | 0 | 0/0 | 0/0 | 364.1 | 491.3 |
| previous_best | 100/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0/0 | 0/0 | 353.6 | 489.7 |

## Gating
- Passed: `False`
- Reliable: `True`
- Composite: `0.9775`
- vs negamax win rate lower than previous best (0.963 < 1.000)
- mcts_lite evaluation skipped/missing (diagnostic-only this run)

## Note
- strong_local is diagnostic only and should not be treated as Kaggle-equivalent.