# Pipeline Evaluation

- Label: `final_latest_eval`
- Evaluator backend: `local_turn_engine_v2`
- Eval profile: `kaggle_like`
- Candidate timeout ms: `2000`
- Opponent timeout ms: `2000`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`
- Previous best: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\archived_best\previous_best_snapshot_iter0030.pt`
- previous_best_available: `True`
- side_bias_warning: `False`
- unstable_previous_best_eval: `False`

## Matrix
| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | TO_GAME(A/B) | TO_MOVE(A/B) | AvgStepMs(A) | P95StepMs(A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 30/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0/0 | 0/0 | 62.1 | 80.0 |
| negamax | 80/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0/0 | 0/0 | 116.7 | 391.8 |
| previous_best | 100/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0/0 | 0/0 | 299.3 | 471.3 |

## Gating
- Passed: `True`
- Reliable: `True`
- Composite: `1.0000`
- mcts_lite evaluation skipped/missing (diagnostic-only this run)