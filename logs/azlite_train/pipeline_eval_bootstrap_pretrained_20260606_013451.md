# Pipeline Evaluation

- Label: `bootstrap_pretrained`
- Evaluator backend: `local_turn_engine_v2`
- Eval profile: `kaggle_like`
- Candidate timeout ms: `2000`
- Opponent timeout ms: `2000`
- Checkpoint: `checkpoints\azlite_pretrained.pt`
- Previous best: `None`
- previous_best_available: `False`
- side_bias_warning: `False`
- unstable_previous_best_eval: `False`

## Matrix
| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | TO_GAME(A/B) | TO_MOVE(A/B) | AvgStepMs(A) | P95StepMs(A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 30/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0/0 | 0/0 | 53.6 | 81.7 |
| negamax | 54/24/2 | 67.5% | 72.5% | 62.5% | 0 | 0/0 | 0/0 | 53.2 | 81.0 |

## Gating
- Passed: `True`
- Reliable: `True`
- Composite: `0.6967`
- mcts_lite evaluation skipped/missing (diagnostic-only this run)