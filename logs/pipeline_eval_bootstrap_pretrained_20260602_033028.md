# Pipeline Evaluation

- Label: `bootstrap_pretrained`
- Evaluator backend: `local_turn_engine_v2`
- Eval profile: `kaggle_like`
- Candidate timeout ms: `2000`
- Opponent timeout ms: `4000`
- Checkpoint: `checkpoints\azlite_pretrained.pt`
- Previous best: `None`
- previous_best_available: `False`
- side_bias_warning: `False`
- unstable_previous_best_eval: `False`

## Matrix
| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | TO_GAME(A/B) | TO_MOVE(A/B) | AvgStepMs(A) | P95StepMs(A) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| random | 20/0/0 | 100.0% | 100.0% | 100.0% | 0 | 0/0 | 0/0 | 109.3 | 169.7 |
| negamax | 44/56/0 | 44.0% | 72.0% | 16.0% | 0 | 0/0 | 0/0 | 94.6 | 161.2 |
| mcts_lite | 81/19/0 | 81.0% | 92.0% | 70.0% | 0 | 0/0 | 0/0 | 109.7 | 167.7 |

## Gating
- Passed: `True`
- Reliable: `True`
- Composite: `0.6104`
- all gating checks passed

## Warning
> random is not a reliable gating metric; model may be overfitting weak play or relying on tactical shortcuts only.