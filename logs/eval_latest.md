# Evaluation Report

- Timestamp: `2026-06-02T00:33:33.404043`
- Evaluator backend: `local_turn_engine_v2`
- Candidate: `random`
- Checkpoint: `None`
- Games per matchup: `0`
- Eval profile: `quick`
- Candidate timeout ms: `2000`
- Opponent timeout ms: `4000`
- Simulations: `100`
- previous_best_available: `False`
- side_bias_warning: `False`
- unstable_previous_best_eval: `False`
- skipped_opponents: `['mcts_lite', 'negamax', 'random']`

## Matrix
| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | TO_GAME(A/B) | TO_MOVE(A/B) | AvgSteps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## Gating
- Passed: `False`
- Composite: `unavailable`
- all key metrics (negamax/mcts_lite/previous_best) are unreliable or missing
- negamax evaluation skipped/missing (diagnostic-only this run)
- mcts_lite evaluation skipped/missing (diagnostic-only this run)

- JSON log: `logs\eval_20260602_003333.json`