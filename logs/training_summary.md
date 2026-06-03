# Training Summary

- Updated at: `2026-06-02T08:33:32.921777+00:00`
- Best checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt` (iteration `0`)
- Latest checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt` (iteration `0`)
- Latest passed gating: `False`
- Recommended checkpoint: `best.pt`
- Recommended path: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt`

## Final Checkpoint Comparison
| Metric | best.pt | latest.pt |
|---|---|---|
| Label | final_best_eval | final_latest_eval |
| Iteration | 0 | 0 |
| Reliable | n/a | False |
| Gating passed | n/a | False |
| previous_best_available | n/a | True |
| side_bias_warning | n/a | True |
| unstable_previous_best_eval | n/a | True |
| vs random | n/a | 18/2/0 (90.0%) |
| vs negamax | n/a | 20/0/0 (100.0%) [UNRELIABLE, reliable=100.0%/20] |
| vs mcts_lite | n/a | 1/0/0 (100.0%) [UNRELIABLE, reliable=100.0%/1] |

## Strong Local Diagnostic
| Metric | best.pt | latest.pt |
|---|---|---|
| Label | final_best_eval_strong_local | final_latest_eval_strong_local |
| vs random | n/a | 18/2/0 (90.0%) |
| vs negamax | n/a | 20/0/0 (100.0%) [UNRELIABLE, reliable=100.0%/20] |
| vs mcts_lite | n/a | 1/0/0 (100.0%) [UNRELIABLE, reliable=100.0%/1] |

- Final comparison contains unreliable eval(s); those rows are not valid for strength ranking.

- side_bias_warning=true: previous_best matchup shows strong FP/SP asymmetry; treat comparison as unstable.
- unstable_previous_best_eval=true: require larger game count before promotion decisions.

- latest checkpoint did not pass gating; current best remains iteration 0.

## Recent 5 Evaluations
### Eval #1
- Time: `2026-06-01T04:13:04.101074+00:00`
- Mode: `selfplay_candidate`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt`
- Reliable: `True`
- Gating passed: `True`
- Reasons: all gating checks passed

### Eval #2
- Time: `2026-06-01T07:17:35.140286+00:00`
- Mode: `final_latest_eval`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`
- Reliable: `False`
- Gating passed: `False`
- Reasons: negamax: opponent_timeout_rate 0.980 > 0.050; win_rate excluded from gating; mcts_lite: opponent_timeout_rate 1.000 > 0.050; win_rate excluded from gating; negamax eval unreliable; cannot use for gating; mcts_lite eval unreliable; cannot use for gating; vs previous_best win rate below 55% (0.500 < 0.550); previous_best side bias too high (1.000 > 0.400); unstable, require more games; vs previous_best appears 50/50 by total WR but has strong first/second-player bias; negamax: opponent_timeout_rate_exceeded: 0.980 > 0.050; timeout_result_policy=fail_eval; mcts_lite: opponent_timeout_rate_exceeded: 1.000 > 0.050; timeout_result_policy=fail_eval; previous_best matchup unstable due severe first/second-player side bias; require more games.
- Unreliable reasons: negamax: opponent_timeout_rate_exceeded: 0.980 > 0.050; negamax: timeout_result_policy=fail_eval; mcts_lite: opponent_timeout_rate_exceeded: 1.000 > 0.050; mcts_lite: timeout_result_policy=fail_eval

### Eval #3
- Time: `2026-06-01T07:59:06.504195+00:00`
- Mode: `final_latest_eval`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`
- Reliable: `False`
- Gating passed: `False`
- Reasons: negamax: opponent_timeout_rate 1.000 > 0.050; win_rate excluded from gating; mcts_lite: opponent_timeout_rate 1.000 > 0.050; win_rate excluded from gating; negamax eval unreliable; cannot use for gating; mcts_lite eval unreliable; cannot use for gating; vs previous_best win rate below 55% (0.500 < 0.550); previous_best side bias too high (1.000 > 0.400); unstable, require more games; vs previous_best appears 50/50 by total WR but has strong first/second-player bias; negamax: opponent_timeout_rate_exceeded: 1.000 > 0.050; timeout_result_policy=fail_eval; mcts_lite: opponent_timeout_rate_exceeded: 1.000 > 0.050; timeout_result_policy=fail_eval; previous_best matchup unstable due severe first/second-player side bias; require more games.
- Unreliable reasons: negamax: opponent_timeout_rate_exceeded: 1.000 > 0.050; negamax: timeout_result_policy=fail_eval; mcts_lite: opponent_timeout_rate_exceeded: 1.000 > 0.050; mcts_lite: timeout_result_policy=fail_eval

### Eval #4
- Time: `2026-06-01T08:09:23.178681+00:00`
- Mode: `final_latest_eval_strong_local`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`
- Reliable: `False`
- Gating passed: `False`
- Reasons: negamax: opponent_timeout_rate 1.000 > 0.050; win_rate excluded from gating; mcts_lite: opponent_timeout_rate 1.000 > 0.050; win_rate excluded from gating; negamax eval unreliable; cannot use for gating; mcts_lite eval unreliable; cannot use for gating; vs previous_best win rate below 55% (0.500 < 0.550); previous_best side bias too high (1.000 > 0.400); unstable, require more games; vs previous_best appears 50/50 by total WR but has strong first/second-player bias; negamax: opponent_timeout_rate_exceeded: 1.000 > 0.050; timeout_result_policy=fail_eval; mcts_lite: opponent_timeout_rate_exceeded: 1.000 > 0.050; timeout_result_policy=fail_eval; previous_best matchup unstable due severe first/second-player side bias; require more games.
- Unreliable reasons: negamax: opponent_timeout_rate_exceeded: 1.000 > 0.050; negamax: timeout_result_policy=fail_eval; mcts_lite: opponent_timeout_rate_exceeded: 1.000 > 0.050; mcts_lite: timeout_result_policy=fail_eval

### Eval #5
- Time: `2026-06-01T19:30:28.052319+00:00`
- Mode: `bootstrap_pretrained`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_pretrained.pt`
- Reliable: `True`
- Gating passed: `True`
- Reasons: all gating checks passed

## Bottleneck
- pipeline final eval unreliable because opponent timeout rate is too high

## Next Suggestions
- `--simulations 80` reduce search load first to remove timeout/illegal.
- `--self-play-games 50` keep coverage stable while fixing reliability.