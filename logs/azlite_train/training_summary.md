# Training Summary

- Updated at: `2026-06-06T03:31:10.289057+00:00`
- Best checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt` (iteration `3`)
- Latest checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt` (iteration `30`)
- Latest passed gating: `True`
- Recommended checkpoint: `latest.pt`
- Recommended path: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`

## Final Checkpoint Comparison
| Metric | best.pt | latest.pt |
|---|---|---|
| Label | final_best_eval | final_latest_eval |
| Iteration | 3 | 30 |
| Reliable | True | True |
| Gating passed | False | True |
| previous_best_available | True | True |
| side_bias_warning | True | False |
| unstable_previous_best_eval | True | False |
| vs random | 30/0/0 (100.0%) | 30/0/0 (100.0%) |
| vs negamax | 79/0/1 (98.8%) | 80/0/0 (100.0%) |
| vs mcts_lite | n/a | n/a |

## Strong Local Diagnostic
| Metric | best.pt | latest.pt |
|---|---|---|
| Label | final_best_eval_strong_local | final_latest_eval_strong_local |
| vs random | 30/0/0 (100.0%) | 30/0/0 (100.0%) |
| vs negamax | 78/0/2 (97.5%) | 77/3/0 (96.2%) |
| vs mcts_lite | n/a | n/a |

- side_bias_warning=true: previous_best matchup shows strong FP/SP asymmetry; treat comparison as unstable.
- unstable_previous_best_eval=true: require larger game count before promotion decisions.

## Recent 5 Evaluations
### Eval #1
- Time: `2026-06-05T17:34:51.719080+00:00`
- Mode: `bootstrap_pretrained`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_pretrained.pt`
- Reliable: `True`
- Gating passed: `True`
- Reasons: mcts_lite evaluation skipped/missing (diagnostic-only this run)

### Eval #2
- Time: `2026-06-06T00:05:49.598595+00:00`
- Mode: `final_latest_eval`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`
- Reliable: `True`
- Gating passed: `True`
- Reasons: mcts_lite evaluation skipped/missing (diagnostic-only this run)

### Eval #3
- Time: `2026-06-06T01:35:54.905003+00:00`
- Mode: `final_latest_eval_strong_local`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`
- Reliable: `True`
- Gating passed: `False`
- Reasons: vs negamax win rate lower than previous best (0.963 < 1.000); mcts_lite evaluation skipped/missing (diagnostic-only this run)

### Eval #4
- Time: `2026-06-06T02:24:35.283255+00:00`
- Mode: `final_best_eval`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt`
- Reliable: `True`
- Gating passed: `False`
- Reasons: mcts_lite evaluation skipped/missing (diagnostic-only this run); vs previous_best win rate below 55% (0.500 < 0.550); vs previous_best appears 50/50 by total WR but has strong first/second-player bias; ignored due extreme side bias; previous_best matchup unstable due severe first/second-player side bias; require more games.

### Eval #5
- Time: `2026-06-06T03:31:10.152180+00:00`
- Mode: `final_best_eval_strong_local`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt`
- Reliable: `True`
- Gating passed: `False`
- Reasons: mcts_lite evaluation skipped/missing (diagnostic-only this run); vs previous_best win rate below 55% (0.500 < 0.550); vs previous_best appears 50/50 by total WR but has strong first/second-player bias; ignored due extreme side bias; previous_best matchup unstable due severe first/second-player side bias; require more games.

## Bottleneck
- random is saturated; progress should be measured against negamax/mcts_lite/previous_best.

## Next Suggestions
- `--simulations 116` raise search quality for stronger targets.
- `--self-play-games 96` increase self-play coverage.
- `--teacher-depth 6` strengthen teacher bootstrap.