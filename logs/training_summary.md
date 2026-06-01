# Training Summary

- Updated at: `2026-06-01T04:13:04.225598+00:00`
- Current best checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt`
- Current latest checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\latest.pt`

## Recent 5 Evaluations
### Eval #1
- Time: `2026-05-31T19:48:03.851537+00:00`
- Mode: `bootstrap_pretrained`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_pretrained.pt`
- vs random: `191/8/1 (95.5%)`
- vs negamax: `174/26/0 (87.0%)`
- vs mcts_lite: `172/14/14 (86.0%)`
- vs previous best: `n/a`
- Gating passed: `True`
- Reasons: all gating checks passed

### Eval #2
- Time: `2026-06-01T04:13:04.101074+00:00`
- Mode: `selfplay_candidate`
- Checkpoint: `D:\205zd\Desktop\Kaggle\checkpoints\azlite_train\best.pt`
- vs random: `189/11/0 (94.5%)`
- vs negamax: `189/11/0 (94.5%)`
- vs mcts_lite: `184/15/1 (92.0%)`
- vs previous best: `n/a`
- Gating passed: `True`
- Reasons: all gating checks passed

## Bottleneck
- 暂无明显瓶颈，建议扩大量级继续训练。

## Next Suggestions
- `--iterations 25` 继续扩大迭代轮数。
- `--eval-games 200` 保持评估稳定性。
- `--simulations 100` 可先不变，观察趋势。