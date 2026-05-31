# Troubleshooting

## 1) Loss becomes NaN

Symptoms:
- training stops with NaN loss

Checks:
- policy targets contain NaN
- values contain NaN
- illegal columns in target policy
- learning rate too high

Actions:
- lower `--lr`
- inspect latest self-play/teacher `.npz`
- enforce target validity checks before optimizer step

## 2) Value saturates to all +1 / -1

Symptoms:
- value predictions almost always `+1` or `-1`

Checks:
- replay buffer diversity
- class/outcome imbalance
- too aggressive optimization

Actions:
- reduce lr
- increase data diversity (more self-play games, stronger opponents)
- verify value target perspective consistency

## 3) Policy mass on illegal columns

Symptoms:
- target or predicted policy assigns probability to full columns

Checks:
- legal mask application in data generation and inference
- top-row occupancy logic

Actions:
- hard-mask illegal moves and renormalize
- add assertion in training loop and self-play export

## 4) MCTS always picks center column

Symptoms:
- center move dominates almost every state

Checks:
- priors too flat / bad
- `c_puct` too low or too high
- tactical shortcut over-trigger
- simulation count too low

Actions:
- ablate `c_puct` and simulations
- compare with tactical on/off
- inspect visit distribution at root

## 5) Random high but negamax low

Symptoms:
- random win rate looks strong, negamax weak

Interpretation:
- likely overfitting weak play or shortcut artifacts

Actions:
- treat as non-promotion result
- tune with negamax/previous-best as primary gate
- run ablation to isolate misleading components

## 6) Submission timeout

Symptoms:
- per-step runtime exceeds Kaggle constraints

Checks:
- simulation budget too high
- no early stop in search loop
- heavy inference path

Actions:
- lower simulation caps
- tighten time budget and reserve buffer
- ensure fallback is fast and legal

## 7) Checkpoint agent and submission agent disagree

Symptoms:
- same state gives very different policy/value

Checks:
- export precision/dtype mismatch
- tensor ordering mismatch
- missing mask or perspective mismatch
- numpy vs torch forward differences

Actions:
- run `build_azlite_submission.py --validate`
- compare policy L1, value diff, argmax match rate

## 8) Self-play policy_target all zeros

Symptoms:
- `policy_target` row sum is zero

Checks:
- root visits collected correctly
- legal move set empty unexpectedly
- normalization path broken

Actions:
- force fallback to uniform over legal moves when visit sum is zero
- assert non-zero row sums before saving

## 9) Value sign/perspective error

Symptoms:
- training appears unstable; value contradicts terminal result

Checks:
- backprop sign flip per ply in MCTS
- `terminal_value(board, current_player)` convention
- value backfill in self-play examples

Actions:
- verify every ply flips sign on backprop
- verify winner-to-sample mapping:
  - sample player == winner -> `+1`
  - sample player != winner -> `-1`
  - draw -> `0`

