# Evaluation and Gating

## Why Evaluation Is Strict

Random-only metrics are insufficient. A model can overfit weak play and still lose badly to stronger search agents.

## Primary Evaluation Opponents

- `random` (sanity only)
- `negamax`
- `mcts_lite` / `original`
- `previous best azlite`

## Main Tools

- `python -m azlite.evaluate ...`
- `python -m azlite.ablation ...`

## Example Commands

Standard checkpoint evaluation:

```bash
python -m azlite.evaluate --checkpoint checkpoints/best.pt --opponents random,negamax,mcts_lite --games 200
```

Ablation:

```bash
python -m azlite.ablation \
  --teacher-checkpoint checkpoints/azlite_pretrained.pt \
  --selfplay-checkpoint checkpoints/azlite_train/best.pt \
  --opponents random,negamax,original,mcts_lite \
  --games 200
```

## Metrics to Track

- W/L/D
- win rate
- first-player win rate
- second-player win rate
- illegal actions
- average step time
- p95 step time
- timeout count

## Promotion / Gating Principles

- Candidate must not regress materially vs `negamax`
- Candidate should be competitive vs `previous best`
- Illegal actions must be zero
- Timeout profile must be submission-safe
- Random score is not a promotion criterion

## Required Honesty

- If AlphaZero-lite is weaker than original MCTS-lite, report it explicitly.
- Do not hide failed runs or filtered opponents.
- Keep full logs for postmortem.

