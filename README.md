# Kaggle ConnectX (feature/mcts-lite)

This branch is actively upgrading from the original MCTS-lite line to an AlphaZero-lite line.

## Project Status

- `feature/mcts-lite` is a mixed development branch: **MCTS-lite baseline + AlphaZero-lite experiments**.
- The original MCTS-lite / handcrafted search agent is still the baseline reference.
- AlphaZero-lite is the new experimental mainline for neural policy-value + PUCT + self-play.
- We do **not** assume AlphaZero-lite is always stronger yet. All promotion should pass strict evaluation gates.

## Current Agents

- `random`
- `negamax`
- `original mcts_lite` (legacy/handcrafted baseline)
- `azlite policy-only`
- `azlite PUCT MCTS` (neural evaluator + search)

## Why Not Only Random Win Rate

- Random win rate is only a sanity check.
- Primary decision metrics are performance vs:
  - `negamax`
  - `previous best`
  - `mcts_lite` / `original`
- If random is high but negamax is low, treat it as a warning signal, not progress.

## AlphaZero-lite Design

- Board encoding uses **current-player perspective tensor**.
- Network outputs:
  - `policy head`: action logits/probabilities over 7 columns
  - `value head`: scalar in `[-1, 1]` from current-player perspective
- Search uses **PUCT MCTS**.
- Self-play policy target is **root visit distribution** (not raw network policy).
- Value target is **final game outcome** (`z`), mapped to each sample's player perspective.

## Training Commands

Generate teacher data:

```bash
python -m azlite.teacher_data --positions 50000 --teacher negamax --teacher-depth 5
```

Pretrain:

```bash
python -m azlite.pretrain --data data/teacher/teacher_depth5.npz --epochs 10
```

Self-play training:

```bash
python -m azlite.train --checkpoint checkpoints/azlite_pretrained.pt --iterations 20 --self-play-games 50 --simulations 100
```

Full pipeline:

```bash
python -m azlite.pipeline --mode full
```

Evaluation:

```bash
python -m azlite.evaluate --checkpoint checkpoints/best.pt --opponents random,negamax,mcts_lite --games 200
```

Build submission:

```bash
python build_azlite_submission.py --checkpoint checkpoints/best.pt --output submission.py --mode numpy
```

## Repository Docs

- Legacy/handcrafted algorithm notes: `docs/ALGORITHM.md` (kept intact as historical baseline docs)
- AlphaZero-lite design: `docs/azlite_design.md`
- Training pipeline: `docs/training_pipeline.md`
- Evaluation and gating: `docs/evaluation.md`
- Submission export/build: `docs/submission_build.md`
- Troubleshooting: `docs/troubleshooting.md`

## Troubleshooting (Quick Index)

See `docs/troubleshooting.md` for detailed fixes, including:

- `loss NaN`
- value head saturation (`all +1/-1`)
- policy mass on illegal moves
- MCTS repeatedly picking center only
- random high but negamax low
- submission timeout
- checkpoint agent vs submission agent mismatch
- self-play `policy_target` all-zero
- value sign/perspective errors

