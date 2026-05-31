# Training Pipeline

## Goals

- Reproducible training flow
- Resume after interruption
- Honest gating (no random-only promotion)

## Directory Layout

- `data/teacher/`
- `data/selfplay/`
- `checkpoints/`
- `logs/`

## Modes

`azlite.pipeline` supports:

1. `bootstrap`
- Generate teacher data
- Supervised pretrain
- Optional evaluation

2. `selfplay`
- Resume or start from checkpoint
- Generate self-play data
- Train
- Evaluate and gate

3. `full`
- If no pretrained checkpoint: run bootstrap first
- Then run selfplay

## Typical Commands

Bootstrap only:

```bash
python -m azlite.pipeline --mode bootstrap --teacher-positions 50000 --teacher-depth 5
```

Self-play only (resume):

```bash
python -m azlite.pipeline --mode selfplay --resume --iterations 20 --self-play-games 50 --simulations 100
```

Full pipeline:

```bash
python -m azlite.pipeline --mode full --iterations 20 --self-play-games 50 --simulations 100 --eval-games 200
```

## Resume Behavior

- Model checkpoint recovery from `latest.pt`
- Replay buffer recovery from latest snapshot
- Pipeline state from `checkpoints/pipeline_state.json`

## Failure Safety

- NaN loss: training stops and reports error
- Evaluation failure: checkpoint is kept
- Gating fail: candidate does not overwrite best
- Submission build fail: does not invalidate checkpoints

## Metadata Requirements

Checkpoint metadata should include:

- branch name
- git commit hash
- iteration
- teacher data path
- self-play games
- simulations
- train steps
- eval results
- config

