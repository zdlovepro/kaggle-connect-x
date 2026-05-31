# Submission Build (AlphaZero-lite)

## Objective

Export trained neural checkpoint into a **single-file, self-contained** Kaggle `submission.py`.

## Components

- Export: `azlite/export_model.py`
- Submission runtime template: `azlite/submission_agent.py`
- Builder: `build_azlite_submission.py`

## Modes

### Mode A: torch runtime

- Simpler runtime path when torch is available.
- Heavier dependency profile.

### Mode B: numpy runtime (default recommendation)

- No torch dependency in final submission file.
- Uses embedded weights + numpy forward + PUCT.
- Better for Kaggle single-file portability.

## Build Command

```bash
python build_azlite_submission.py --checkpoint checkpoints/best.pt --output submission.py --mode numpy
```

## Validation Command

```bash
python build_azlite_submission.py --checkpoint checkpoints/best.pt --output submission.py --mode numpy --validate
```

Validation includes:

- checkpoint agent vs built submission agent policy/value alignment
- matchup stats vs random / negamax / original
- W/L/D, timing, p95, illegal moves

## Runtime Safety in Submission

- Immediate win and block checks first
- Time-budgeted MCTS loop with early stop
- Legal fallback move always available
- No noisy debug logging

## Important Constraint

Final Kaggle upload must be a single `submission.py` with no local package import dependency for runtime assets.

