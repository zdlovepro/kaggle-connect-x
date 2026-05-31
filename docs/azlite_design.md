# AlphaZero-lite Design (Current Branch)

## Scope and Honesty

This document describes the **current implementation direction** on `feature/mcts-lite`.
It is not a claim that AlphaZero-lite has already fully surpassed the legacy MCTS-lite baseline.

## High-Level Architecture

- Board/rules foundation: `azlite/board.py`
- Search: `azlite/puct_mcts.py`
- Model: `azlite/model.py`
- Data generation:
  - teacher bootstrap: `azlite/teacher_data.py`
  - self-play: `azlite/self_play.py`
- Training loop: `azlite/train.py`
- Pipeline orchestrator: `azlite/pipeline.py`

## State Representation

- Input tensor is always **current-player perspective**:
  - channel 0: current player's stones
  - channel 1: opponent stones
  - channel 2 (optional): legal-drop plane
- Value target is also from current-player perspective.

## Network

Current default network is lightweight Conv + FC:

- Conv( C -> 32, 3x3 )
- Conv( 32 -> 64, 3x3 )
- Conv( 64 -> 64, 3x3 )
- FC( 64*6*7 -> 128 )
- Policy head: FC(128 -> 7)
- Value head: FC(128 -> 1) + tanh

## Search (PUCT)

- Selection score:
  - `Q + c_puct * P * sqrt(parent_N) / (1 + child_N)`
- Evaluator can be swapped:
  - `UniformEvaluator`
  - `HeuristicEvaluator`
  - `NeuralEvaluator`
- Tactical shortcuts can be on/off for controlled ablation.

## Training Targets

- Policy target (`pi`): root visit distribution from MCTS.
- Value target (`z`): terminal game result, backfilled per sample perspective.

## Why Keep Legacy MCTS-lite

- Legacy line is the baseline for strength and regression checks.
- AlphaZero-lite line is new and still being validated.
- Evaluation must prove gains against stronger opponents, not only random.

