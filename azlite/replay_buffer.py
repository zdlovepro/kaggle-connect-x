"""Replay buffer for AlphaZero-lite training."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


ROWS = 6
COLS = 7


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_metadata(raw) -> dict:
    if raw is None:
        return {}
    try:
        if isinstance(raw, np.ndarray):
            if raw.shape == ():
                raw = raw.item()
            elif raw.size == 1:
                raw = raw.reshape(-1)[0]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            return json.loads(raw)
    except Exception:
        return {}
    return {}


class ReplayBuffer:
    """Simple FIFO replay buffer with capped size."""

    def __init__(self, max_size: int = 200_000):
        if int(max_size) <= 0:
            raise ValueError(f"max_size must be > 0, got {max_size}")
        self.max_size = int(max_size)
        self._states: np.ndarray | None = None
        self._policies: np.ndarray | None = None
        self._values: np.ndarray | None = None

    def __len__(self) -> int:
        if self._states is None:
            return 0
        return int(self._states.shape[0])

    @property
    def state_shape(self) -> Tuple[int, int, int] | None:
        if self._states is None:
            return None
        return tuple(int(v) for v in self._states.shape[1:])

    def _validate_arrays(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.asarray(states, dtype=np.float32)
        p = np.asarray(policies, dtype=np.float32)
        v = np.asarray(values, dtype=np.float32)

        if s.ndim != 4:
            raise ValueError(f"states must be 4D (N,C,6,7), got shape={s.shape}")
        if s.shape[2:] != (ROWS, COLS):
            raise ValueError(f"states spatial shape must be (6,7), got {s.shape[2:]}")
        if p.ndim != 2 or p.shape[1] != COLS:
            raise ValueError(f"policies must be shape (N,7), got {p.shape}")
        if v.ndim != 1:
            raise ValueError(f"values must be shape (N,), got {v.shape}")
        if s.shape[0] != p.shape[0] or s.shape[0] != v.shape[0]:
            raise ValueError(
                "states/policies/values N mismatch: "
                f"{s.shape[0]} / {p.shape[0]} / {v.shape[0]}"
            )
        if np.isnan(s).any():
            raise ValueError("states contain NaN")
        if np.isnan(p).any():
            raise ValueError("policies contain NaN")
        if np.isnan(v).any():
            raise ValueError("values contain NaN")
        return s, p, v

    def _trim_to_max_size(self) -> None:
        n = len(self)
        if n <= self.max_size:
            return
        start = n - self.max_size
        self._states = self._states[start:].copy()
        self._policies = self._policies[start:].copy()
        self._values = self._values[start:].copy()

    def _append_arrays(self, states: np.ndarray, policies: np.ndarray, values: np.ndarray) -> None:
        states, policies, values = self._validate_arrays(states, policies, values)
        if states.shape[0] == 0:
            return

        if self._states is None:
            self._states = states.copy()
            self._policies = policies.copy()
            self._values = values.copy()
            self._trim_to_max_size()
            return

        if self._states.shape[1:] != states.shape[1:]:
            raise ValueError(
                "state channel shape mismatch: "
                f"buffer={self._states.shape[1:]}, incoming={states.shape[1:]}"
            )

        self._states = np.concatenate([self._states, states], axis=0).astype(np.float32, copy=False)
        self._policies = np.concatenate([self._policies, policies], axis=0).astype(np.float32, copy=False)
        self._values = np.concatenate([self._values, values], axis=0).astype(np.float32, copy=False)
        self._trim_to_max_size()

    def add_npz(self, path: str | Path) -> int:
        """Append states/policies/values from an NPZ file. Returns added count."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"npz not found: {p}")
        data = np.load(str(p), allow_pickle=True)
        if "states" not in data or "policies" not in data or "values" not in data:
            raise ValueError(f"npz missing required arrays: {p}")
        states = np.asarray(data["states"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)
        n0 = len(self)
        self._append_arrays(states, policies, values)
        return len(self) - n0

    def add_examples(self, examples) -> int:
        """Append examples from arrays, dict, or SelfPlayExample-like sequence."""
        if examples is None:
            return 0

        # dict-style: {"states":..., "policies":..., "values":...}
        if isinstance(examples, dict):
            states = examples.get("states")
            policies = examples.get("policies")
            values = examples.get("values")
            n0 = len(self)
            self._append_arrays(states, policies, values)
            return len(self) - n0

        # tuple/list of arrays: (states, policies, values)
        if (
            isinstance(examples, (tuple, list))
            and len(examples) == 3
            and all(hasattr(x, "shape") for x in examples)
        ):
            states, policies, values = examples
            n0 = len(self)
            self._append_arrays(states, policies, values)
            return len(self) - n0

        # sequence of SelfPlayExample-like objects
        if isinstance(examples, Sequence):
            if len(examples) == 0:
                return 0
            states_list = []
            policies_list = []
            values_list = []
            for ex in examples:
                if isinstance(ex, dict):
                    states_list.append(np.asarray(ex["state"], dtype=np.float32))
                    policies_list.append(np.asarray(ex["policy_target"], dtype=np.float32))
                    values_list.append(float(ex["value_target"]))
                else:
                    states_list.append(np.asarray(ex.state, dtype=np.float32))
                    policies_list.append(np.asarray(ex.policy_target, dtype=np.float32))
                    values_list.append(float(ex.value_target))
            states = np.stack(states_list, axis=0)
            policies = np.stack(policies_list, axis=0)
            values = np.asarray(values_list, dtype=np.float32)
            n0 = len(self)
            self._append_arrays(states, policies, values)
            return len(self) - n0

        raise TypeError("Unsupported examples format for add_examples")

    def sample_batch(self, batch_size: int):
        if len(self) == 0:
            raise ValueError("ReplayBuffer is empty")
        b = max(1, int(batch_size))
        n = len(self)
        replace = n < b
        idx = np.random.choice(n, size=b, replace=replace)
        return (
            self._states[idx].astype(np.float32, copy=False),
            self._policies[idx].astype(np.float32, copy=False),
            self._values[idx].astype(np.float32, copy=False),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if self._states is None:
            states = np.zeros((0, 3, ROWS, COLS), dtype=np.float32)
            policies = np.zeros((0, COLS), dtype=np.float32)
            values = np.zeros((0,), dtype=np.float32)
        else:
            states = self._states
            policies = self._policies
            values = self._values

        metadata = {
            "max_size": int(self.max_size),
            "size": int(len(self)),
            "state_shape": list(states.shape[1:]),
            "created_at": _utc_now_iso(),
        }
        np.savez_compressed(
            str(p),
            states=states.astype(np.float32, copy=False),
            policies=policies.astype(np.float32, copy=False),
            values=values.astype(np.float32, copy=False),
            metadata=json.dumps(metadata, ensure_ascii=False),
        )
        return p

    def load(self, path: str | Path) -> "ReplayBuffer":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"ReplayBuffer file not found: {p}")
        data = np.load(str(p), allow_pickle=True)
        states = np.asarray(data["states"], dtype=np.float32)
        policies = np.asarray(data["policies"], dtype=np.float32)
        values = np.asarray(data["values"], dtype=np.float32)

        metadata = _parse_metadata(data["metadata"] if "metadata" in data else None)
        max_size = metadata.get("max_size")
        if max_size is not None:
            try:
                self.max_size = max(1, int(max_size))
            except Exception:
                pass

        self._states = None
        self._policies = None
        self._values = None
        self._append_arrays(states, policies, values)
        return self

