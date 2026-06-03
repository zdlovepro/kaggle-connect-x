"""Runtime CPU/worker helpers for controlled local parallelism.

The main goal is to speed up teacher/self-play generation on multi-core CPUs
without fully saturating the machine. We keep:
  - a small number of intra-op threads in the main process
  - single-threaded worker processes
  - a few cores reserved for the OS / foreground use
"""

from __future__ import annotations

import os
from typing import Dict, Optional


def detected_cpu_count() -> int:
    try:
        count = int(os.cpu_count() or 0)
    except Exception:
        count = 0
    return max(1, count)


def suggest_cpu_plan(total_cpus: Optional[int] = None) -> Dict[str, int]:
    total = max(1, int(total_cpus or detected_cpu_count()))
    reserve_cores = max(2, total // 4)
    usable_cores = max(1, total - reserve_cores)
    main_threads = max(1, min(3, usable_cores // 4 if usable_cores >= 4 else 1))
    worker_budget = max(1, usable_cores - main_threads)
    teacher_workers = max(1, min(8, worker_budget))
    selfplay_workers = max(1, min(6, worker_budget))
    eval_workers = max(1, min(4, worker_budget))
    return {
        "total_cpus": total,
        "main_threads": main_threads,
        "teacher_workers": teacher_workers,
        "selfplay_workers": selfplay_workers,
        "eval_workers": eval_workers,
        "reserve_cores": reserve_cores,
        "usable_cores": usable_cores,
        "worker_threads": 1,
    }


def _apply_env_threads(threads: int) -> None:
    t = str(max(1, int(threads)))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = t


def configure_cpu_runtime(
    intra_threads: Optional[int] = None,
    interop_threads: int = 1,
) -> Dict[str, int]:
    plan = suggest_cpu_plan()
    threads = max(1, int(intra_threads or plan["main_threads"]))
    _apply_env_threads(threads)
    applied = {
        "intra_threads": threads,
        "interop_threads": max(1, int(interop_threads)),
    }
    try:
        import torch

        torch.set_num_threads(applied["intra_threads"])
        try:
            torch.set_num_interop_threads(applied["interop_threads"])
        except Exception:
            pass
    except Exception:
        pass
    return applied


def configure_worker_runtime(worker_threads: int = 1) -> Dict[str, int]:
    return configure_cpu_runtime(intra_threads=max(1, int(worker_threads)), interop_threads=1)
