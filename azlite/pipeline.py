"""AlphaZero-lite end-to-end pipeline (bootstrap / selfplay / full).

Goals:
  - Recoverable and reproducible training flow.
  - Stronger evaluation matrix than random-only checks.
  - Safe checkpoint promotion and metadata traceability.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from azlite import evaluate as eval_mod
from azlite.teacher_data import (
    DEFAULT_SOURCE_MIX,
    build_teacher_dataset,
    parse_source_mix,
    save_dataset_npz,
)
from evaluate.build_submission import build_submission


ROOT = Path(__file__).resolve().parents[1]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelinePaths:
    data_teacher: Path
    data_selfplay: Path
    checkpoints: Path
    logs: Path
    train_ckpt_dir: Path
    pipeline_state: Path
    eval_history: Path
    training_summary: Path


def _ensure_layout(train_ckpt_dir: Path) -> PipelinePaths:
    if not train_ckpt_dir.is_absolute():
        train_ckpt_dir = ROOT / train_ckpt_dir
    data_teacher = ROOT / "data" / "teacher"
    data_selfplay = ROOT / "data" / "selfplay"
    checkpoints = ROOT / "checkpoints"
    logs = ROOT / "logs"

    for d in (data_teacher, data_selfplay, checkpoints, logs, train_ckpt_dir):
        d.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(
        data_teacher=data_teacher,
        data_selfplay=data_selfplay,
        checkpoints=checkpoints,
        logs=logs,
        train_ckpt_dir=train_ckpt_dir,
        pipeline_state=checkpoints / "pipeline_state.json",
        eval_history=logs / "pipeline_eval_history.json",
        training_summary=logs / "training_summary.md",
    )


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _git_capture(args: Sequence[str]) -> str:
    try:
        cp = subprocess.run(
            list(args),
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return cp.stdout.strip()
    except Exception:
        return "unknown"


def _git_info() -> Dict[str, str]:
    return {
        "branch": _git_capture(("git", "rev-parse", "--abbrev-ref", "HEAD")),
        "commit": _git_capture(("git", "rev-parse", "HEAD")),
    }


def _require_torch() -> None:
    try:
        import torch  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "This mode requires PyTorch. Install torch in your runtime first."
        ) from exc


def _run_subprocess(cmd: Sequence[str]) -> None:
    print("[pipeline] run:", " ".join(cmd))
    subprocess.run(list(cmd), cwd=str(ROOT), check=True)


def _merge_eval_results(existing: Any, incoming: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(existing, dict):
        out.update(existing)
    if isinstance(incoming, dict):
        out.update(incoming)
    return out


def _update_checkpoint_metadata(checkpoint_path: Path, extra_meta: Dict[str, Any]) -> None:
    if not checkpoint_path.exists():
        return
    try:
        import torch
    except Exception:
        print(f"[pipeline] WARN: skip metadata patch (torch unavailable): {checkpoint_path}")
        return

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    meta = dict(ckpt.get("metadata") or {})

    extra = dict(extra_meta)
    if "eval_results" in extra:
        meta["eval_results"] = _merge_eval_results(meta.get("eval_results"), extra.pop("eval_results"))
    meta.update(extra)
    ckpt["metadata"] = meta
    torch.save(ckpt, str(checkpoint_path))


def _warning_random_overfit(candidate_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
    r_rand = candidate_results.get("random")
    r_nega = candidate_results.get("negamax")
    if r_rand is None or r_nega is None:
        return None
    if float(r_rand["win_rate"]) >= 0.80 and float(r_nega["win_rate"]) <= 0.55:
        return (
            "random is not a reliable gating metric; model may be overfitting weak play "
            "or relying on tactical shortcuts only."
        )
    return None


def _format_eval_rows(rows: Sequence[Dict[str, Any]]) -> str:
    lines = [
        "| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | Timeout(A) | Timeout(B) | AvgStepMs(A) | P95StepMs(A) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {opp} | {w}/{l}/{d} | {wr:.1f}% | {fp:.1f}% | {sp:.1f}% | {ill} | {to_a} | {to_b} | {avg:.1f} | {p95:.1f} |".format(
                opp=r.get("opponent", "?"),
                w=int(r["wins"]),
                l=int(r["losses"]),
                d=int(r["draws"]),
                wr=float(r["win_rate"]) * 100.0,
                fp=float(r.get("first_player_win_rate", 0.0)) * 100.0,
                sp=float(r.get("second_player_win_rate", 0.0)) * 100.0,
                ill=int(r.get("invalid_actions", {}).get("candidate", r["illegal_actions"]["agent_a"])),
                to_a=int(r.get("candidate_timeouts", r["timeouts"]["agent_a"])),
                to_b=int(r.get("opponent_timeouts", r["timeouts"]["agent_b"])),
                avg=float(r["avg_step_time_sec"]["agent_a"]) * 1000.0,
                p95=float(r["p95_step_time_sec"]["agent_a"]) * 1000.0,
            )
        )
    return "\n".join(lines)


def _evaluate_checkpoint_matrix(
    checkpoint_path: Path,
    previous_best_checkpoint: Optional[Path],
    opponents: Sequence[str],
    games: int,
    simulations: int,
    device: str,
    seed: int,
    max_opponent_timeout_rate: float,
    max_candidate_timeout_rate: float,
    timeout_result_policy: str,
    logs_dir: Path,
    label: str,
) -> Dict[str, Any]:
    candidate = eval_mod.create_agent(
        "checkpoint_puct",
        checkpoint=str(checkpoint_path),
        previous_best_checkpoint=str(previous_best_checkpoint) if previous_best_checkpoint else None,
        simulations=int(simulations),
        device=device,
        seed=int(seed),
    )

    candidate_results: Dict[str, Dict[str, Any]] = {}
    opponent_agents: List[Tuple[str, Any]] = []
    for i, spec in enumerate(opponents):
        opp = eval_mod.create_agent(
            spec,
            checkpoint=str(checkpoint_path),
            previous_best_checkpoint=str(previous_best_checkpoint) if previous_best_checkpoint else None,
            simulations=int(simulations),
            device=device,
            seed=int(seed) + 100 + i,
        )
        opponent_agents.append((spec, opp))

    for i, (spec, opp) in enumerate(opponent_agents):
        r = eval_mod.play_match(
            candidate,
            opp,
            num_games=int(games),
            swap_sides=True,
            seed=int(seed) + 1000 + i,
            max_opponent_timeout_rate=float(max_opponent_timeout_rate),
            max_candidate_timeout_rate=float(max_candidate_timeout_rate),
            timeout_result_policy=str(timeout_result_policy),
        )
        r["opponent"] = spec
        candidate_results[spec] = r

    previous_best_results: Optional[Dict[str, Dict[str, Any]]] = None
    if previous_best_checkpoint is not None and previous_best_checkpoint.exists():
        prev_best_agent = eval_mod.create_agent(
            "previous_best",
            checkpoint=str(checkpoint_path),
            previous_best_checkpoint=str(previous_best_checkpoint),
            simulations=int(simulations),
            device=device,
            seed=int(seed) + 2222,
        )
        r_prev = eval_mod.play_match(
            candidate,
            prev_best_agent,
            num_games=max(20, int(games)),
            swap_sides=True,
            seed=int(seed) + 2223,
            max_opponent_timeout_rate=float(max_opponent_timeout_rate),
            max_candidate_timeout_rate=float(max_candidate_timeout_rate),
            timeout_result_policy=str(timeout_result_policy),
        )
        r_prev["opponent"] = "previous_best"
        candidate_results["previous_best"] = r_prev

        previous_best_results = {}
        for i, (spec, opp) in enumerate(opponent_agents):
            rr = eval_mod.play_match(
                prev_best_agent,
                opp,
                num_games=int(games),
                swap_sides=True,
                seed=int(seed) + 3000 + i,
                max_opponent_timeout_rate=float(max_opponent_timeout_rate),
                max_candidate_timeout_rate=float(max_candidate_timeout_rate),
                timeout_result_policy=str(timeout_result_policy),
            )
            rr["opponent"] = spec
            previous_best_results[spec] = rr

    passed, reasons = eval_mod.should_promote_candidate(
        candidate_results,
        previous_best_results,
        max_opponent_timeout_rate=float(max_opponent_timeout_rate),
        max_candidate_timeout_rate=float(max_candidate_timeout_rate),
    )
    warning = _warning_random_overfit(candidate_results)
    reliability_warnings = []
    overall_reliable = True
    overall_unreliable_reasons: List[str] = []
    for opp, r in candidate_results.items():
        timeout_rate = float(r.get("opponent_timeout_rate", 0.0))
        if timeout_rate > float(max_opponent_timeout_rate):
            msg = (
                f"[eval][warning] opponent={opp} timeout_rate={timeout_rate*100.0:.1f}%, "
                "win_rate is unreliable and excluded from gating."
            )
            reliability_warnings.append(msg)
            print(msg)
        if not bool(r.get("reliable", True)):
            overall_reliable = False
            reasons_local = r.get("unreliable_reasons") or []
            if reasons_local:
                for reason in reasons_local:
                    overall_unreliable_reasons.append(f"{opp}: {reason}")
            else:
                overall_unreliable_reasons.append(f"{opp}: unreliable matchup")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"pipeline_eval_{label}_{ts}.json"
    md_path = logs_dir / f"pipeline_eval_{label}_{ts}.md"

    payload = {
        "created_at": _utc_now_iso(),
        "label": label,
        "evaluator_backend": eval_mod.EVALUATOR_BACKEND,
        "candidate_checkpoint": str(checkpoint_path.resolve()),
        "checkpoint": str(checkpoint_path.resolve()),
        "previous_best_checkpoint": (
            str(previous_best_checkpoint.resolve())
            if previous_best_checkpoint is not None and previous_best_checkpoint.exists()
            else None
        ),
        "opponents": list(opponents),
        "reliable": bool(overall_reliable),
        "unreliable_reasons": overall_unreliable_reasons,
        "candidate_results": candidate_results,
        "previous_best_results": previous_best_results,
        "gating": {
            "passed": bool(passed),
            "reasons": reasons,
            "warning": warning,
            "reliability_warnings": reliability_warnings,
        },
        "config": {
            "games": int(games),
            "simulations": int(simulations),
            "device": device,
            "seed": int(seed),
            "max_opponent_timeout_rate": float(max_opponent_timeout_rate),
            "max_candidate_timeout_rate": float(max_candidate_timeout_rate),
            "timeout_result_policy": str(timeout_result_policy),
        },
    }
    _save_json(json_path, payload)

    rows = [candidate_results[k] for k in candidate_results.keys()]
    md_lines = [
        "# Pipeline Evaluation",
        "",
        f"- Label: `{label}`",
        f"- Evaluator backend: `{eval_mod.EVALUATOR_BACKEND}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Previous best: `{payload['previous_best_checkpoint']}`",
        "",
        "## Matrix",
        _format_eval_rows(rows),
        "",
        "## Gating",
        f"- Passed: `{passed}`",
        f"- Reliable: `{overall_reliable}`",
    ]
    for reason in reasons:
        md_lines.append(f"- {reason}")
    if warning:
        md_lines.extend(["", "## Warning", f"> {warning}"])
    if reliability_warnings:
        md_lines.extend(["", "## Reliability Warnings"])
        for w in reliability_warnings:
            md_lines.append(f"- {w}")
    if overall_unreliable_reasons:
        md_lines.extend(["", "## Unreliable Reasons"])
        for reason in overall_unreliable_reasons:
            md_lines.append(f"- {reason}")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    payload["log_paths"] = {
        "json": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
    }
    return payload


def _append_eval_history(history_path: Path, entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    history = _load_json(history_path, [])
    if not isinstance(history, list):
        history = []
    history.append(entry)
    _save_json(history_path, history)
    return history


def _sync_selfplay_npz(src_dir: Path, dst_dir: Path) -> int:
    if not src_dir.exists():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for npz in sorted(src_dir.glob("selfplay_*.npz")):
        target = dst_dir / npz.name
        if target.exists():
            continue
        shutil.copy2(npz, target)
        copied += 1
    return copied


def _default_teacher_path(paths: PipelinePaths, positions: int, depth: int) -> Path:
    return paths.data_teacher / f"teacher_d{int(depth)}_n{int(positions)}.npz"


def _snapshot_previous_best(best_path: Path, checkpoints_dir: Path) -> Optional[Path]:
    if not best_path.exists():
        return None
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap = checkpoints_dir / f"previous_best_{ts}.pt"
    shutil.copy2(best_path, snap)
    return snap


def _build_context_metadata(
    args: argparse.Namespace,
    git_info: Dict[str, str],
    teacher_data_path: Optional[Path],
    eval_results: Optional[Dict[str, Any]],
    iteration: Optional[int] = None,
) -> Dict[str, Any]:
    return {
        "branch_name": git_info.get("branch", "unknown"),
        "git_commit_hash": git_info.get("commit", "unknown"),
        "iteration": int(iteration) if iteration is not None else 0,
        "teacher_data_path": str(teacher_data_path.resolve()) if teacher_data_path else None,
        "self_play_games": int(args.self_play_games),
        "simulations": int(args.simulations),
        "train_steps": int(args.train_steps),
        "config": vars(args),
        "eval_results": eval_results or {},
    }


def _recommend_next_params(args: argparse.Namespace, latest_eval: Optional[Dict[str, Any]], bottleneck: str) -> List[str]:
    recs: List[str] = []
    if "unreliable" in bottleneck or "timeout" in bottleneck or "illegal" in bottleneck:
        recs.append(
            f"`--simulations {max(32, int(args.simulations) - 20)}` reduce search load first to remove timeout/illegal."
        )
        recs.append(
            f"`--self-play-games {int(args.self_play_games)}` keep coverage stable while fixing reliability."
        )
        return recs

    if "random high but negamax low" in bottleneck or "negamax" in bottleneck:
        recs.append(
            f"`--simulations {int(args.simulations) + 20}` raise search quality for stronger targets."
        )
        recs.append(
            f"`--self-play-games {int(args.self_play_games) + max(2, int(args.self_play_games)//5)}` increase self-play coverage."
        )
        recs.append(
            f"`--teacher-depth {int(args.teacher_depth) + 1}` strengthen teacher bootstrap."
        )
        return recs

    if "previous best" in bottleneck or "regression" in bottleneck:
        recs.append(f"`--train-lr {max(1e-5, float(args.train_lr) * 0.7):.6f}` lower lr for stability.")
        recs.append(f"`--train-steps {int(args.train_steps) + max(20, int(args.train_steps)//5)}` add train steps.")
        return recs

    recs.append(f"`--iterations {int(args.iterations) + 5}` continue with more iterations.")
    recs.append(f"`--eval-games {max(100, int(args.eval_games))}` keep evaluation stability.")
    recs.append(f"`--simulations {int(args.simulations)}` keep simulations unchanged and observe trend.")
    return recs


def _infer_bottleneck(latest_eval: Optional[Dict[str, Any]]) -> str:
    if not latest_eval:
        return "no evaluation data yet."

    cand = latest_eval.get("candidate_results", {})
    g = latest_eval.get("gating", {})
    reasons = list(g.get("reasons") or [])
    reliable = bool(latest_eval.get("reliable", True))
    unreliable_reasons = list(latest_eval.get("unreliable_reasons") or [])

    if (not reliable) or unreliable_reasons:
        return "pipeline final eval unreliable because opponent timeout rate is too high"

    illegal_or_timeout = any(
        ("illegal" in r.lower()) or ("timeout" in r.lower()) or ("runtime errors" in r.lower())
        for r in reasons
    )
    if illegal_or_timeout:
        return "stability bottleneck: illegal action / timeout / runtime error exists."

    wr_random = float(cand.get("random", {}).get("win_rate", 0.0))
    wr_nega = float(cand.get("negamax", {}).get("win_rate", 0.0))
    wr_mcts = float(cand.get("mcts_lite", {}).get("win_rate", 0.0))
    wr_prev = float(cand.get("previous_best", {}).get("win_rate", 1.0))

    if wr_random >= 0.8 and wr_nega <= 0.55:
        return "strength bottleneck: random high but negamax low."
    if wr_prev < 0.55:
        return "regression bottleneck: weak edge vs previous best."
    if wr_nega < 0.5:
        return "search-quality bottleneck: low win rate vs negamax."
    if wr_mcts < 0.5:
        return "strength bottleneck: low win rate vs mcts_lite."
    return "no obvious bottleneck."

def _write_training_summary(
    summary_path: Path,
    state: Dict[str, Any],
    history: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    def _find_last_eval(label: str) -> Optional[Dict[str, Any]]:
        for rec in reversed(history):
            if str(rec.get("label", "")) == label:
                return rec
        return None

    def _fmt_match(rec: Optional[Dict[str, Any]], opp: str) -> str:
        if not rec:
            return "n/a"
        cand = rec.get("candidate_results", {}) or {}
        if opp not in cand:
            return "n/a"
        r = cand[opp]
        raw_wr = float(r.get("win_rate", 0.0)) * 100.0
        base = f"{int(r.get('wins', 0))}/{int(r.get('losses', 0))}/{int(r.get('draws', 0))} ({raw_wr:.1f}%)"
        if not bool(r.get("reliable", True)):
            rel_wr = float(r.get("reliable_win_rate", r.get("win_rate", 0.0))) * 100.0
            rel_games = int(r.get("reliable_games", 0))
            return f"{base} [UNRELIABLE, reliable={rel_wr:.1f}%/{rel_games}]"
        return base

    recent = history[-5:]
    final_latest_eval = _find_last_eval("final_latest_eval")
    final_best_eval = _find_last_eval("final_best_eval")
    bottleneck_source = final_latest_eval or (recent[-1] if recent else None)
    bottleneck = _infer_bottleneck(bottleneck_source)
    recs = _recommend_next_params(args, bottleneck_source, bottleneck)

    best_ckpt = str(state.get("best_checkpoint", "") or "")
    latest_ckpt = str(state.get("latest_checkpoint", "") or "")
    best_iteration = int(state.get("best_iteration", 0) or 0)
    latest_iteration = int(state.get("last_iteration", 0) or 0)

    latest_passed = bool(
        final_latest_eval
        and bool(final_latest_eval.get("gating", {}).get("passed", False))
        and bool(final_latest_eval.get("reliable", False))
    )
    best_reliable = bool(final_best_eval and bool(final_best_eval.get("reliable", False)))
    latest_reliable = bool(final_latest_eval and bool(final_latest_eval.get("reliable", False)))

    recommended_label = "latest.pt" if latest_passed else "best.pt"
    recommended_path = latest_ckpt if latest_passed else best_ckpt

    lines = [
        "# Training Summary",
        "",
        f"- Updated at: `{_utc_now_iso()}`",
        f"- Best checkpoint: `{best_ckpt}` (iteration `{best_iteration}`)",
        f"- Latest checkpoint: `{latest_ckpt}` (iteration `{latest_iteration}`)",
        f"- Latest passed gating: `{latest_passed}`",
        f"- Recommended checkpoint: `{recommended_label}`",
        f"- Recommended path: `{recommended_path}`",
        "",
        "## Final Checkpoint Comparison",
        "| Metric | best.pt | latest.pt |",
        "|---|---|---|",
        f"| Label | final_best_eval | final_latest_eval |",
        f"| Iteration | {best_iteration} | {latest_iteration} |",
        f"| Reliable | {bool(final_best_eval.get('reliable', False)) if final_best_eval else 'n/a'} | {bool(final_latest_eval.get('reliable', False)) if final_latest_eval else 'n/a'} |",
        f"| Gating passed | {bool(final_best_eval.get('gating', {}).get('passed', False)) if final_best_eval else 'n/a'} | {bool(final_latest_eval.get('gating', {}).get('passed', False)) if final_latest_eval else 'n/a'} |",
        f"| vs random | {_fmt_match(final_best_eval, 'random')} | {_fmt_match(final_latest_eval, 'random')} |",
        f"| vs negamax | {_fmt_match(final_best_eval, 'negamax')} | {_fmt_match(final_latest_eval, 'negamax')} |",
        f"| vs mcts_lite | {_fmt_match(final_best_eval, 'mcts_lite')} | {_fmt_match(final_latest_eval, 'mcts_lite')} |",
        "",
    ]

    if (final_best_eval and not best_reliable) or (final_latest_eval and not latest_reliable):
        lines.append("- Final comparison contains unreliable eval(s); those rows are not valid for strength ranking.")
        lines.append("")

    if latest_ckpt and (not latest_passed):
        lines.append(
            f"- latest checkpoint did not pass gating; current best remains iteration {best_iteration}."
        )
        lines.append("")

    lines.append("## Recent 5 Evaluations")
    if not recent:
        lines.append("- No evaluation records yet.")
    else:
        for idx, rec in enumerate(recent, start=1):
            lines.extend(
                [
                    f"### Eval #{idx}",
                    f"- Time: `{rec.get('created_at', '')}`",
                    f"- Mode: `{rec.get('label', '')}`",
                    f"- Checkpoint: `{rec.get('checkpoint', '')}`",
                    f"- Reliable: `{rec.get('reliable', True)}`",
                    f"- Gating passed: `{rec.get('gating', {}).get('passed', False)}`",
                ]
            )
            reasons = rec.get("gating", {}).get("reasons") or []
            if reasons:
                lines.append(f"- Reasons: {'; '.join(str(x) for x in reasons)}")
            unr = rec.get("unreliable_reasons") or []
            if unr:
                lines.append(f"- Unreliable reasons: {'; '.join(str(x) for x in unr)}")
            lines.append("")

    lines.extend(
        [
            "## Bottleneck",
            f"- {bottleneck}",
            "",
            "## Next Suggestions",
        ]
    )
    for rec in recs:
        lines.append(f"- {rec}")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _run_bootstrap(
    args: argparse.Namespace,
    paths: PipelinePaths,
    git_info: Dict[str, str],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    teacher_path = Path(args.teacher_data_path) if args.teacher_data_path else _default_teacher_path(
        paths, args.teacher_positions, args.teacher_depth
    )

    if args.resume and teacher_path.exists():
        print(f"[pipeline] bootstrap: reuse teacher data {teacher_path}")
    else:
        mix = parse_source_mix(args.teacher_source_mix)
        print("[pipeline] bootstrap: generating teacher data...")
        states, policies, values, metadata = build_teacher_dataset(
            positions=int(args.teacher_positions),
            teacher=str(args.teacher),
            teacher_depth=int(args.teacher_depth),
            policy_mode=str(args.teacher_policy_mode),
            policy_temperature=float(args.teacher_policy_temperature),
            value_mode=str(args.teacher_value_mode),
            value_scale=float(args.teacher_value_scale),
            source_mix=mix,
            include_legal_channel=not bool(args.teacher_no_legal_channel),
            seed=int(args.seed),
            mcts_sims=int(args.teacher_mcts_sims),
            mcts_time_ms=float(args.teacher_mcts_time_ms),
            negamax_time_ms=float(args.teacher_negamax_time_ms),
            rollout_policy=str(args.teacher_rollout_policy),
        )
        save_dataset_npz(teacher_path, states, policies, values, metadata)
        print(f"[pipeline] bootstrap: teacher data saved {teacher_path.resolve()}")

    pretrain_ckpt = Path(args.pretrain_checkpoint)
    if args.resume and pretrain_ckpt.exists():
        print(f"[pipeline] bootstrap: reuse pretrained checkpoint {pretrain_ckpt}")
    else:
        _require_torch()
        from azlite import pretrain as pre
        from azlite.model import ConnectXNet, save_checkpoint

        print("[pipeline] bootstrap: pretraining...")
        states, policies, values, data_meta = pre._load_dataset(teacher_path)
        model = ConnectXNet(in_channels=int(states.shape[1]))
        model, optimizer, train_metrics = pre.train_supervised(
            model=model,
            states=states,
            policies=policies,
            values=values,
            device=str(args.device),
            epochs=int(args.pretrain_epochs),
            batch_size=int(args.pretrain_batch_size),
            lr=float(args.pretrain_lr),
            value_loss_weight=float(args.pretrain_value_loss_weight),
        )
        metadata = {
            "iteration": 0,
            "train_steps": int(args.pretrain_epochs)
            * int(np.ceil(states.shape[0] / max(1, int(args.pretrain_batch_size)))),
            "simulations": int(args.simulations),
            "teacher_data_path": str(teacher_path.resolve()),
            "train_metrics": train_metrics,
            "data_metadata": data_meta,
            "branch_name": git_info.get("branch", "unknown"),
            "git_commit_hash": git_info.get("commit", "unknown"),
            "self_play_games": 0,
            "config": vars(args),
            "eval_results": {},
        }
        save_checkpoint(model, optimizer, pretrain_ckpt, metadata=metadata)
        print(f"[pipeline] bootstrap: pretrained checkpoint saved {pretrain_ckpt.resolve()}")

    eval_payload = None
    if not args.skip_bootstrap_eval:
        try:
            opponents = [x.strip() for x in str(args.opponents).split(",") if x.strip()]
            eval_payload = _evaluate_checkpoint_matrix(
                checkpoint_path=pretrain_ckpt,
                previous_best_checkpoint=None,
                opponents=opponents,
                games=int(args.eval_games),
                simulations=int(args.simulations),
                device=str(args.device),
                seed=int(args.seed),
                max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
                max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
                timeout_result_policy=str(args.timeout_result_policy),
                logs_dir=paths.logs,
                label="bootstrap_pretrained",
            )
            history = _append_eval_history(paths.eval_history, eval_payload)
            _write_training_summary(paths.training_summary, state, history, args)
            print("[pipeline] bootstrap: evaluation done.")
        except Exception as exc:
            print(f"[pipeline] WARN: bootstrap eval failed, checkpoint kept. err={exc}")
            traceback.print_exc()

    extra_meta = _build_context_metadata(args, git_info, teacher_path, eval_payload, iteration=0)
    _update_checkpoint_metadata(pretrain_ckpt, extra_meta)

    state.update(
        {
            "teacher_data_path": str(teacher_path.resolve()),
            "pretrained_checkpoint": str(pretrain_ckpt.resolve()),
            "updated_at": _utc_now_iso(),
            "last_run_mode": "bootstrap",
        }
    )
    _save_json(paths.pipeline_state, state)
    return state


def _run_selfplay(
    args: argparse.Namespace,
    paths: PipelinePaths,
    git_info: Dict[str, str],
    state: Dict[str, Any],
) -> Dict[str, Any]:
    _require_torch()

    ckpt_dir = paths.train_ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = ckpt_dir / "latest.pt"
    best_ckpt = ckpt_dir / "best.pt"
    teacher_path = Path(state["teacher_data_path"]) if state.get("teacher_data_path") else None

    start_ckpt: Optional[Path] = None
    if args.checkpoint:
        start_ckpt = Path(args.checkpoint)
    elif args.resume and latest_ckpt.exists():
        start_ckpt = latest_ckpt
    elif state.get("pretrained_checkpoint"):
        start_ckpt = Path(state["pretrained_checkpoint"])
    elif Path(args.pretrain_checkpoint).exists():
        start_ckpt = Path(args.pretrain_checkpoint)

    prev_best_snapshot = _snapshot_previous_best(best_ckpt, paths.checkpoints)
    if prev_best_snapshot is not None:
        print(f"[pipeline] selfplay: snapshot previous best -> {prev_best_snapshot}")

    cmd = [
        sys.executable,
        "-m",
        "azlite.train",
        "--iterations",
        str(int(args.iterations)),
        "--self-play-games",
        str(int(args.self_play_games)),
        "--simulations",
        str(int(args.simulations)),
        "--batch-size",
        str(int(args.batch_size)),
        "--train-steps",
        str(int(args.train_steps)),
        "--lr",
        str(float(args.train_lr)),
        "--buffer-size",
        str(int(args.buffer_size)),
        "--device",
        str(args.device),
        "--checkpoint-dir",
        str(ckpt_dir),
        "--eval-interval",
        str(int(args.eval_interval)),
        "--eval-games",
        str(int(args.eval_games)),
        "--max-opponent-timeout-rate",
        str(float(args.max_opponent_timeout_rate)),
        "--max-candidate-timeout-rate",
        str(float(args.max_candidate_timeout_rate)),
        "--timeout-result-policy",
        str(args.timeout_result_policy),
        "--seed",
        str(int(args.seed)),
    ]
    if args.resume:
        cmd.append("--resume")
    if start_ckpt is not None and start_ckpt.exists() and not args.resume:
        cmd.extend(["--checkpoint", str(start_ckpt)])
    if teacher_path is not None and teacher_path.exists():
        cmd.extend(["--teacher-data", str(teacher_path)])

    print("[pipeline] selfplay: training...")
    _run_subprocess(cmd)

    copied = _sync_selfplay_npz(ckpt_dir / "selfplay_npz", paths.data_selfplay)
    if copied > 0:
        print(f"[pipeline] selfplay: copied {copied} self-play npz -> {paths.data_selfplay}")

    if (not latest_ckpt.exists()) and (not best_ckpt.exists()):
        raise RuntimeError("Training finished but neither latest.pt nor best.pt exists.")

    opponents = [x.strip() for x in str(args.opponents).split(",") if x.strip()]
    history = _load_json(paths.eval_history, [])
    if not isinstance(history, list):
        history = []

    eval_payload_latest: Optional[Dict[str, Any]] = None
    eval_payload_best: Optional[Dict[str, Any]] = None

    if latest_ckpt.exists():
        try:
            eval_payload_latest = _evaluate_checkpoint_matrix(
                checkpoint_path=latest_ckpt,
                previous_best_checkpoint=prev_best_snapshot,
                opponents=opponents,
                games=int(args.eval_games),
                simulations=int(args.simulations),
                device=str(args.device),
                seed=int(args.seed) + 77,
                max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
                max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
                timeout_result_policy=str(args.timeout_result_policy),
                logs_dir=paths.logs,
                label="final_latest_eval",
            )
            history = _append_eval_history(paths.eval_history, eval_payload_latest)
            print(
                f"[pipeline] final_latest_eval: passed={eval_payload_latest['gating']['passed']} "
                f"reliable={eval_payload_latest.get('reliable', True)} "
                f"reasons={eval_payload_latest['gating']['reasons']}"
            )
        except Exception as exc:
            print(f"[pipeline] WARN: final_latest_eval failed, checkpoint kept. err={exc}")
            traceback.print_exc()
            history = _load_json(paths.eval_history, [])
            if not isinstance(history, list):
                history = []

    if best_ckpt.exists():
        try:
            eval_payload_best = _evaluate_checkpoint_matrix(
                checkpoint_path=best_ckpt,
                previous_best_checkpoint=prev_best_snapshot,
                opponents=opponents,
                games=int(args.eval_games),
                simulations=int(args.simulations),
                device=str(args.device),
                seed=int(args.seed) + 78,
                max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
                max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
                timeout_result_policy=str(args.timeout_result_policy),
                logs_dir=paths.logs,
                label="final_best_eval",
            )
            history = _append_eval_history(paths.eval_history, eval_payload_best)
            print(
                f"[pipeline] final_best_eval: passed={eval_payload_best['gating']['passed']} "
                f"reliable={eval_payload_best.get('reliable', True)} "
                f"reasons={eval_payload_best['gating']['reasons']}"
            )
        except Exception as exc:
            print(f"[pipeline] WARN: final_best_eval failed, checkpoint kept. err={exc}")
            traceback.print_exc()
            history = _load_json(paths.eval_history, [])
            if not isinstance(history, list):
                history = []

    latest_passed_gating = bool(
        eval_payload_latest is not None
        and bool(eval_payload_latest.get("gating", {}).get("passed", False))
        and bool(eval_payload_latest.get("reliable", False))
    )
    if latest_ckpt.exists() and (not latest_passed_gating):
        print("[pipeline] latest checkpoint did not pass gating; best checkpoint remains unchanged.")

    if args.build_submission:
        try:
            cfg = build_submission(
                profile=str(args.build_profile),
                template_path=Path(args.submission_template),
                output_path=Path(args.submission_output),
            )
            print(
                "[pipeline] submission build done: "
                f"output={Path(args.submission_output).resolve()} source={cfg.source}"
            )
        except Exception as exc:
            print(f"[pipeline] WARN: submission build failed (checkpoint unaffected). err={exc}")
            traceback.print_exc()

    train_state = _load_json(ckpt_dir / "train_state.json", {})
    last_iteration = None
    best_iteration = None
    if isinstance(train_state, dict) and "last_iteration" in train_state:
        try:
            last_iteration = int(train_state.get("last_iteration"))
        except Exception:
            last_iteration = None
    if isinstance(train_state, dict) and "best_iteration" in train_state:
        try:
            best_iteration = int(train_state.get("best_iteration"))
        except Exception:
            best_iteration = None

    final_eval_bundle = {
        "final_latest_eval": eval_payload_latest,
        "final_best_eval": eval_payload_best,
    }

    extra_meta = _build_context_metadata(
        args,
        git_info,
        teacher_path,
        final_eval_bundle,
        iteration=last_iteration,
    )
    _update_checkpoint_metadata(latest_ckpt, extra_meta)
    _update_checkpoint_metadata(best_ckpt, extra_meta)

    recommended_ckpt = None
    if latest_ckpt.exists() and latest_passed_gating:
        recommended_ckpt = latest_ckpt
    elif best_ckpt.exists():
        recommended_ckpt = best_ckpt
    elif latest_ckpt.exists():
        recommended_ckpt = latest_ckpt

    if recommended_ckpt is not None:
        print(f"[pipeline] recommendation: use {recommended_ckpt}")

    state.update(
        {
            "last_iteration": int(last_iteration) if last_iteration is not None else 0,
            "best_iteration": int(best_iteration) if best_iteration is not None else 0,
            "latest_checkpoint": str(latest_ckpt.resolve()) if latest_ckpt.exists() else "",
            "best_checkpoint": str(best_ckpt.resolve()) if best_ckpt.exists() else "",
            "latest_eval_passed_gating": bool(latest_passed_gating),
            "latest_eval_reliable": bool(
                eval_payload_latest is not None and bool(eval_payload_latest.get("reliable", False))
            ),
            "best_eval_reliable": bool(
                eval_payload_best is not None and bool(eval_payload_best.get("reliable", False))
            ),
            "recommended_checkpoint": str(recommended_ckpt.resolve()) if recommended_ckpt else "",
            "final_eval_latest_log": (
                str(eval_payload_latest.get("log_paths", {}).get("json", ""))
                if eval_payload_latest is not None
                else ""
            ),
            "final_eval_best_log": (
                str(eval_payload_best.get("log_paths", {}).get("json", ""))
                if eval_payload_best is not None
                else ""
            ),
            "train_checkpoint_dir": str(ckpt_dir.resolve()),
            "updated_at": _utc_now_iso(),
            "last_run_mode": "selfplay",
        }
    )
    _save_json(paths.pipeline_state, state)

    _write_training_summary(paths.training_summary, state, history, args)
    return state


def _main() -> None:
    parser = argparse.ArgumentParser(description="AlphaZero-lite pipeline")
    parser.add_argument("--mode", choices=("bootstrap", "selfplay", "full"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    # Bootstrap options.
    parser.add_argument("--teacher-positions", type=int, default=50000)
    parser.add_argument("--teacher-depth", type=int, default=5)
    parser.add_argument("--teacher", choices=("auto", "strong", "negamax", "heuristic", "mcts"), default="auto")
    parser.add_argument("--teacher-source-mix", type=str, default=DEFAULT_SOURCE_MIX)
    parser.add_argument("--teacher-policy-mode", choices=("soft", "one_hot"), default="soft")
    parser.add_argument("--teacher-policy-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-value-mode", choices=("score", "teacher", "rollout"), default="score")
    parser.add_argument("--teacher-value-scale", type=float, default=10000.0)
    parser.add_argument("--teacher-rollout-policy", choices=("heuristic", "negamax", "mcts", "random"), default="heuristic")
    parser.add_argument("--teacher-mcts-sims", type=int, default=96)
    parser.add_argument("--teacher-mcts-time-ms", type=float, default=90.0)
    parser.add_argument("--teacher-negamax-time-ms", type=float, default=220.0)
    parser.add_argument("--teacher-no-legal-channel", action="store_true")
    parser.add_argument("--teacher-data-path", type=str, default=None)

    parser.add_argument("--pretrain-checkpoint", type=str, default="checkpoints/azlite_pretrained.pt")
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--pretrain-batch-size", type=int, default=256)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-value-loss-weight", type=float, default=1.0)
    parser.add_argument("--skip-bootstrap-eval", action="store_true")

    # Selfplay/train options.
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional explicit start checkpoint")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--self-play-games", type=int, default=50)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=128)
    parser.add_argument("--train-lr", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--max-opponent-timeout-rate", type=float, default=0.05)
    parser.add_argument("--max-candidate-timeout-rate", type=float, default=0.01)
    parser.add_argument(
        "--timeout-result-policy",
        type=str,
        choices=("loss", "exclude", "fail_eval"),
        default="fail_eval",
    )
    parser.add_argument("--opponents", type=str, default="random,negamax,mcts_lite")

    # Paths / outputs.
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/azlite_train")
    parser.add_argument("--build-submission", action="store_true")
    parser.add_argument("--build-profile", choices=("auto", "optuna", "td"), default="auto")
    parser.add_argument("--submission-template", type=str, default=str(ROOT / "submission.py"))
    parser.add_argument("--submission-output", type=str, default=str(ROOT / "submission.py"))
    args = parser.parse_args()

    paths = _ensure_layout(Path(args.checkpoint_dir))
    git_info = _git_info()
    state = _load_json(paths.pipeline_state, {})
    if not isinstance(state, dict):
        state = {}

    print(
        f"[pipeline] mode={args.mode} resume={args.resume} "
        f"branch={git_info.get('branch')} commit={git_info.get('commit')}"
    )

    try:
        if args.mode == "bootstrap":
            state = _run_bootstrap(args, paths, git_info, state)
        elif args.mode == "selfplay":
            state = _run_selfplay(args, paths, git_info, state)
        else:
            pre_ckpt = Path(args.pretrain_checkpoint)
            if not pre_ckpt.exists():
                print("[pipeline] full: pretrained checkpoint missing, run bootstrap first.")
                state = _run_bootstrap(args, paths, git_info, state)
            else:
                print(f"[pipeline] full: reuse pretrained checkpoint {pre_ckpt}")
                state["pretrained_checkpoint"] = str(pre_ckpt.resolve())
            state = _run_selfplay(args, paths, git_info, state)
    finally:
        # Keep summary as best-effort artifact even on failures.
        history = _load_json(paths.eval_history, [])
        if not isinstance(history, list):
            history = []
        _write_training_summary(paths.training_summary, state, history, args)
        _save_json(paths.pipeline_state, state)

    print("[pipeline] done.")
    print(f"[pipeline] best checkpoint: {state.get('best_checkpoint', '')}")
    print(f"[pipeline] summary: {paths.training_summary.resolve()}")


if __name__ == "__main__":
    _main()
