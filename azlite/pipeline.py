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
        "| Opponent | W/L/D | WR | Illegal(A) | Timeout(A) | AvgStepMs(A) | P95StepMs(A) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {opp} | {w}/{l}/{d} | {wr:.1f}% | {ill} | {to} | {avg:.1f} | {p95:.1f} |".format(
                opp=r.get("opponent", "?"),
                w=int(r["wins"]),
                l=int(r["losses"]),
                d=int(r["draws"]),
                wr=float(r["win_rate"]) * 100.0,
                ill=int(r["illegal_actions"]["agent_a"]),
                to=int(r["timeouts"]["agent_a"]),
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
            )
            rr["opponent"] = spec
            previous_best_results[spec] = rr

    passed, reasons = eval_mod.should_promote_candidate(candidate_results, previous_best_results)
    warning = _warning_random_overfit(candidate_results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = logs_dir / f"pipeline_eval_{ts}.json"
    md_path = logs_dir / f"pipeline_eval_{ts}.md"

    payload = {
        "created_at": _utc_now_iso(),
        "label": label,
        "checkpoint": str(checkpoint_path.resolve()),
        "previous_best_checkpoint": (
            str(previous_best_checkpoint.resolve())
            if previous_best_checkpoint is not None and previous_best_checkpoint.exists()
            else None
        ),
        "opponents": list(opponents),
        "candidate_results": candidate_results,
        "previous_best_results": previous_best_results,
        "gating": {
            "passed": bool(passed),
            "reasons": reasons,
            "warning": warning,
        },
        "config": {
            "games": int(games),
            "simulations": int(simulations),
            "device": device,
            "seed": int(seed),
        },
    }
    _save_json(json_path, payload)

    rows = [candidate_results[k] for k in candidate_results.keys()]
    md_lines = [
        "# Pipeline Evaluation",
        "",
        f"- Label: `{label}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Previous best: `{payload['previous_best_checkpoint']}`",
        "",
        "## Matrix",
        _format_eval_rows(rows),
        "",
        "## Gating",
        f"- Passed: `{passed}`",
    ]
    for reason in reasons:
        md_lines.append(f"- {reason}")
    if warning:
        md_lines.extend(["", "## Warning", f"> {warning}"])
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
    if "稳定性" in bottleneck:
        recs.append(
            f"`--simulations {max(32, int(args.simulations) - 20)}` 降一点，先消除 timeout/illegal。"
        )
        recs.append(
            f"`--self-play-games {int(args.self_play_games)}` 保持，先看稳定性是否恢复。"
        )
        return recs

    if "random 高但 negamax 低" in bottleneck or "negamax" in bottleneck:
        recs.append(
            f"`--simulations {int(args.simulations) + 20}` 提升根访问分布质量。"
        )
        recs.append(
            f"`--self-play-games {int(args.self_play_games) + max(2, int(args.self_play_games)//5)}` 增加自博弈覆盖。"
        )
        recs.append(
            f"`--teacher-depth {int(args.teacher_depth) + 1}` 做更强 bootstrap。"
        )
        return recs

    if "previous best" in bottleneck or "退化" in bottleneck:
        recs.append(f"`--train-lr {max(1e-5, float(args.train_lr) * 0.7):.6f}` 稍降学习率。")
        recs.append(f"`--train-steps {int(args.train_steps) + max(20, int(args.train_steps)//5)}` 稍增训练步数。")
        return recs

    recs.append(f"`--iterations {int(args.iterations) + 5}` 继续扩大迭代轮数。")
    recs.append(f"`--eval-games {max(100, int(args.eval_games))}` 保持评估稳定性。")
    recs.append(f"`--simulations {int(args.simulations)}` 可先不变，观察趋势。")
    return recs


def _infer_bottleneck(latest_eval: Optional[Dict[str, Any]]) -> str:
    if not latest_eval:
        return "暂无评估数据，当前瓶颈未知。"

    cand = latest_eval.get("candidate_results", {})
    g = latest_eval.get("gating", {})
    reasons = list(g.get("reasons") or [])

    illegal_or_timeout = any(
        ("illegal" in r.lower()) or ("timeout" in r.lower()) or ("runtime errors" in r.lower())
        for r in reasons
    )
    if illegal_or_timeout:
        return "稳定性瓶颈：存在非法动作、超时或运行时异常。"

    wr_random = float(cand.get("random", {}).get("win_rate", 0.0))
    wr_nega = float(cand.get("negamax", {}).get("win_rate", 0.0))
    wr_mcts = float(cand.get("mcts_lite", {}).get("win_rate", 0.0))
    wr_prev = float(cand.get("previous_best", {}).get("win_rate", 1.0))

    if wr_random >= 0.8 and wr_nega <= 0.55:
        return "强度瓶颈：random 高但 negamax 低，疑似弱对手过拟合。"
    if wr_prev < 0.55:
        return "退化瓶颈：对 previous best 优势不足。"
    if wr_nega < 0.5:
        return "搜索质量瓶颈：对 negamax 胜率偏低。"
    if wr_mcts < 0.5:
        return "对抗强度瓶颈：对 mcts_lite 胜率偏低。"
    return "暂无明显瓶颈，建议扩大量级继续训练。"


def _write_training_summary(
    summary_path: Path,
    state: Dict[str, Any],
    history: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    recent = history[-5:]
    latest_eval = recent[-1] if recent else None
    bottleneck = _infer_bottleneck(latest_eval)
    recs = _recommend_next_params(args, latest_eval, bottleneck)

    best_ckpt = state.get("best_checkpoint", "")
    latest_ckpt = state.get("latest_checkpoint", "")

    lines = [
        "# Training Summary",
        "",
        f"- Updated at: `{_utc_now_iso()}`",
        f"- Current best checkpoint: `{best_ckpt}`",
        f"- Current latest checkpoint: `{latest_ckpt}`",
        "",
        "## Recent 5 Evaluations",
    ]

    if not recent:
        lines.append("- No evaluation records yet.")
    else:
        for idx, rec in enumerate(recent, start=1):
            cand = rec.get("candidate_results", {})
            def _fmt(op: str) -> str:
                if op not in cand:
                    return "n/a"
                r = cand[op]
                return f"{r['wins']}/{r['losses']}/{r['draws']} ({float(r['win_rate'])*100:.1f}%)"

            lines.extend(
                [
                    f"### Eval #{idx}",
                    f"- Time: `{rec.get('created_at', '')}`",
                    f"- Mode: `{rec.get('label', '')}`",
                    f"- Checkpoint: `{rec.get('checkpoint', '')}`",
                    f"- vs random: `{_fmt('random')}`",
                    f"- vs negamax: `{_fmt('negamax')}`",
                    f"- vs mcts_lite: `{_fmt('mcts_lite')}`",
                    f"- vs previous best: `{_fmt('previous_best')}`",
                    f"- Gating passed: `{rec.get('gating', {}).get('passed', False)}`",
                ]
            )
            warning = rec.get("gating", {}).get("warning")
            if warning:
                lines.append(f"- Warning: {warning}")
            reasons = rec.get("gating", {}).get("reasons") or []
            if reasons:
                lines.append(f"- Reasons: {'; '.join(str(x) for x in reasons)}")
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

    if best_ckpt.exists():
        candidate_ckpt = best_ckpt
    elif latest_ckpt.exists():
        candidate_ckpt = latest_ckpt
    else:
        raise RuntimeError("Training finished but neither latest.pt nor best.pt exists.")

    eval_payload = None
    opponents = [x.strip() for x in str(args.opponents).split(",") if x.strip()]
    try:
        eval_payload = _evaluate_checkpoint_matrix(
            checkpoint_path=candidate_ckpt,
            previous_best_checkpoint=prev_best_snapshot,
            opponents=opponents,
            games=int(args.eval_games),
            simulations=int(args.simulations),
            device=str(args.device),
            seed=int(args.seed) + 77,
            logs_dir=paths.logs,
            label="selfplay_candidate",
        )
        history = _append_eval_history(paths.eval_history, eval_payload)
        print(
            f"[pipeline] selfplay eval: passed={eval_payload['gating']['passed']} "
            f"reasons={eval_payload['gating']['reasons']}"
        )
    except Exception as exc:
        print(f"[pipeline] WARN: selfplay eval failed, checkpoint kept. err={exc}")
        traceback.print_exc()
        history = _load_json(paths.eval_history, [])

    if eval_payload is not None:
        can_promote = bool(eval_payload["gating"]["passed"])
        if can_promote and latest_ckpt.exists() and candidate_ckpt == latest_ckpt:
            shutil.copy2(latest_ckpt, best_ckpt)
            candidate_ckpt = best_ckpt
            print("[pipeline] selfplay: promoted latest -> best by pipeline gating.")
        elif not can_promote:
            print("[pipeline] selfplay: candidate failed gating, best checkpoint unchanged.")

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
    if isinstance(train_state, dict) and "last_iteration" in train_state:
        try:
            last_iteration = int(train_state.get("last_iteration"))
        except Exception:
            last_iteration = None

    extra_meta = _build_context_metadata(
        args,
        git_info,
        teacher_path,
        eval_payload,
        iteration=last_iteration,
    )
    _update_checkpoint_metadata(latest_ckpt, extra_meta)
    _update_checkpoint_metadata(best_ckpt, extra_meta)

    state.update(
        {
            "latest_checkpoint": str(latest_ckpt.resolve()) if latest_ckpt.exists() else "",
            "best_checkpoint": str(best_ckpt.resolve()) if best_ckpt.exists() else "",
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
