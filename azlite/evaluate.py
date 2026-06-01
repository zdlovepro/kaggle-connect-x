"""Robust evaluation matrix for ConnectX AlphaZero-lite.

This module intentionally keeps CLI/reporting/gating logic, while delegating
actual agent construction and match execution to `azlite.eval_core`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from azlite.eval_core import (
    EVALUATOR_BACKEND,
    KAGGLE_ACT_TIMEOUT_SEC,
    KAGGLE_P95_TIMEOUT_SEC,
    UnifiedAgent,
    create_agent,
    play_match,
)


def _ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _aggregate_candidate_metrics(candidate_results: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    total_illegal = 0
    total_timeout = 0
    total_errors = 0
    total_step_time_sum = 0.0
    total_step_count = 0
    max_p95 = 0.0
    weighted_steps = 0.0
    total_games = 0

    for _, r in candidate_results.items():
        total_illegal += int(r.get("invalid_actions", {}).get("candidate", r["illegal_actions"]["agent_a"]))
        total_timeout += int(r.get("candidate_timeouts", r["timeouts"]["agent_a"]))
        total_errors += int(r["errors"]["agent_a"])
        total_step_time_sum += float(r["step_time_sum_sec"]["agent_a"])
        total_step_count += int(r["step_count"]["agent_a"])
        max_p95 = max(max_p95, float(r["p95_step_time_sec"]["agent_a"]))
        weighted_steps += float(r["avg_steps"]) * int(r["num_games"])
        total_games += int(r["num_games"])

    avg_step = total_step_time_sum / max(1, total_step_count)
    avg_steps = weighted_steps / max(1, total_games)
    return {
        "total_illegal_agent_a": int(total_illegal),
        "total_timeout_agent_a": int(total_timeout),
        "total_errors_agent_a": int(total_errors),
        "avg_step_time_sec_agent_a": avg_step,
        "p95_step_time_sec_agent_a": max_p95,
        "avg_steps": avg_steps,
    }


def should_promote_candidate(
    candidate_results,
    previous_best_results,
    max_opponent_timeout_rate: float = 0.05,
    max_candidate_timeout_rate: float = 0.01,
):
    """Gating rules for checkpoint promotion.

    Rules:
      1. vs negamax win-rate must be >= previous best.
      2. vs previous_best must be >= 55%.
      3. random is sanity check only (not gating).
      4. illegal actions must be 0.
      5. avg per-step time < Kaggle limit.
      6. p95 step time acceptable.
      7. if opponent timeouts are high, matchup win-rate is not reliable.
    """
    passed = True
    reasons: List[str] = []

    cand_negamax = candidate_results.get("negamax")
    prev_negamax = None if not previous_best_results else previous_best_results.get("negamax")
    cand_prevbest = candidate_results.get("previous_best")

    gating_relevant_opponents = {"negamax", "previous_best"}
    for opp, r in candidate_results.items():
        games = max(1, int(r.get("num_games", r.get("games", 0)) or 0))
        cand_timeout_rate = float(
            r.get("candidate_timeout_rate", float(r.get("candidate_timeouts", r["timeouts"]["agent_a"])) / games)
        )
        opp_timeout_rate = float(
            r.get("opponent_timeout_rate", float(r.get("opponent_timeouts", r["timeouts"]["agent_b"])) / games)
        )
        if cand_timeout_rate > float(max_candidate_timeout_rate):
            passed = False
            reasons.append(
                f"{opp}: candidate_timeout_rate {cand_timeout_rate:.3f} "
                f"> {float(max_candidate_timeout_rate):.3f}"
            )
        if opp_timeout_rate > float(max_opponent_timeout_rate):
            if opp in gating_relevant_opponents:
                passed = False
                reasons.append(
                    f"{opp}: opponent_timeout_rate {opp_timeout_rate:.3f} "
                    f"> {float(max_opponent_timeout_rate):.3f}; win_rate excluded from gating"
                )

    agg = _aggregate_candidate_metrics(candidate_results)
    if agg["total_illegal_agent_a"] != 0:
        passed = False
        reasons.append(f"illegal actions > 0 ({agg['total_illegal_agent_a']})")

    if agg["total_errors_agent_a"] != 0:
        passed = False
        reasons.append(f"agent runtime errors > 0 ({agg['total_errors_agent_a']})")

    if agg["avg_step_time_sec_agent_a"] >= KAGGLE_ACT_TIMEOUT_SEC:
        passed = False
        reasons.append(
            f"avg step time {agg['avg_step_time_sec_agent_a']:.4f}s "
            f">= Kaggle limit {KAGGLE_ACT_TIMEOUT_SEC:.2f}s"
        )

    if agg["p95_step_time_sec_agent_a"] >= KAGGLE_P95_TIMEOUT_SEC:
        passed = False
        reasons.append(
            f"p95 step time {agg['p95_step_time_sec_agent_a']:.4f}s "
            f">= acceptable {KAGGLE_P95_TIMEOUT_SEC:.2f}s"
        )

    if cand_negamax is None:
        passed = False
        reasons.append("missing negamax evaluation in candidate_results")
    elif prev_negamax is not None:
        if not bool(cand_negamax.get("reliable", True)):
            passed = False
            reasons.append("negamax eval unreliable; cannot use for gating")
        elif float(cand_negamax["win_rate"]) < float(prev_negamax["win_rate"]):
            passed = False
            reasons.append(
                "vs negamax win rate lower than previous best "
                f"({cand_negamax['win_rate']:.3f} < {prev_negamax['win_rate']:.3f})"
            )

    if previous_best_results is not None:
        if cand_prevbest is None:
            passed = False
            reasons.append("missing vs previous_best match in candidate_results")
        elif (not bool(cand_prevbest.get("reliable", True))):
            passed = False
            reasons.append("vs previous_best eval unreliable; cannot use for gating")
        elif float(cand_prevbest["win_rate"]) < 0.55:
            passed = False
            reasons.append(
                "vs previous_best win rate below 55% "
                f"({cand_prevbest['win_rate']:.3f} < 0.550)"
            )

    for opp, r in candidate_results.items():
        if bool(r.get("reliable", True)):
            continue
        if opp not in gating_relevant_opponents:
            continue
        passed = False
        reasons_local = r.get("unreliable_reasons") or []
        warning = "; ".join(str(x) for x in reasons_local) if reasons_local else "unreliable win-rate due opponent instability"
        reasons.append(f"{opp}: {warning}")

    if passed and not reasons:
        reasons.append("all gating checks passed")
    return passed, reasons


def _format_console_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = [
        "Opponent",
        "W/L/D",
        "WR",
        "FP_WR",
        "SP_WR",
        "Invalid(A)",
        "Timeout(A)",
        "Timeout(B)",
        "AvgSteps",
    ]
    col_widths = [max(len(h), 12) for h in headers]

    formatted_rows: List[List[str]] = []
    for r in rows:
        line = [
            str(r["opponent"]),
            f"{r['wins']}/{r['losses']}/{r['draws']}",
            f"{r['win_rate']*100:.1f}%",
            f"{r.get('first_player_win_rate', 0.0)*100:.1f}%",
            f"{r.get('second_player_win_rate', 0.0)*100:.1f}%",
            str(r.get("invalid_actions", {}).get("candidate", r["illegal_actions"]["agent_a"])),
            str(r.get("candidate_timeouts", r["timeouts"]["agent_a"])),
            str(r.get("opponent_timeouts", r["timeouts"]["agent_b"])),
            f"{r['avg_steps']:.2f}",
        ]
        formatted_rows.append(line)
        for i, cell in enumerate(line):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_line(cols: Sequence[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols))

    sep = "-+-".join("-" * w for w in col_widths)
    out = [_fmt_line(headers), sep]
    for line in formatted_rows:
        out.append(_fmt_line(line))
    return "\n".join(out)


def _format_markdown_table(rows: Sequence[Dict[str, Any]]) -> str:
    head = (
        "| Opponent | W/L/D | WR | FP_WR | SP_WR | Invalid(A) | Timeout(A) | Timeout(B) | AvgSteps |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    body_lines = []
    for r in rows:
        body_lines.append(
            "| {opp} | {w}/{l}/{d} | {wr:.1f}% | {fp:.1f}% | {sp:.1f}% | {ill} | {toa} | {tob} | {steps:.2f} |".format(
                opp=r["opponent"],
                w=r["wins"],
                l=r["losses"],
                d=r["draws"],
                wr=r["win_rate"] * 100.0,
                fp=r.get("first_player_win_rate", 0.0) * 100.0,
                sp=r.get("second_player_win_rate", 0.0) * 100.0,
                ill=r.get("invalid_actions", {}).get("candidate", r["illegal_actions"]["agent_a"]),
                toa=r.get("candidate_timeouts", r["timeouts"]["agent_a"]),
                tob=r.get("opponent_timeouts", r["timeouts"]["agent_b"]),
                steps=r["avg_steps"],
            )
        )
    return head + ("\n" + "\n".join(body_lines) if body_lines else "")


def _parse_agent_list(spec: str) -> List[str]:
    return [s.strip() for s in (spec or "").split(",") if s.strip()]


def _warn_random_overfit(candidate_results: Dict[str, Dict[str, Any]]) -> Optional[str]:
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


def _collect_reliability_warnings(
    candidate_results: Dict[str, Dict[str, Any]],
    max_opponent_timeout_rate: float,
) -> List[str]:
    warnings: List[str] = []
    for opp, r in candidate_results.items():
        timeout_rate = float(r.get("opponent_timeout_rate", 0.0))
        if timeout_rate <= float(max_opponent_timeout_rate):
            continue
        warnings.append(
            f"[eval][warning] opponent={opp} timeout_rate={timeout_rate*100.0:.1f}%, "
            "win_rate is unreliable and excluded from gating."
        )
    return warnings


def _main() -> None:
    parser = argparse.ArgumentParser(description="Robust evaluation matrix for AlphaZero-lite")
    parser.add_argument("--checkpoint", type=str, default=None, help="Candidate checkpoint path")
    parser.add_argument(
        "--candidate-agent",
        type=str,
        default="checkpoint_puct",
        help="Candidate agent spec (checkpoint_puct/checkpoint_policy/original/mcts_lite/...)",
    )
    parser.add_argument(
        "--opponents",
        type=str,
        default="random,negamax,mcts_lite",
        help="Comma-separated opponents specs",
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-opponent-timeout-rate", type=float, default=0.05)
    parser.add_argument("--max-candidate-timeout-rate", type=float, default=0.01)
    parser.add_argument(
        "--timeout-result-policy",
        type=str,
        choices=("loss", "exclude", "fail_eval"),
        default="fail_eval",
    )
    parser.add_argument(
        "--previous-best-checkpoint",
        type=str,
        default=None,
        help="Optional previous-best checkpoint for comparison and gating",
    )
    parser.add_argument("--logs-dir", type=str, default="logs")
    args = parser.parse_args()

    candidate = create_agent(
        args.candidate_agent,
        checkpoint=args.checkpoint,
        previous_best_checkpoint=args.previous_best_checkpoint,
        simulations=int(args.simulations),
        device=args.device,
        seed=int(args.seed),
    )
    opponents = _parse_agent_list(args.opponents)
    if not opponents:
        raise ValueError("--opponents cannot be empty")

    opponent_agents: List[Tuple[str, UnifiedAgent]] = []
    for i, opp_spec in enumerate(opponents):
        opp = create_agent(
            opp_spec,
            checkpoint=args.checkpoint,
            previous_best_checkpoint=args.previous_best_checkpoint,
            simulations=int(args.simulations),
            device=args.device,
            seed=int(args.seed) + i + 1,
        )
        opponent_agents.append((opp_spec, opp))

    candidate_results: Dict[str, Dict[str, Any]] = {}
    for i, (opp_spec, opp_agent) in enumerate(opponent_agents):
        result = play_match(
            candidate,
            opp_agent,
            num_games=int(args.games),
            swap_sides=True,
            seed=int(args.seed) + 1000 + i,
            max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
            max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
            timeout_result_policy=str(args.timeout_result_policy),
        )
        result["opponent"] = opp_spec
        candidate_results[opp_spec] = result

    # Always evaluate candidate vs previous_best when checkpoint provided.
    if args.previous_best_checkpoint:
        prev_best_opp = create_agent(
            "previous_best",
            checkpoint=args.checkpoint,
            previous_best_checkpoint=args.previous_best_checkpoint,
            simulations=int(args.simulations),
            device=args.device,
            seed=int(args.seed) + 9999,
        )
        r_prev = play_match(
            candidate,
            prev_best_opp,
            num_games=max(20, int(args.games)),
            swap_sides=True,
            seed=int(args.seed) + 2000,
            max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
            max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
            timeout_result_policy=str(args.timeout_result_policy),
        )
        r_prev["opponent"] = "previous_best"
        candidate_results["previous_best"] = r_prev

    previous_best_results = None
    if args.previous_best_checkpoint:
        previous_best_results = {}
        previous_best = create_agent(
            "previous_best",
            checkpoint=args.checkpoint,
            previous_best_checkpoint=args.previous_best_checkpoint,
            simulations=int(args.simulations),
            device=args.device,
            seed=int(args.seed) + 3000,
        )
        for i, (opp_spec, opp_agent) in enumerate(opponent_agents):
            if opp_spec == "previous_best":
                continue
            rr = play_match(
                previous_best,
                opp_agent,
                num_games=int(args.games),
                swap_sides=True,
                seed=int(args.seed) + 4000 + i,
                max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
                max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
                timeout_result_policy=str(args.timeout_result_policy),
            )
            rr["opponent"] = opp_spec
            previous_best_results[opp_spec] = rr

    passed, gate_reasons = should_promote_candidate(
        candidate_results,
        previous_best_results,
        max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
        max_candidate_timeout_rate=float(args.max_candidate_timeout_rate),
    )
    warning = _warn_random_overfit(candidate_results)
    reliability_warnings = _collect_reliability_warnings(
        candidate_results,
        max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
    )
    overall_reliable = True
    overall_unreliable_reasons: List[str] = []
    for opp, r in candidate_results.items():
        if bool(r.get("reliable", True)):
            continue
        overall_reliable = False
        for reason in (r.get("unreliable_reasons") or []):
            overall_unreliable_reasons.append(f"{opp}: {reason}")

    rows = [candidate_results[k] for k in candidate_results.keys()]
    table_text = _format_console_table(rows)
    print("\nEvaluation Matrix")
    print(f"evaluator_backend={EVALUATOR_BACKEND}")
    print(table_text)

    print("\nGating")
    print(f"passed={passed}")
    for reason in gate_reasons:
        print(f"- {reason}")
    if warning:
        print(f"\nWARNING: {warning}")
    for w in reliability_warnings:
        print(w)

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts_now()
    json_path = logs_dir / f"eval_{ts}.json"
    md_path = logs_dir / "eval_latest.md"

    payload = {
        "created_at": datetime.now().isoformat(),
        "evaluator_backend": EVALUATOR_BACKEND,
        "config": {
            "candidate_agent": args.candidate_agent,
            "candidate_checkpoint": args.checkpoint,
            "checkpoint": args.checkpoint,
            "opponents": opponents,
            "games": int(args.games),
            "simulations": int(args.simulations),
            "device": args.device,
            "seed": int(args.seed),
            "max_opponent_timeout_rate": float(args.max_opponent_timeout_rate),
            "max_candidate_timeout_rate": float(args.max_candidate_timeout_rate),
            "timeout_result_policy": str(args.timeout_result_policy),
            "previous_best_checkpoint": args.previous_best_checkpoint,
        },
        "reliable": bool(overall_reliable),
        "unreliable_reasons": overall_unreliable_reasons,
        "candidate_results": candidate_results,
        "previous_best_results": previous_best_results,
        "gating": {
            "passed": bool(passed),
            "reasons": gate_reasons,
            "warning": warning,
            "reliability_warnings": reliability_warnings,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Evaluation Report",
        "",
        f"- Timestamp: `{payload['created_at']}`",
        f"- Evaluator backend: `{EVALUATOR_BACKEND}`",
        f"- Candidate: `{args.candidate_agent}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Games per matchup: `{int(args.games)}`",
        f"- Simulations: `{int(args.simulations)}`",
        "",
        "## Matrix",
        _format_markdown_table(rows),
        "",
        "## Gating",
        f"- Passed: `{passed}`",
    ]
    for reason in gate_reasons:
        md_lines.append(f"- {reason}")
    if warning:
        md_lines.append("")
        md_lines.append("## Warning")
        md_lines.append(
            "> random is not a reliable gating metric; model may be overfitting weak play or relying on tactical shortcuts only."
        )
    if reliability_warnings:
        md_lines.append("")
        md_lines.append("## Reliability Warnings")
        for w in reliability_warnings:
            md_lines.append(f"- {w}")
    md_lines.append("")
    md_lines.append(f"- JSON log: `{json_path}`")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nSaved JSON: {json_path.resolve()}")
    print(f"Saved Markdown: {md_path.resolve()}")


if __name__ == "__main__":
    _main()
