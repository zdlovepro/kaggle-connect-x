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
    games_for_opponent,
    play_match,
    resolve_eval_config,
)

COMPOSITE_WEIGHTS: Dict[str, float] = {
    "random": 0.03,
    "negamax": 0.42,
    "mcts_lite": 0.30,
    "previous_best": 0.25,
}
COMPOSITE_KEY_METRICS: Tuple[str, ...] = ("negamax", "mcts_lite", "previous_best")
PREVIOUS_BEST_SIDE_BIAS_THRESHOLD = 0.40


def _ts_now() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def compute_composite(candidate_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    components: Dict[str, Dict[str, Any]] = {}
    included_weight_sum = 0.0

    for metric, weight in COMPOSITE_WEIGHTS.items():
        rec = candidate_results.get(metric)
        present = rec is not None
        reliable = bool(rec.get("reliable", True)) if present else False
        win_rate = float(rec.get("win_rate", 0.0)) if present else None
        included = bool(present and reliable)
        if included:
            included_weight_sum += float(weight)
        components[metric] = {
            "configured_weight": float(weight),
            "normalized_weight": 0.0,
            "present": bool(present),
            "reliable": bool(reliable),
            "included": bool(included),
            "win_rate": win_rate,
            "status": (
                "included"
                if included
                else ("unreliable" if present else "missing")
            ),
        }

    score = None
    if included_weight_sum > 0:
        score = 0.0
        for metric, comp in components.items():
            if not comp["included"]:
                continue
            norm_w = float(comp["configured_weight"]) / float(included_weight_sum)
            comp["normalized_weight"] = norm_w
            score += norm_w * float(comp["win_rate"])

    all_key_unreliable = all(
        (metric not in candidate_results) or (not bool(candidate_results[metric].get("reliable", True)))
        for metric in COMPOSITE_KEY_METRICS
    )

    return {
        "score": score,
        "weights": dict(COMPOSITE_WEIGHTS),
        "components": components,
        "included_weight_sum": float(included_weight_sum),
        "all_key_metrics_unreliable": bool(all_key_unreliable),
        "key_metrics": list(COMPOSITE_KEY_METRICS),
    }


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
    cand_mcts = candidate_results.get("mcts_lite")
    prev_negamax = None if not previous_best_results else previous_best_results.get("negamax")
    prev_mcts = None if not previous_best_results else previous_best_results.get("mcts_lite")
    cand_prevbest = candidate_results.get("previous_best")
    composite_info = compute_composite(candidate_results)

    if bool(composite_info.get("all_key_metrics_unreliable", False)):
        passed = False
        reasons.append("all key metrics (negamax/mcts_lite/previous_best) are unreliable or missing")

    gating_relevant_opponents = {"negamax", "mcts_lite", "previous_best"}
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

    if cand_mcts is None:
        passed = False
        reasons.append("missing mcts_lite evaluation in candidate_results")
    elif not bool(cand_mcts.get("reliable", True)):
        passed = False
        reasons.append("mcts_lite eval unreliable; cannot use for gating")
    elif prev_mcts is not None:
        if not bool(prev_mcts.get("reliable", True)):
            passed = False
            reasons.append("previous-best mcts_lite eval unreliable; cannot compare")
        elif float(cand_mcts["win_rate"]) < float(prev_mcts["win_rate"]):
            passed = False
            reasons.append(
                "vs mcts_lite win rate lower than previous best "
                f"({cand_mcts['win_rate']:.3f} < {prev_mcts['win_rate']:.3f})"
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

    if cand_prevbest is not None:
        prev_side_bias = float(cand_prevbest.get("side_bias", 0.0))
        if prev_side_bias > PREVIOUS_BEST_SIDE_BIAS_THRESHOLD:
            passed = False
            reasons.append(
                "previous_best side bias too high "
                f"({prev_side_bias:.3f} > {PREVIOUS_BEST_SIDE_BIAS_THRESHOLD:.3f}); unstable, require more games"
            )
            prev_wr = float(cand_prevbest.get("win_rate", 0.0))
            if abs(prev_wr - 0.5) <= 0.05:
                reasons.append(
                    "vs previous_best appears 50/50 by total WR but has strong first/second-player bias"
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


def _previous_best_side_bias_status(candidate_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rec = candidate_results.get("previous_best")
    if rec is None:
        return {
            "available": False,
            "side_bias_warning": False,
            "unstable_previous_best_eval": False,
            "side_bias": None,
            "message": "previous_best matchup unavailable",
        }
    side_bias = float(rec.get("side_bias", 0.0))
    unstable = bool(side_bias > PREVIOUS_BEST_SIDE_BIAS_THRESHOLD)
    return {
        "available": True,
        "side_bias_warning": bool(unstable),
        "unstable_previous_best_eval": bool(unstable),
        "side_bias": side_bias,
        "message": (
            f"previous_best side bias high ({side_bias:.3f})"
            if unstable
            else f"previous_best side bias acceptable ({side_bias:.3f})"
        ),
    }


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
    parser.add_argument(
        "--eval-profile",
        type=str,
        choices=("quick", "strong_local", "kaggle_like"),
        default="kaggle_like",
    )
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--eval-games-random", type=int, default=None)
    parser.add_argument("--eval-games-negamax", type=int, default=None)
    parser.add_argument("--eval-games-mcts-lite", type=int, default=None)
    parser.add_argument("--eval-games-previous-best", type=int, default=None)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--candidate-timeout-ms", type=float, default=None)
    parser.add_argument("--opponent-timeout-ms", type=float, default=None)
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

    eval_cfg = resolve_eval_config(
        eval_profile=str(args.eval_profile),
        base_games=int(args.games),
        candidate_timeout_ms=args.candidate_timeout_ms,
        opponent_timeout_ms=args.opponent_timeout_ms,
        eval_games_random=args.eval_games_random,
        eval_games_negamax=args.eval_games_negamax,
        eval_games_mcts_lite=args.eval_games_mcts_lite,
        eval_games_previous_best=args.eval_games_previous_best,
    )
    eval_profile = str(eval_cfg["eval_profile"])
    candidate_timeout_ms = float(eval_cfg["candidate_timeout_ms"])
    opponent_timeout_ms = float(eval_cfg["opponent_timeout_ms"])

    candidate = create_agent(
        args.candidate_agent,
        checkpoint=args.checkpoint,
        previous_best_checkpoint=args.previous_best_checkpoint,
        simulations=int(args.simulations),
        device=args.device,
        seed=int(args.seed),
        time_budget_ms=candidate_timeout_ms,
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
            time_budget_ms=opponent_timeout_ms,
        )
        opponent_agents.append((opp_spec, opp))

    candidate_results: Dict[str, Dict[str, Any]] = {}
    for i, (opp_spec, opp_agent) in enumerate(opponent_agents):
        games_this = games_for_opponent(eval_cfg, opp_spec)
        result = play_match(
            candidate,
            opp_agent,
            num_games=int(games_this),
            swap_sides=True,
            seed=int(args.seed) + 1000 + i,
            candidate_timeout_ms=candidate_timeout_ms,
            opponent_timeout_ms=opponent_timeout_ms,
            eval_profile=eval_profile,
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
            time_budget_ms=opponent_timeout_ms,
        )
        r_prev = play_match(
            candidate,
            prev_best_opp,
            num_games=max(20, games_for_opponent(eval_cfg, "previous_best")),
            swap_sides=True,
            seed=int(args.seed) + 2000,
            candidate_timeout_ms=candidate_timeout_ms,
            opponent_timeout_ms=opponent_timeout_ms,
            eval_profile=eval_profile,
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
            time_budget_ms=candidate_timeout_ms,
        )
        for i, (opp_spec, opp_agent) in enumerate(opponent_agents):
            if opp_spec == "previous_best":
                continue
            games_this = games_for_opponent(eval_cfg, opp_spec)
            rr = play_match(
                previous_best,
                opp_agent,
                num_games=int(games_this),
                swap_sides=True,
                seed=int(args.seed) + 4000 + i,
                candidate_timeout_ms=candidate_timeout_ms,
                opponent_timeout_ms=opponent_timeout_ms,
                eval_profile=eval_profile,
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
    composite = compute_composite(candidate_results)
    warning = _warn_random_overfit(candidate_results)
    reliability_warnings = _collect_reliability_warnings(
        candidate_results,
        max_opponent_timeout_rate=float(args.max_opponent_timeout_rate),
    )
    prev_bias_status = _previous_best_side_bias_status(candidate_results)
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
    print(
        f"eval_profile={eval_profile} "
        f"candidate_timeout_ms={candidate_timeout_ms:.0f} "
        f"opponent_timeout_ms={opponent_timeout_ms:.0f}"
    )
    print(f"games_by_opponent={eval_cfg.get('games_by_opponent', {})}")
    print(table_text)

    print("\nGating")
    print(f"passed={passed}")
    if composite.get("score") is None:
        print("composite=unavailable (no reliable component)")
    else:
        print(f"composite={float(composite['score']):.4f}")
    for reason in gate_reasons:
        print(f"- {reason}")
    if warning:
        print(f"\nWARNING: {warning}")
    for w in reliability_warnings:
        print(w)
    if prev_bias_status.get("side_bias_warning", False):
        print(f"[eval][warning] {prev_bias_status.get('message')}")
    if eval_profile == "strong_local":
        print(
            "[eval][note] strong_local is diagnostic only and should not be treated as Kaggle-equivalent."
        )

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
            "games_by_opponent": dict(eval_cfg.get("games_by_opponent", {})),
            "simulations": int(args.simulations),
            "device": args.device,
            "seed": int(args.seed),
            "eval_profile": eval_profile,
            "candidate_timeout_ms": candidate_timeout_ms,
            "opponent_timeout_ms": opponent_timeout_ms,
            "max_opponent_timeout_rate": float(args.max_opponent_timeout_rate),
            "max_candidate_timeout_rate": float(args.max_candidate_timeout_rate),
            "timeout_result_policy": str(args.timeout_result_policy),
            "previous_best_checkpoint": args.previous_best_checkpoint,
        },
        "reliable": bool(overall_reliable),
        "unreliable_reasons": overall_unreliable_reasons,
        "previous_best_available": bool(prev_bias_status.get("available", False)),
        "side_bias_warning": bool(prev_bias_status.get("side_bias_warning", False)),
        "unstable_previous_best_eval": bool(prev_bias_status.get("unstable_previous_best_eval", False)),
        "candidate_results": candidate_results,
        "previous_best_results": previous_best_results,
        "composite": composite,
        "gating": {
            "passed": bool(passed),
            "reasons": gate_reasons,
            "warning": warning,
            "reliability_warnings": reliability_warnings,
        },
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    composite_text = "unavailable"
    if composite.get("score") is not None:
        composite_text = f"{float(composite['score']):.4f}"

    md_lines = [
        "# Evaluation Report",
        "",
        f"- Timestamp: `{payload['created_at']}`",
        f"- Evaluator backend: `{EVALUATOR_BACKEND}`",
        f"- Candidate: `{args.candidate_agent}`",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Games per matchup: `{int(args.games)}`",
        f"- Eval profile: `{eval_profile}`",
        f"- Candidate timeout ms: `{candidate_timeout_ms:.0f}`",
        f"- Opponent timeout ms: `{opponent_timeout_ms:.0f}`",
        f"- Simulations: `{int(args.simulations)}`",
        f"- previous_best_available: `{bool(prev_bias_status.get('available', False))}`",
        f"- side_bias_warning: `{bool(prev_bias_status.get('side_bias_warning', False))}`",
        f"- unstable_previous_best_eval: `{bool(prev_bias_status.get('unstable_previous_best_eval', False))}`",
        "",
        "## Matrix",
        _format_markdown_table(rows),
        "",
        "## Gating",
        f"- Passed: `{passed}`",
        f"- Composite: `{composite_text}`",
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
    if eval_profile == "strong_local":
        md_lines.append("")
        md_lines.append(
            "- strong_local is diagnostic only and should not be treated as Kaggle-equivalent."
        )
    md_lines.append("")
    md_lines.append(f"- JSON log: `{json_path}`")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nSaved JSON: {json_path.resolve()}")
    print(f"Saved Markdown: {md_path.resolve()}")


if __name__ == "__main__":
    _main()
