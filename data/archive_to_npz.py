#!/usr/bin/env python3
"""
Convert archived Kaggle ConnectX high-score data into azlite-compatible NPZ.

This script is designed for the archive bundle currently stored at:
  data/archive.zip

The bundle contains:
  - train-all-*.pkl: submission_id -> [episode_id, ...]
  - episodes.tar.xz: archived episode files (sometimes info-only, sometimes
    replay + info depending on how the archive was created)

Output format:
  - states:   (N, C, 6, 7)
  - policies: (N, 7) one-hot from the archived player's chosen move
  - values:   (N,) final reward from that player's perspective in {-1,0,1}
  - metadata: JSON string

Important:
  The current archive.zip in this repository appears to include *_info.json
  metadata but not the replay JSON files needed to reconstruct board states.
  In that case this script will write a missing-replay manifest and exit with
  a clear explanation instead of failing silently.
"""

from __future__ import annotations

import argparse
import json
import pickle
import tarfile
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path, PurePosixPath
import sys
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from azlite.board import DEFAULT_COLUMNS, DEFAULT_ROWS, is_legal_move, to_tensor  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reward_to_value(reward: object) -> float:
    try:
        value = float(reward)
    except (TypeError, ValueError):
        return 0.0
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _resolve_pkl_name(archive_zip: Path, threshold: Optional[int], pkl_name: Optional[str]) -> str:
    if pkl_name:
        return str(pkl_name)
    if threshold is None:
        threshold = 1600
    candidate = f"train-all-{int(threshold)}.pkl"
    with zipfile.ZipFile(archive_zip) as zf:
        if candidate in zf.namelist():
            return candidate
        options = sorted(name for name in zf.namelist() if name.startswith("train-all-") and name.endswith(".pkl"))
    raise FileNotFoundError(
        f"{candidate} not found in {archive_zip}. Available PKLs: {', '.join(options) if options else '(none)'}"
    )


def _load_submission_episode_map(archive_zip: Path, pkl_name: str) -> Dict[int, List[int]]:
    with zipfile.ZipFile(archive_zip) as zf:
        with zf.open(pkl_name) as f:
            raw = pickle.load(f)
    result: Dict[int, List[int]] = {}
    for submission_id, episodes in dict(raw).items():
        try:
            sid = int(submission_id)
        except (TypeError, ValueError):
            continue
        cleaned: List[int] = []
        for episode_id in list(episodes):
            try:
                cleaned.append(int(episode_id))
            except (TypeError, ValueError):
                continue
        if cleaned:
            result[sid] = cleaned
    return result


def _build_episode_targets(submission_to_episodes: Mapping[int, Sequence[int]]) -> Dict[int, set[int]]:
    episode_targets: Dict[int, set[int]] = defaultdict(set)
    for submission_id, episode_ids in submission_to_episodes.items():
        for episode_id in episode_ids:
            episode_targets[int(episode_id)].add(int(submission_id))
    return dict(episode_targets)


def _read_json_file(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    obj = json.loads(text)
    if isinstance(obj, str):
        obj = json.loads(obj)
    return obj


def _load_info_and_replay_from_dir(replay_dir: Path, episode_id: int) -> Tuple[Optional[dict], Optional[object]]:
    candidates = [
        replay_dir / f"{episode_id}_info.json",
        replay_dir / "archive" / f"{episode_id}_info.json",
    ]
    replay_candidates = [
        replay_dir / f"{episode_id}.json",
        replay_dir / "archive" / f"{episode_id}.json",
    ]

    info_obj = None
    replay_obj = None
    for path in candidates:
        if path.exists():
            obj = _read_json_file(path)
            if isinstance(obj, dict):
                info_obj = obj
            break
    for path in replay_candidates:
        if path.exists():
            replay_obj = _read_json_file(path)
            break
    return info_obj, replay_obj


def _stream_tar_assets(
    archive_zip: Path,
    target_episode_ids: set[int],
    need_replays: bool,
    existing_info: MutableMapping[int, dict],
    existing_replays: MutableMapping[int, object],
) -> Tuple[int, int]:
    """
    Stream the inner episodes.tar.xz once and collect only target episodes.

    Returns:
      (infos_found_in_tar, replays_found_in_tar)
    """
    if not target_episode_ids:
        return 0, 0

    infos_found = 0
    replays_found = 0
    with zipfile.ZipFile(archive_zip) as zf:
        if "episodes.tar.xz" not in zf.namelist():
            return 0, 0
        with zf.open("episodes.tar.xz") as xz_stream:
            with tarfile.open(fileobj=xz_stream, mode="r|xz") as tf:
                remaining_info = target_episode_ids.difference(existing_info)
                remaining_replay = target_episode_ids.difference(existing_replays) if need_replays else set()
                for member in tf:
                    if not member.isfile():
                        continue
                    base = PurePosixPath(member.name).name
                    if base.endswith("_info.json"):
                        episode_text = base[: -len("_info.json")]
                        want_info = True
                        want_replay = False
                    elif base.endswith(".json"):
                        episode_text = base[: -len(".json")]
                        want_info = False
                        want_replay = True
                    else:
                        continue

                    try:
                        episode_id = int(episode_text)
                    except ValueError:
                        continue
                    if episode_id not in target_episode_ids:
                        continue
                    if want_info and episode_id not in remaining_info:
                        continue
                    if want_replay and (not need_replays or episode_id not in remaining_replay):
                        continue

                    payload = tf.extractfile(member)
                    if payload is None:
                        continue
                    try:
                        raw = payload.read()
                    finally:
                        payload.close()

                    try:
                        obj = json.loads(raw.decode("utf-8"))
                        if isinstance(obj, str):
                            obj = json.loads(obj)
                    except Exception:
                        continue

                    if want_info:
                        if isinstance(obj, dict):
                            existing_info[episode_id] = obj
                            remaining_info.discard(episode_id)
                            infos_found += 1
                    else:
                        existing_replays[episode_id] = obj
                        remaining_replay.discard(episode_id)
                        replays_found += 1

                    if not remaining_info and (not need_replays or not remaining_replay):
                        break
    return infos_found, replays_found


def _extract_replay_steps(replay_obj: object) -> List[object]:
    if isinstance(replay_obj, dict):
        if isinstance(replay_obj.get("steps"), list):
            return replay_obj["steps"]
        if isinstance(replay_obj.get("environment"), dict) and isinstance(replay_obj["environment"].get("steps"), list):
            return replay_obj["environment"]["steps"]
    if isinstance(replay_obj, list):
        return replay_obj
    return []


def _iter_episode_examples(
    episode_id: int,
    info_obj: dict,
    replay_obj: object,
    allowed_submission_ids: set[int],
    include_legal_channel: bool,
    max_state_repeats: int,
    repeat_counts: MutableMapping[bytes, int],
) -> Iterator[Tuple[np.ndarray, np.ndarray, float]]:
    tracked: Dict[int, Tuple[int, float]] = {}
    for agent in list(info_obj.get("agents") or []):
        try:
            agent_index = int(agent["index"])
            submission_id = int(agent["submissionId"])
        except (KeyError, TypeError, ValueError):
            continue
        if submission_id not in allowed_submission_ids:
            continue
        tracked[agent_index] = (submission_id, _reward_to_value(agent.get("reward")))
    if not tracked:
        return

    seen_turns: set[Tuple[int, int, int]] = set()
    steps = _extract_replay_steps(replay_obj)
    for step_idx, step in enumerate(steps):
        if not isinstance(step, list):
            continue
        for agent_index, entry in enumerate(step):
            if agent_index not in tracked or not isinstance(entry, dict):
                continue

            action = entry.get("action")
            if not isinstance(action, int):
                continue
            if action < 0 or action >= DEFAULT_COLUMNS:
                continue

            obs = entry.get("observation") or {}
            board = obs.get("board")
            if not isinstance(board, list) or len(board) != DEFAULT_ROWS * DEFAULT_COLUMNS:
                continue

            mark = obs.get("mark")
            if mark not in (1, 2):
                mark = agent_index + 1
            if mark not in (1, 2):
                continue

            board_arr = np.asarray(board, dtype=np.int8).reshape(DEFAULT_ROWS, DEFAULT_COLUMNS)
            if not is_legal_move(board_arr, int(action)):
                continue

            turn_key = (episode_id, agent_index, step_idx)
            if turn_key in seen_turns:
                continue
            seen_turns.add(turn_key)

            dedup_key = board_arr.astype(np.int8, copy=False).tobytes() + bytes((int(mark), int(action)))
            prev = int(repeat_counts.get(dedup_key, 0))
            if prev >= max_state_repeats:
                continue
            repeat_counts[dedup_key] = prev + 1

            state = to_tensor(board_arr, int(mark), include_legal_channel=include_legal_channel).astype(np.float32, copy=False)
            policy = np.zeros(DEFAULT_COLUMNS, dtype=np.float32)
            policy[int(action)] = 1.0
            value = tracked[agent_index][1]
            yield state, policy, float(value)


def _write_missing_manifest(path: Path, missing_episode_ids: Sequence[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(str(int(x)) for x in missing_episode_ids)
    path.write_text(text, encoding="utf-8")


def convert_archive(
    archive_zip: Path,
    output: Path,
    *,
    threshold: Optional[int],
    pkl_name: Optional[str],
    replay_dir: Optional[Path],
    include_legal_channel: bool,
    max_state_repeats: int,
    limit_episodes: Optional[int],
    max_samples: Optional[int],
    missing_manifest: Optional[Path],
    dry_run: bool,
    verbose_every: int,
) -> dict:
    resolved_pkl = _resolve_pkl_name(archive_zip, threshold=threshold, pkl_name=pkl_name)
    submission_to_episodes = _load_submission_episode_map(archive_zip, resolved_pkl)
    if not submission_to_episodes:
        raise RuntimeError(f"No submissions found in {resolved_pkl}")

    episode_targets = _build_episode_targets(submission_to_episodes)
    target_episode_ids = sorted(episode_targets)
    if limit_episodes is not None and limit_episodes > 0:
        target_episode_ids = target_episode_ids[: int(limit_episodes)]
        keep = set(target_episode_ids)
        episode_targets = {ep: subs for ep, subs in episode_targets.items() if ep in keep}

    info_by_episode: Dict[int, dict] = {}
    replay_by_episode: Dict[int, object] = {}

    # Prefer extracted directory assets if the user has them.
    if replay_dir is not None:
        for episode_id in target_episode_ids:
            info_obj, replay_obj = _load_info_and_replay_from_dir(replay_dir, episode_id)
            if info_obj is not None:
                info_by_episode[episode_id] = info_obj
            if replay_obj is not None:
                replay_by_episode[episode_id] = replay_obj

    infos_from_tar, replays_from_tar = _stream_tar_assets(
        archive_zip=archive_zip,
        target_episode_ids=set(target_episode_ids),
        need_replays=True,
        existing_info=info_by_episode,
        existing_replays=replay_by_episode,
    )

    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    values: List[float] = []
    repeat_counts: Dict[bytes, int] = {}

    converted_episode_ids: List[int] = []
    missing_replay_ids: List[int] = []
    info_missing_ids: List[int] = []

    for idx, episode_id in enumerate(target_episode_ids, start=1):
        info_obj = info_by_episode.get(episode_id)
        replay_obj = replay_by_episode.get(episode_id)

        if info_obj is None:
            info_missing_ids.append(episode_id)
            continue
        if replay_obj is None:
            missing_replay_ids.append(episode_id)
            continue

        before = len(states)
        for state, policy, value in _iter_episode_examples(
            episode_id=episode_id,
            info_obj=info_obj,
            replay_obj=replay_obj,
            allowed_submission_ids=set(episode_targets[episode_id]),
            include_legal_channel=include_legal_channel,
            max_state_repeats=max_state_repeats,
            repeat_counts=repeat_counts,
        ):
            states.append(state)
            policies.append(policy)
            values.append(value)
            if max_samples is not None and max_samples > 0 and len(states) >= int(max_samples):
                break

        if len(states) > before:
            converted_episode_ids.append(episode_id)

        if max_samples is not None and max_samples > 0 and len(states) >= int(max_samples):
            break
        if verbose_every > 0 and (idx % verbose_every == 0 or idx == len(target_episode_ids)):
            print(
                f"[archive_to_npz] processed episodes={idx}/{len(target_episode_ids)} "
                f"converted={len(converted_episode_ids)} samples={len(states)} "
                f"missing_replay={len(missing_replay_ids)} missing_info={len(info_missing_ids)}"
            )

    metadata = {
        "source": "archive_highscore",
        "archive_zip": str(archive_zip),
        "replay_dir": str(replay_dir) if replay_dir is not None else None,
        "pkl_name": resolved_pkl,
        "threshold": int(threshold) if threshold is not None else None,
        "submissions": sorted(int(x) for x in submission_to_episodes.keys()),
        "episodes_requested": len(target_episode_ids),
        "episodes_with_info": len(info_by_episode),
        "episodes_with_replay": len(replay_by_episode),
        "episodes_converted": len(converted_episode_ids),
        "infos_found_in_tar": int(infos_from_tar),
        "replays_found_in_tar": int(replays_from_tar),
        "samples_saved": len(states),
        "include_legal_channel": bool(include_legal_channel),
        "max_state_repeats": int(max_state_repeats),
        "missing_replays": len(missing_replay_ids),
        "missing_infos": len(info_missing_ids),
        "value_target": "archived_agent_final_reward_sign",
        "policy_target": "archived_agent_action_one_hot",
        "created_at": _utc_now_iso(),
    }

    if missing_manifest is not None and missing_replay_ids:
        _write_missing_manifest(missing_manifest, missing_replay_ids)
        metadata["missing_replay_manifest"] = str(missing_manifest)

    if dry_run:
        return metadata

    if not states:
        reason = (
            "No usable training samples were produced. "
            "This usually means the archive only contains *_info.json metadata but no replay JSON files. "
            "Provide replay files via --replay-dir or rebuild the archive with actual replays."
        )
        raise RuntimeError(reason)

    states_arr = np.stack(states).astype(np.float32, copy=False)
    policies_arr = np.stack(policies).astype(np.float32, copy=False)
    values_arr = np.asarray(values, dtype=np.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output),
        states=states_arr,
        policies=policies_arr,
        values=values_arr,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert archive.zip high-score episodes into azlite-compatible NPZ.",
    )
    parser.add_argument("--archive-zip", type=Path, default=Path("data/archive.zip"))
    parser.add_argument("--replay-dir", type=Path, default=None, help="Optional extracted replay directory.")
    parser.add_argument("--threshold", type=int, default=1600, help="Use train-all-<threshold>.pkl")
    parser.add_argument("--pkl-name", type=str, default=None, help="Explicit train-all-*.pkl name inside archive.zip")
    parser.add_argument("--output", type=Path, default=Path("data/teacher/archive_highscore_1600.npz"))
    parser.add_argument("--missing-manifest", type=Path, default=Path("data/archive_missing_replays_1600.txt"))
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-state-repeats", type=int, default=1)
    parser.add_argument("--include-legal-channel", action="store_true", default=True)
    parser.add_argument("--no-legal-channel", dest="include_legal_channel", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose-every", type=int, default=500)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    archive_zip = Path(args.archive_zip)
    if not archive_zip.exists():
        parser.error(f"archive zip not found: {archive_zip}")

    try:
        metadata = convert_archive(
            archive_zip=archive_zip,
            output=Path(args.output),
            threshold=int(args.threshold) if args.threshold is not None else None,
            pkl_name=args.pkl_name,
            replay_dir=Path(args.replay_dir) if args.replay_dir else None,
            include_legal_channel=bool(args.include_legal_channel),
            max_state_repeats=max(1, int(args.max_state_repeats)),
            limit_episodes=int(args.limit_episodes) if args.limit_episodes else None,
            max_samples=int(args.max_samples) if args.max_samples else None,
            missing_manifest=Path(args.missing_manifest) if args.missing_manifest else None,
            dry_run=bool(args.dry_run),
            verbose_every=max(0, int(args.verbose_every)),
        )
    except Exception as exc:
        print(f"[archive_to_npz][error] {exc}")
        return 1

    print("[archive_to_npz] done")
    for key in [
        "pkl_name",
        "episodes_requested",
        "episodes_with_info",
        "episodes_with_replay",
        "episodes_converted",
        "samples_saved",
        "missing_replays",
        "missing_infos",
        "missing_replay_manifest",
    ]:
        if key in metadata:
            print(f"[archive_to_npz] {key}={metadata[key]}")
    if args.dry_run:
        print("[archive_to_npz] dry-run only; no npz written")
    else:
        print(f"[archive_to_npz] output={Path(args.output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
