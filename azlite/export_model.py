"""Export AlphaZero-lite checkpoint to a Python weight module.

Primary output:
  agents/azlite_weights.py

The exported payload is designed for single-file Kaggle submission builds:
  - architecture metadata
  - board metadata (rows/columns/inarow/channels)
  - normalization metadata
  - packed tensors (base64 + optional zlib)
"""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = ROOT / "agents" / "azlite_weights.py"

REQUIRED_TENSORS = (
    "features.0.weight",
    "features.0.bias",
    "features.2.weight",
    "features.2.bias",
    "features.4.weight",
    "features.4.bias",
    "backbone.1.weight",
    "backbone.1.bias",
    "policy_head.weight",
    "policy_head.bias",
    "value_head.0.weight",
    "value_head.0.bias",
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pack_array(arr: np.ndarray, compress: bool = True) -> Dict[str, Any]:
    np_arr = np.asarray(arr)
    raw = np_arr.tobytes(order="C")
    encoding = "base64"
    if compress:
        raw = zlib.compress(raw, level=3)
        encoding = "base64+zlib"
    b64 = base64.b64encode(raw).decode("ascii")
    return {
        "shape": [int(v) for v in np_arr.shape],
        "dtype": str(np_arr.dtype),
        "encoding": encoding,
        "data": b64,
    }


def unpack_array(spec: Mapping[str, Any], out_dtype: np.dtype | None = np.float32) -> np.ndarray:
    raw = base64.b64decode(str(spec["data"]).encode("ascii"))
    encoding = str(spec.get("encoding", "base64"))
    if "zlib" in encoding:
        raw = zlib.decompress(raw)
    arr = np.frombuffer(raw, dtype=np.dtype(spec.get("dtype", "float32")))
    arr = arr.reshape(tuple(int(v) for v in spec["shape"]))
    if out_dtype is None:
        return arr
    return arr.astype(out_dtype, copy=False)


def _load_checkpoint_tensors(checkpoint_path: Path, out_dtype: np.dtype) -> tuple[Dict[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local runtime
        raise RuntimeError("PyTorch is required to export checkpoint weights.") from exc

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Checkpoint missing model_state_dict")

    missing = [k for k in REQUIRED_TENSORS if k not in state_dict]
    if missing:
        raise RuntimeError(f"Checkpoint missing required tensors: {missing}")

    tensors: Dict[str, np.ndarray] = {}
    for key in REQUIRED_TENSORS:
        t = state_dict[key]
        arr = t.detach().cpu().numpy().astype(out_dtype, copy=False)
        tensors[key] = arr

    meta = ckpt.get("metadata") or {}
    model_cfg = ckpt.get("model_config") or {}
    return tensors, dict(meta), dict(model_cfg)


def _make_export_payload(
    tensors: Mapping[str, np.ndarray],
    checkpoint_path: Path,
    checkpoint_meta: Mapping[str, Any],
    model_cfg: Mapping[str, Any],
    rows: int,
    columns: int,
    inarow: int,
    runtime_mode: str,
    compress: bool,
) -> Dict[str, Any]:
    in_channels = int(model_cfg.get("in_channels", 3))
    packed = {k: _pack_array(v, compress=compress) for k, v in tensors.items()}
    payload = {
        "version": 1,
        "arch": "connectx_conv3_fc128",
        "runtime_mode": runtime_mode,
        "channels": in_channels,
        "rows": int(rows),
        "columns": int(columns),
        "inarow": int(inarow),
        "normalization": {
            "type": "binary_current_player_planes",
            "include_legal_channel": bool(in_channels >= 3),
            "value_range": [0.0, 1.0],
            "dtype": "float32",
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "exported_at": _utc_now_iso(),
            "metadata": dict(checkpoint_meta),
            "model_config": dict(model_cfg),
        },
        "tensors": packed,
    }
    return payload


def _payload_to_module_text(payload: Mapping[str, Any]) -> str:
    header = (
        '"""Auto-generated AlphaZero-lite export.\n'
        "Do not edit by hand; regenerate with `python -m azlite.export_model`.\n"
        '"""\n\n'
    )
    return header + f"AZLITE_EXPORT = {repr(dict(payload))}\n"


def save_export_module(payload: Mapping[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_payload_to_module_text(payload), encoding="utf-8")
    return output_path


def export_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path = DEFAULT_OUTPUT,
    rows: int = 6,
    columns: int = 7,
    inarow: int = 4,
    runtime_mode: str = "numpy",
    dtype: str = "float32",
    compress: bool = True,
) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    out_path = Path(output_path)

    np_dtype = np.dtype(dtype)
    tensors, ckpt_meta, model_cfg = _load_checkpoint_tensors(ckpt_path, np_dtype)
    payload = _make_export_payload(
        tensors=tensors,
        checkpoint_path=ckpt_path.resolve(),
        checkpoint_meta=ckpt_meta,
        model_cfg=model_cfg,
        rows=int(rows),
        columns=int(columns),
        inarow=int(inarow),
        runtime_mode=str(runtime_mode),
        compress=bool(compress),
    )
    save_export_module(payload, out_path)
    return payload


def _estimate_size_mb(payload: Mapping[str, Any]) -> float:
    data = json.dumps(payload, ensure_ascii=False)
    return len(data.encode("utf-8")) / (1024.0 * 1024.0)


def _main() -> None:
    parser = argparse.ArgumentParser(description="Export AlphaZero-lite checkpoint to Python module")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (e.g. checkpoints/best.pt)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output module path, e.g. agents/azlite_weights.py")
    parser.add_argument("--mode", choices=("numpy", "torch"), default="numpy", help="Preferred runtime mode for submission")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--columns", type=int, default=7)
    parser.add_argument("--inarow", type=int, default=4)
    parser.add_argument("--no-compress", action="store_true", help="Disable zlib compression for tensor blobs")
    args = parser.parse_args()

    payload = export_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        rows=int(args.rows),
        columns=int(args.columns),
        inarow=int(args.inarow),
        runtime_mode=str(args.mode),
        dtype=str(args.dtype),
        compress=not args.no_compress,
    )
    size_mb = _estimate_size_mb(payload)
    print(f"[export_model] output={Path(args.output).resolve()}")
    print(
        f"[export_model] arch={payload['arch']} channels={payload['channels']} "
        f"rows={payload['rows']} cols={payload['columns']} inarow={payload['inarow']}"
    )
    print(f"[export_model] runtime_mode={payload['runtime_mode']} approx_payload_size={size_mb:.2f} MB")


if __name__ == "__main__":
    _main()

