#!/usr/bin/env python3
"""Export, install, rebuild, and verify a gated ModernBERT checkpoint."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open


if len(sys.argv) != 2:
    raise SystemExit(f"usage: {Path(sys.argv[0]).name} CHECKPOINT_DIR")

checkpoint = Path(sys.argv[1]).resolve()
repo = Path(__file__).resolve().parents[1]
assets = repo / "assets"
source = checkpoint / "xtrbert.pt"
source_config = checkpoint / "config.json"
source_tokenizer = checkpoint / "tokenizer.json"

for path in (source, source_config, source_tokenizer):
    if not path.is_file():
        raise SystemExit(f"checkpoint is missing {path.name}: {checkpoint}")

config = json.loads(source_config.read_text())
state = torch.load(source, map_location="cpu", weights_only=True)
required = {
    "encoder.embeddings.tok_embeddings.weight",
    "linear.weight",
    "linear.bias",
    "token_gate_norm.weight",
    "token_gate_norm.bias",
    "token_gate.weight",
    "token_gate.bias",
}
missing = sorted(required - state.keys())
if missing:
    raise SystemExit(f"checkpoint is missing required tensors: {', '.join(missing)}")

hidden = state["encoder.embeddings.tok_embeddings.weight"].shape[1]
if state["linear.weight"].shape != (96, hidden):
    raise SystemExit(f"expected a 96d projection, got {tuple(state['linear.weight'].shape)}")
if state["token_gate.weight"].shape != (1, hidden):
    raise SystemExit(f"invalid token gate shape: {tuple(state['token_gate.weight'].shape)}")
if config.get("projection_dim") != 96 or config.get("token_gate") is not True:
    raise SystemExit("config.json must declare projection_dim=96 and token_gate=true")

for name, tensor in state.items():
    if not isinstance(tensor, torch.Tensor):
        raise SystemExit(f"checkpoint entry is not a tensor: {name}")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise SystemExit(f"checkpoint tensor contains NaN or Inf: {name}")
del state


def atomic_copy(source_path: Path, destination: Path) -> None:
    """Copy a file without ever exposing a partial destination."""
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source_path, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


names = (
    "modernbert.safetensors",
    "modernbert.gguf",
    "modernbert-config.json",
    "modernbert-tokenizer.json",
)

with tempfile.TemporaryDirectory(prefix="modernbert-export-", dir=checkpoint) as staging_name:
    staging = Path(staging_name)

    # Keep all model/config translation in the repository's canonical exporter.
    subprocess.run(
        [sys.executable, str(repo / "scripts" / "export_modernbert.py"),
         str(checkpoint), str(staging)],
        cwd=repo,
        check=True,
    )

    runtime_config = json.loads((staging / "modernbert-config.json").read_text())
    expected_config = {
        "projection_dim": 96,
        "token_gate": True,
        "rope_theta": 10_000.0,
        "global_rope_theta": 80_000.0,
    }
    for key, expected in expected_config.items():
        if runtime_config.get(key) != expected:
            raise RuntimeError(
                f"exported config has {key}={runtime_config.get(key)!r}, expected {expected!r}"
            )

    exported_safetensors = staging / "modernbert.safetensors"
    with safe_open(exported_safetensors, framework="pt", device="cpu") as exported:
        exported_keys = set(exported.keys())
    missing = sorted(required - exported_keys)
    if missing:
        raise RuntimeError(f"exported safetensors is missing: {', '.join(missing)}")

    exported_gguf = staging / "modernbert.gguf"
    subprocess.run(
        ["cargo", "run", "--quiet", "--release", "-p", "quantize", "--",
         str(exported_safetensors), str(exported_gguf)],
        cwd=repo,
        check=True,
    )
    with exported_gguf.open("rb") as gguf_file:
        gguf_magic = gguf_file.read(4)
    if exported_gguf.stat().st_size == 0 or gguf_magic != b"GGUF":
        raise RuntimeError("GGUF converter produced an invalid file")

    # Preserve a self-contained export beside the training checkpoint.
    for name in names:
        atomic_copy(staging / name, checkpoint / name)

    # Install the exact runtime filenames, then prove the Rust binary can load them.
    assets.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="modernbert-backup-", dir=repo) as backup_name:
        backup = Path(backup_name)
        previously_present = {name: (assets / name).is_file() for name in names}
        for name, present in previously_present.items():
            if present:
                shutil.copyfile(assets / name, backup / name)

        try:
            for name in names:
                atomic_copy(staging / name, assets / name)
            subprocess.run(
                ["make", "warp-cli", "EXTRA_FEATURES=deterministic"],
                cwd=repo,
                check=True,
            )
            subprocess.run(
                [str(repo / "warp-cli"), "saliency", "export validation"],
                cwd=repo,
                stdout=subprocess.DEVNULL,
                check=True,
            )
        except BaseException:
            for name, present in previously_present.items():
                destination = assets / name
                if present:
                    atomic_copy(backup / name, destination)
                else:
                    destination.unlink(missing_ok=True)
            raise

for name in names:
    path = assets / name
    print(f"installed {path} ({path.stat().st_size:,} bytes)")
print("verified: warp-cli rebuilt and loaded the installed gated GGUF")
print("note: rebuild any existing index before scoring; its embeddings use the old weights")
