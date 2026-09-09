#!/usr/bin/env python3
"""Export an xtrbert MobileCLIP2 adapter checkpoint for Witchcraft."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


GATED_MLP_EXPECTED = {
    "net.0.weight": (512,),
    "net.0.bias": (512,),
    "net.1.weight": (256, 512),
    "net.1.bias": (256,),
    "net.3.weight": (96, 256),
    "token_gate_norm.weight": (512,),
    "token_gate_norm.bias": (512,),
    "token_gate.weight": (1, 512),
    "token_gate.bias": (1,),
}

RESIDUAL_MLP_EXPECTED = {
    "base.net.0.weight": (512,),
    "base.net.0.bias": (512,),
    "base.net.1.weight": (256, 512),
    "base.net.1.bias": (256,),
    "base.net.3.weight": (96, 256),
    "residual.0.weight": (512,),
    "residual.0.bias": (512,),
    "residual.1.weight": (128, 512),
    "residual.1.bias": (128,),
    "residual.3.weight": (96, 128),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Directory or mobileclip2_modernbert.pt file")
    parser.add_argument("output", nargs="?", default="assets/mobileclip2-xtr.safetensors")
    parser.add_argument(
        "--residual-scale",
        type=float,
        help="Residual multiplier; defaults to the checkpoint's config.json value",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "mobileclip2_modernbert.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["adapter"]
    shapes = {key: tuple(value.shape) for key, value in state.items()}
    if shapes == GATED_MLP_EXPECTED:
        adapter_type = "gated_mlp"
        image_tokens = 128
        residual_scale = None
    elif shapes == RESIDUAL_MLP_EXPECTED:
        adapter_type = "residual_mlp"
        image_tokens = 256
        residual_scale = args.residual_scale
        config_path = checkpoint.parent / "config.json"
        if residual_scale is None and config_path.is_file():
            import json

            residual_scale = float(json.loads(config_path.read_text())["residual_scale"])
        if residual_scale is None:
            raise ValueError("residual adapter requires --residual-scale or config.json")
    else:
        raise ValueError(
            "expected a gated or residual 512px MLP adapter; "
            f"got tensor shapes {shapes}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "adapter_type": adapter_type,
        "image_size": "512",
        "image_tokens": str(image_tokens),
    }
    if residual_scale is not None:
        metadata["residual_scale"] = str(residual_scale)
    save_file(
        {key: value.contiguous() for key, value in state.items()},
        output,
        metadata=metadata,
    )
    print(f"wrote {output} ({adapter_type}, {image_tokens} tokens)")


if __name__ == "__main__":
    main()
