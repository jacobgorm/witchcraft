#!/usr/bin/env python3
"""Export an xtrbert MobileCLIP2 adapter checkpoint for Witchcraft."""

import argparse
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


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

RESIDUAL_GATED_MLP_EXPECTED = {
    **RESIDUAL_MLP_EXPECTED,
    "token_gate_norm.weight": (512,),
    "token_gate_norm.bias": (512,),
    "token_gate.weight": (1, 512),
    "token_gate.bias": (1,),
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
    parser.add_argument(
        "--image-tokens",
        type=int,
        help="Number of gated image tokens to export; defaults to export_image_tokens",
    )
    parser.add_argument(
        "--vision-base",
        default="assets/mobileclip2.safetensors",
        help="Stock full MobileCLIP2 weights to receive fine-tuned vision tensors",
    )
    parser.add_argument(
        "--vision-output",
        help="Write merged full MobileCLIP2 weights here when the checkpoint contains them",
    )
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "mobileclip2_modernbert.pt"
    checkpoint_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint_state["adapter"]
    shapes = {key: tuple(value.shape) for key, value in state.items()}
    config = {}
    config_path = checkpoint.parent / "config.json"
    if config_path.is_file():
        import json

        config = json.loads(config_path.read_text())

    if shapes == GATED_MLP_EXPECTED:
        adapter_type = "gated_mlp"
        image_tokens = args.image_tokens or int(config.get("export_image_tokens", 128))
        residual_scale = None
    elif shapes == RESIDUAL_MLP_EXPECTED:
        adapter_type = "residual_mlp"
        image_tokens = 256
        residual_scale = args.residual_scale
        if residual_scale is None and config:
            residual_scale = float(config["residual_scale"])
        if residual_scale is None:
            raise ValueError("residual adapter requires --residual-scale or config.json")
    elif shapes == RESIDUAL_GATED_MLP_EXPECTED:
        adapter_type = "residual_gated_mlp"
        image_tokens = args.image_tokens or config.get("export_image_tokens")
        if image_tokens is None:
            raise ValueError(
                "gated residual adapter requires --image-tokens or export_image_tokens"
            )
        image_tokens = int(image_tokens)
        residual_scale = args.residual_scale
        if residual_scale is None and config:
            residual_scale = float(config["residual_scale"])
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

    vision_state = checkpoint_state.get("image_visual")
    if vision_state is not None:
        if not args.vision_output:
            raise ValueError(
                "checkpoint contains fine-tuned vision weights; pass --vision-output"
            )
        vision_base = Path(args.vision_base)
        vision_output = Path(args.vision_output)
        full_state = load_file(vision_base, device="cpu")
        replacements = {f"visual.{key}": value for key, value in vision_state.items()}
        missing = sorted(set(replacements) - set(full_state))
        mismatched = sorted(
            key
            for key, value in replacements.items()
            if key in full_state and tuple(value.shape) != tuple(full_state[key].shape)
        )
        if missing or mismatched:
            raise ValueError(
                f"fine-tuned vision tensors do not match base model: "
                f"missing={missing}, shape_mismatch={mismatched}"
            )
        full_state.update(
            {key: value.contiguous() for key, value in replacements.items()}
        )
        with safe_open(vision_base, framework="pt", device="cpu") as source:
            vision_metadata = source.metadata()
        vision_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = vision_output.with_name(f".{vision_output.name}.tmp")
        save_file(full_state, temporary, metadata=vision_metadata)
        os.replace(temporary, vision_output)
        print(
            f"wrote {vision_output} "
            f"({len(replacements)} fine-tuned vision tensors)"
        )


if __name__ == "__main__":
    main()
