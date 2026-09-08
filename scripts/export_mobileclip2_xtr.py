#!/usr/bin/env python3
"""Export an xtrbert MobileCLIP2 adapter checkpoint for Witchcraft."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file


EXPECTED = {
    "net.0.weight": (512,),
    "net.0.bias": (512,),
    "net.1.weight": (256, 512),
    "net.1.bias": (256,),
    "net.3.weight": (96, 256),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Directory or mobileclip2_modernbert.pt file")
    parser.add_argument("output", nargs="?", default="assets/mobileclip2-xtr.safetensors")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "mobileclip2_modernbert.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["adapter"]
    shapes = {key: tuple(value.shape) for key, value in state.items()}
    if shapes != EXPECTED:
        raise ValueError(f"expected v2b MLP adapter {EXPECTED}, got {shapes}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: value.contiguous() for key, value in state.items()}, output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
