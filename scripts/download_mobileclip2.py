import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO_ID = "timm/MobileCLIP2-S0-OpenCLIP"
ASSETS = Path("assets")
FILES = {
    "open_clip_config.json": "mobileclip2-config.json",
    "tokenizer.json": "mobileclip2-tokenizer.json",
    "open_clip_model.safetensors": "mobileclip2.safetensors",
}


ASSETS.mkdir(exist_ok=True)
for source, target in FILES.items():
    print(f"downloading {REPO_ID}/{source}...")
    cached = hf_hub_download(repo_id=REPO_ID, filename=source)
    out = ASSETS / target
    shutil.copyfile(cached, out)
    print(f"wrote {out}")
