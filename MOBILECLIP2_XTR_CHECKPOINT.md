# Current MobileCLIP2 XTR checkpoint

Last updated: 2026-09-12

## Selected model

The active image model is `mobileclip2-modernbert-v22`, checkpoint step 750.
It was selected over step 1000 because it gives the best joint COCO and Visual
Genome result rather than optimizing either dataset in isolation.

The model uses:

- MobileCLIP2-S0 at 512x512 resolution, with its final FastViT block fine-tuned.
- A residual gated MLP mapping each 512d image patch into the frozen
  ModernBERT 96d token space.
- The 128 most salient image tokens at inference time.
- The same Granite-derived ModernBERT v64 text tower used by Witchcraft text
  search.
- The inherited v18 96d output bias, kept frozen during v22 training.

V22 jointly trains one COCO and one Visual Genome microbatch per optimizer
update. Every COCO image is supervised by all five human captions. Training
uses exact MaxSim plus production-style imputed XTR, a 2,048-image shared
negative queue (128 sampled per microbatch), a fixed XTR top-k of 256, native
MobileCLIP ranking distillation, token-gate supervision, and adapter/vision
anchors.

## Retrieval results

All numbers below use 128 gated image tokens and production-style imputed XTR
with `k=40000`.

| Evaluation | Recovered v18 | v22 step 750 |
|---|---:|---:|
| Visual Genome NDCG@10, 1,000 images | 0.1772 | 0.1875 |
| Visual Genome R@10, 1,000 images | 0.3290 | 0.3360 |
| COCO NDCG@10, 1,000 images | 0.4361 | 0.4680 |
| COCO R@10, 1,000 images | 0.7020 | 0.7450 |
| Visual Genome NDCG@10, 2,000 images | 0.1427 | 0.1445 |
| COCO NDCG@10, 2,000 images | 0.3252 | 0.3484 |
| COCO R@10, 2,000 images | 0.5490 | 0.5865 |

Step 1000 was nearly tied on Visual Genome but regressed on the larger COCO
evaluation, so step 750 is the selected checkpoint.

## Durable locations

Do not put checkpoints or logs under `/mnt/cache`; that filesystem is wiped
when the GPU host is recreated. It is used only for replaceable datasets and
model-download caches.

- GPU best checkpoint:
  `/home/jhansen/xtrbert-runs/mobileclip2-modernbert-v22-coco-vg-joint/best`
- GPU training log:
  `/home/jhansen/xtrbert-runs/logs/mobileclip2-modernbert-512-v22-coco-vg-joint.log`
- Mac checkpoint copy:
  `/Users/jhansen/models/xtrbert/mobileclip2-v22-step750`
- Previous v18 Witchcraft export backup:
  `/Users/jhansen/models/xtrbert/mobileclip2-v18-export`
- Training configuration:
  `~/src/xtrbert/config-mobileclip2-modernbert-512-v22-coco-vg-joint.yaml`

The GPU v18 initializer and v64 text checkpoint were reconstructed from the
Witchcraft exports and also live under `/home/jhansen/xtrbert-runs`. The script
`~/src/xtrbert/recover_mobileclip2_training_artifacts.py` documents that
recovery path.

## Export to Witchcraft

The active assets are:

- `assets/mobileclip2-xtr.safetensors`: adapter, token gate, and output bias.
- `assets/mobileclip2.safetensors`: MobileCLIP2 with the fine-tuned vision
  tensors merged in.

To export the selected checkpoint again:

```sh
cd ~/src/witchcraft
env/bin/python scripts/export_mobileclip2_xtr.py \
  /Users/jhansen/models/xtrbert/mobileclip2-v22-step750 \
  assets/mobileclip2-xtr.safetensors \
  --vision-base assets/mobileclip2.safetensors \
  --vision-output assets/mobileclip2.safetensors
make mobileclip2-xtr
```

Expected export description:

```text
residual_gated_mlp, 128 tokens, output bias
784 fine-tuned vision tensors
```

The two asset files are intentionally ignored by Git. Preserve a checkpoint
copy outside the repository before replacing them.

## Using the model

Smoke-test image/text scoring without changing the database:

```sh
./warp-cli scoreimage /path/to/image.jpg "a textual query"
```

Existing image embeddings were produced by the previous weights and must be
recomputed before judging v22 retrieval. Text embeddings do not need to be
recomputed because the ModernBERT text tower is unchanged.

For a directory already present in the image cache, force image re-embedding
and then rebuild the semantic index once:

```sh
./warp-cli embedimages --force /path/to/images
./warp-cli index
```

Without `--force`, `embedimages` skips files already cached under the same
filename. It prints progress for each file and warns and continues past broken
image files.

The image-vector score boost remains `+0.21` in `src/lib.rs`. V22's held-out
absolute scores still support that setting: after the boost, mean positive
image scores are about 0.736 versus 0.751 for text, while mean negative scores
are about 0.585 versus 0.578. The terminal image-display cutoff remains 0.66
and is independent of the retrieval boost.
