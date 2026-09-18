# Stock MobileCLIP2 baseline

Witchcraft can exhaustively score stock MobileCLIP2-S0 512-dimensional global
embeddings alongside the 96-dimensional XTR index. The baseline uses the
official unmodified weights and native 256x256 input resolution.

Download the separate stock checkpoint and build the CLI:

```sh
make mobileclip2-stock-assets warp-cli
```

Register images, embed any missing pooled vectors, and update the dedicated
4-bit polar-compressed IVF index:

```sh
./warp-cli clipregister /path/to/images [...]
./warp-cli clipembed
./warp-cli clipindex
```

This stores one normalized 512d pooled MobileCLIP2 vector per image as fixed-scale
signed int8 in the isolated `mobileclip2-ivf.sqlite` database. Bad images produce
a warning and are skipped before rowids are allocated.

Query the IVF index and compare it with XTR:

```sh
./warp-cli clipquery a photo of children on a beach
./warp-cli query a photo of children on a beach
```

Pass `-d` to display the top images:

```sh
./warp-cli clipquery -d a red tractor in winter
```

The raw cosine values are not calibrated to XTR scores; compare rankings and
relevance rather than score magnitude.

Use `--exact` to exhaustively score the same cached vectors:

```sh
./warp-cli clipquery --exact a man with a hat in front of two canoes
```
