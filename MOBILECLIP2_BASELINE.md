# Stock MobileCLIP2 baseline

Witchcraft can exhaustively score stock MobileCLIP2-S0 512-dimensional global
embeddings alongside the 96-dimensional XTR index. The baseline uses the
official unmodified weights and native 256x256 input resolution.

Download the separate stock checkpoint and build the CLI:

```sh
make mobileclip2-stock-assets warp-cli
```

Create a matrix for the same image directories used by Witchcraft:

```sh
./warp-cli clipembedimages mobileclip2-baseline /path/to/images [...]
```

This writes normalized row-major `f32` values to
`mobileclip2-baseline.f32` and model metadata plus the corresponding filenames
to `mobileclip2-baseline.json`. Bad images produce a warning and are skipped.

Run an exact exhaustive cosine query and compare it with XTR:

```sh
./warp-cli clipquery mobileclip2-baseline a photo of children on a beach
./warp-cli query a photo of children on a beach
```

Pass `-d` to `clipquery` to display its top images with `viu`:

```sh
./warp-cli clipquery -d mobileclip2-baseline a red tractor in winter
```

The raw cosine values are not calibrated to XTR scores; compare rankings and
relevance rather than score magnitude.

## Pooled-vector IVF experiment

The existing matrix can also be imported into a dedicated Witchcraft IVF index.
This stores one normalized 512d pooled MobileCLIP2 vector per image and uses
4-bit polar residual compression. It does not modify `mydb.sqlite`.

```sh
./warp-cli clipivfbuild mobileclip2-baseline
./warp-cli clipivfquery a man with a hat in front of two canoes
```

Pass `--force` to overwrite already imported rows. Compare IVF retrieval with
exhaustive cosine scoring over the same cached vectors using:

```sh
./warp-cli clipivfquery --exact a man with a hat in front of two canoes
```
