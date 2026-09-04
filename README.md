# ScrollScout

**CPU-first ink prospecting for the Vesuvius Challenge.**
Ranks 4 cm² windows of an ink-prediction image (or a raw surface render) by how
much *text-like structure* they contain, so a human — or a GPU budget — only
looks at the most promising regions of the 13 scrolls eligible for the
[First Letters prizes](https://scrollprize.org/prizes#first-letters-prizes).

No training, no GPU, no model weights: classical signal processing on top of
predictions produced by the public `scrollprize/ink_9um` models (or any other).

```
                     GPU (rented / free tier)             CPU (this repo)
 tifxyz segment ──► vc_render_tifxyz ──► ink_9um ──► scrollscout aggregate ──► scrollscout score
                     (surface volume)   (N runs:      (mean / std / consistency)   (ranked 4 cm²
                                        seeds, ckpts,                              windows + heatmap)
                                        depth windows)
```

## Install

```bash
git clone <this repo> && cd scrollscout
pip install -e .            # numpy, scipy, scikit-image, tifffile, pillow
scrollscout --help
```

## Try it in 30 seconds (no scroll data needed)

```bash
scrollscout synth --out demo                                   # synthetic text / noise / stripes
scrollscout score demo/text_tilted.tif --pixel-size-um 100 --out demo/score_text
scrollscout score demo/noise.tif       --pixel-size-um 100 --out demo/score_noise
```

Text windows score ≈ 0.9–1.0, blobby noise ≈ 0.1–0.3, pure stripes ≈ 0.3.
Each run writes `heatmap.png`, `overlay.png` (top-K boxes) and `windows.json`
(boxes in original pixels **and** millimetres, sub-scores, detected line pitch
and tilt).

## On real data

```bash
# 1. a prediction image from the ink tutorial / ink_9um models (ink bright)
scrollscout score predictions/w035_9um.tif --pixel-size-um 9.362 --out out/w035

# 2. several runs of the same segment (2 seeds x checkpoints x depth windows x directions)
scrollscout aggregate preds/w035_*.tif --out out/w035_ens
scrollscout score out/w035_ens/mean.tif --pixel-size-um 9.362 --out out/w035_ens/score

# 3. the 13 eligible scrolls and where their data lives
scrollscout catalog --ls        # needs `aws` CLI; bucket is public (no account)
```

Useful flags: `--window-mm 20 --stride-mm 5` (prize area), `--pitch-min/--pitch-max`
(expected line spacing, mm), `--letter-min/--letter-max` (letter height, mm),
`--mask` (valid region), `--invert` (ink dark).

## How the score works

For every window (default 20 × 20 mm, stride 5 mm, analysed at ~0.1 mm/px):

| sub-score | what it measures | why |
|---|---|---|
| `ink_fraction` | fraction of pixels above a global Otsu threshold, scored with a plausibility band (default 4–35 %) | empty windows and smears (wrong depth, sheet merger) both fail |
| `line_periodicity` | prominence of the dominant period of the row-projection autocorrelation inside the line-pitch range, averaged over 3 vertical strips and multiplied by the cross-strip coherence; searched over tilt angles ±10° | text lines are periodic **and** span the window; smoothed noise is neither |
| `stroke_shape` | fraction of ink area in connected components that are letter-sized and thin/wiggly (perimeter²/4πA ≥ 3) | separates strokes from blobs |
| `anisotropy` | row periodicity vs column periodicity | lines are horizontal bands, not vertical ones |

Final score = weighted geometric mean (weights 1 / 2 / 1.5 / 1), so a window has
to pass *all* tests. Synthetic regression tests live in `tests/`.

## Validation

On real `ink_9um` predictions of a known-text segment (PHerc0139 w035) the
scorer reaches **0.93** with 117/440 windows above 0.6; on raw papyrus renders
of PHerc1447 it tops out at 0.50 with none above 0.6. Details, thresholds and
reproduction commands: [`docs/validation_w035.md`](docs/validation_w035.md)
and [`docs/baseline_pherc1447.md`](docs/baseline_pherc1447.md).

## Limits (read before trusting a number)

* This is a **prioritisation** tool, not a letter detector. A high score means
  "worth a human look and a finer render", never "there is text here".
* Thresholds were set on synthetic data and a handful of public predictions.
  Expect to retune `--pitch-*`, `--letter-*` and the ink band per scroll
  (letter size varies between scribes).
* Prediction images must be *programmatically generated*; ScrollScout never
  touches the image content, it only reads it.
* The `ink_9um` models are sensitive to depth offsets — always feed
  `aggregate` several depth windows and look at `consistency.tif`: text that
  appears in only one depth window is suspect.

## Roadmap

- [x] `score`: ranked 4 cm² windows, heatmap, JSON
- [x] `aggregate`: ensemble / depth-sweep diagnostics
- [x] `catalog`: eligible scrolls, S3 paths
- [ ] `render`: wrapper around `vc_render_tifxyz --remote-url` (stream only the chunks a segment touches)
- [ ] `qc`: patch topology checks (sheet-switch / merger detection, fibre continuity) to score segments *before* spending GPU time
- [ ] `sweep`: job generator for ink_9um runs (seeds × checkpoints × depth windows × directions) with a fixed-seed manifest
- [ ] `pack`: submission packager (scale bar, letter sizes, row baselines on a fibre-visible render, mesh ↔ image naming, held-out validation report)

## License & data

Code: MIT. Vesuvius Challenge data is CC BY-NC 4.0 — see https://scrollprize.org/data.
Nothing here is affiliated with or endorsed by Scroll Prize, Inc.
