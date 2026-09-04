"""Regression tests on synthetic data: text must outrank noise and pure stripes,
and the detected line pitch must match the synthetic one."""
import numpy as np

from scrollscout.letterness import ScoreConfig, score_image
from scrollscout.synth import make_text_image, make_noise_image, make_stripes_image

CFG = ScoreConfig(pixel_size_um=100.0, working_um=100.0)


def _best(img):
    windows, grid, meta = score_image(img, CFG)
    assert windows, "no windows scored"
    return windows[0], grid, meta


def test_text_beats_noise_and_stripes():
    text, _, _ = _best(make_text_image(line_pitch_mm=5.0, seed=0))
    noise, _, _ = _best(make_noise_image(seed=1))
    stripes, _, _ = _best(make_stripes_image(seed=2))
    assert text.score > 2 * noise.score, (text.score, noise.score)
    assert text.score > stripes.score, (text.score, stripes.score)
    assert text.score > 0.5


def test_pitch_recovered():
    for pitch in (4.0, 5.0, 7.0):
        best, _, _ = _best(make_text_image(line_pitch_mm=pitch, seed=int(pitch)))
        assert abs(best.best_pitch_mm - pitch) / pitch < 0.2, (pitch, best.best_pitch_mm)


def test_tilted_text_still_found():
    for tilt in (10.0, -10.0):
        best, _, _ = _best(make_text_image(line_pitch_mm=6.0, tilt_deg=tilt, seed=3))
        assert best.score > 0.5
        # the angle search must engage, with the sign that undoes the tilt
        assert abs(best.best_angle_deg) >= 6 and np.sign(best.best_angle_deg) == -np.sign(tilt)
        assert abs(best.best_pitch_mm - 6.0) < 1.0


def test_boxes_are_inside_image():
    img = make_text_image(seed=4)
    windows, grid, meta = score_image(img, CFG)
    H, W = img.shape
    for w in windows:
        assert 0 <= w.y0 < w.y1 <= H and 0 <= w.x0 < w.x1 <= W


def test_mask_excludes_windows():
    img = make_text_image(seed=5)
    mask = np.zeros(img.shape, dtype=bool)
    mask[:, : img.shape[1] // 2] = True
    windows, grid, meta = score_image(img, CFG, mask=mask)
    assert all(w.x1 <= img.shape[1] // 2 + meta["window_px_working"] for w in windows)
    assert len(windows) < grid.size


# --- benchmark ---------------------------------------------------------------
def _make_bench(tmpdir):
    """Prediction with 4 text patches on noise; sparse label marking only the text."""
    import tifffile
    from scipy import ndimage as ndi
    from scrollscout.synth import make_noise_image
    rng = np.random.default_rng(0)
    px, H, W = 100.0, 1200, 1200
    pred = make_noise_image(shape_mm=(120, 120), pixel_um=px, seed=7)
    label = np.zeros((H, W), bool)
    for k, (y, x) in enumerate([(100, 100), (100, 700), (750, 150), (700, 800)]):
        t = make_text_image(shape_mm=(35, 35), pixel_um=px, line_pitch_mm=5.0, seed=k, margin_mm=2)
        h, w = t.shape
        pred[y:y + h, x:x + w] = np.maximum(pred[y:y + h, x:x + w], t)
        ink = t > np.percentile(t, 90)
        lab_, n = ndi.label(ink)
        keep = np.zeros_like(ink)
        for s in rng.choice(np.arange(1, n + 1), size=max(1, int(0.4 * n)), replace=False):
            keep |= (lab_ == s)
        label[y:y + h, x:x + w] = keep
    pp, lp = f"{tmpdir}/pred.tif", f"{tmpdir}/label.tif"
    tifffile.imwrite(pp, (np.clip(pred, 0, 1) * 255).astype(np.uint8))
    tifffile.imwrite(lp, (label * 255).astype(np.uint8))
    return pp, lp


def test_benchmark_regions_are_informative(tmp_path):
    """A metric that saturates for every ranking measures nothing. Grid regions
    must stay numerous enough that random ordering does NOT reach them all."""
    from scrollscout.benchmark import run
    from scrollscout.letterness import ScoreConfig
    pp, lp = _make_bench(tmp_path)
    cfg = ScoreConfig(pixel_size_um=100.0, working_um=100.0, window_mm=10.0,
                      stride_mm=2.0, auto_mask=False)
    rep = run(pp, lp, cfg, tmp_path / "out")
    assert rep["positives"]["n_regions"] >= 20
    rnd = rep["results"]["baseline: casuale"]
    assert rnd["recall@10"] < 0.5 and rnd["recall@50"] < 0.95
    sat = rep["diagnostics"]["saturation"]
    assert set(sat) == {"ink_fraction", "line_periodicity", "stroke_shape", "anisotropy"}
    assert all(0.0 <= v["frac_at_1.00"] <= 1.0 for v in sat.values())


def test_benchmark_beats_random(tmp_path):
    from scrollscout.benchmark import run
    from scrollscout.letterness import ScoreConfig
    pp, lp = _make_bench(tmp_path)
    cfg = ScoreConfig(pixel_size_um=100.0, working_um=100.0, window_mm=10.0,
                      stride_mm=2.0, auto_mask=False)
    rep = run(pp, lp, cfg, tmp_path / "out")
    full = rep["results"]["ScrollScout (full)"]
    rnd = rep["results"]["baseline: casuale"]
    assert rep["positives"]["n_regions"] > 0
    assert full["average_precision"] > 2 * rnd["average_precision"]
    assert full["recall@10"] > 2 * rnd["recall@10"]
    assert rep["positives"]["region_mm"] == 5.0
    assert 0.0 <= full["recall@100"] <= 1.0
    # NMS must actually suppress overlaps
    assert full["n_kept_after_nms"] <= 100


def test_benchmark_detects_useless_ranking(tmp_path):
    """A benchmark that cannot fail a bad ranking is worthless: check it can."""
    from scrollscout.benchmark import rank_metrics, nms
    import numpy as _np
    boxes = _np.array([[i * 300, 0, i * 300 + 294, 294] for i in range(20)])
    positive = _np.zeros(20, bool); positive[:4] = True
    win_regions = [{1} if positive[i] else set() for i in range(20)]
    good = rank_metrics(_np.arange(20), positive, boxes, win_regions, 1)
    bad = rank_metrics(_np.arange(20)[::-1], positive, boxes, win_regions, 1, ks=(5,))
    assert good["average_precision"] == 1.0
    assert bad["average_precision"] < 0.5
