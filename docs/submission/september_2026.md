ScrollScout — a diagnostic for ink detection on unread scrolls
github.com/FrankTheRope/scrollscout · MIT · tag v0.8

ScrollScout answers one question about an ink prediction: is there text here at all? It measures the prominence of line periodicity in 4 cm² windows and gates it on plausible ink coverage. CPU-only, no training, seconds per segment.

What it establishes, all reproducible from the repo:
• Separation with no overlap. On real predictions, text-bearing segments score 1.000 and raw papyrus 0.11–0.26 at the 95th percentile. Raw ink fraction — the trivial baseline — is 1.000 on both classes and separates nothing.
• Feature selection by measurement. Four hand-designed sub-scores were reduced to one by experiment: ink_fraction takes a single value across a segment (demoted to a gate), stroke_shape inverts sign on real data (no vote), anisotropy changes AP by +0.001 on six annotated PHerc0139 segments (dropped). Each removal is documented with the numbers that forced it.
• A documented negative on PHerc1447, a Grand-Prize-eligible scroll: ink_9um, 2 seeds, 3 depth windows, both directions, on the three pre-rendered segments. No text. Two windows crossed threshold and were retired by seed disagreement (Spearman 0.16–0.37 vs 0.78 where text is present). Sanity-checked by reproducing the tutorial's w035 prediction first.
• Cross-model concordance: on w035, ink_9um (9 µm) and canon_2um (2.4 µm) rankings agree at 0.64 — the ranking follows the papyrus, not the model.
• A limitation found and kept: within a fully written segment, the tool is not shown to beat raw ink fraction (p = 0.56), and the run that showed it also showed why — the ink_9um labels mark seven letters on a segment full of text. Documented in benchmark_pherc0139.md.

Docs: results.md (one page), feature_selection.md, benchmark_pherc0139.md, pherc1447_predictions.md, validation_w035.md, baseline_pherc1447.md.
