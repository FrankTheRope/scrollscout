"""Block-average the 2.399 um predictions 4x and pair them with ink_9um labels.

Writes data/0139_pred/<w>.tif (9.6 um) and two manifests: all pairs, and the
six whose label shares the prediction's canvas (w028/w029 do not; the
benchmark's alignment check rejects them).
"""
import glob
import json
import os

import numpy as np
import tifffile

manifest = []
for f in sorted(glob.glob("data/0139_pred/raw/w0*.tif")):
    w = os.path.basename(f)[:-4]
    a = tifffile.imread(f)
    H, W = a.shape[0] // 4 * 4, a.shape[1] // 4 * 4
    d = a[:H, :W].reshape(H // 4, 4, W // 4, 4).mean(axis=(1, 3)).astype(np.uint8)
    out = f"data/0139_pred/{w}.tif"
    tifffile.imwrite(out, d)
    nat = f"ink_9um/labels/native9-scrollprizeorg-21slices/{w}/{w}_inklabels.tif"
    ali = f"ink_9um/labels/aligned-scrollprizeorg-21slices/pherc0139-{w}/pherc0139-{w}_inklabels.tif"
    lab = nat if os.path.exists(nat) else (ali if os.path.exists(ali) else None)
    if lab is None:
        print(w, "no label")
        continue
    manifest.append({"name": f"pherc0139-{w}", "prediction": out, "label": lab,
                     "pixel_size_um": round(2.399 * 4, 3)})
    print(f"{w}: pred {d.shape}  label {'native9' if lab == nat else 'aligned'}")
json.dump(manifest, open("data/0139_pred/manifest.json", "w"), indent=1)
six = [e for e in manifest if not e["name"].endswith(("w028", "w029"))]
json.dump(six, open("data/0139_pred/manifest6.json", "w"), indent=1)
print(len(manifest), "pairs;", len(six), "with matching canvas -> manifest6.json")
