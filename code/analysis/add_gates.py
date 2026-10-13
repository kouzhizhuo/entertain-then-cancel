#!/usr/bin/env python3
"""Adds "gates": {arm: {gate: {"diff": pp, "lb": one-sided 95% lower bound, "n": paired items}}} to every selection_dev_*.json in results/M1 (gates ix_can, imppres_truth, folio, gsm8k, each vs the file's own NAT reference).
The numbers are the ones the m1_select-family scripts already computed (gate_lb = [lower bound (5th percentile of the 4000-draw paired bootstrap), mean difference]); this only re-labels them and adds the dev stratum sizes (ix_can 136, imppres truth 350+358, folio 200, gsm8k 150). The reference arm has diff 0 / lb 0 by construction."""
import json, sys
from pathlib import Path
M1 = Path(__file__).resolve().parents[1] / "results" / "M1"
N = {"ix_can": 136, "imppres_truth": 708, "folio": 200, "gsm8k": 150}
for f in sorted(M1.glob("selection_dev*.json")):
    d = json.load(open(f)); tab = d.get("table")
    if not tab: continue
    g = {arm: {k: {"diff": v[1], "lb": v[0], "n": N[k]} for k, v in t["gate_lb"].items()} for arm, t in tab.items() if "gate_lb" in t}
    d["gates"] = g; d["gates_note"] = "diff = mean paired difference in pp vs this file's NAT reference; lb = one-sided 95% lower bound (5th percentile of a paired bootstrap, 4000 draws, as m1_select); n = dev items in the stratum; gate passes when lb > -2"
    json.dump(d, open(f, "w"), indent=1); print(f.name, len(g), "arms")
