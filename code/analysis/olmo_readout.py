#!/usr/bin/env python3
"""3-line readout of the OLMo-3-Think same-recipe pair from the canonical exp/results/E1S/summary.json (TEST split; baseline = ntforced).
Delta = NAT minus NT-forced, pp [95% CI over items]."""
import json, sys
from pathlib import Path
S = json.load(open(Path(__file__).resolve().parents[1] / "results" / "E1S" / "summary.json"))
def d(m, s, f, c="nat", k="dR"):
    x = S.get(m, {}).get(s, {}).get(f, {}).get(c, {}).get(k)
    return "n/a" if not x else f"{x['mean']:+.1f} [{x['lo']:+.0f},{x['hi']:+.0f}]"
def r(m, s, f, c):
    x = S.get(m, {}).get(s, {}).get(f, {}).get(c, {}).get("R") or S.get(m, {}).get(s, {}).get(f, {}).get(c, {}).get("acc")
    return "n/a" if not x else f"{x['mean']:.1f}"
ms = [m for m in ("olmo3-7b-think", "olmo3-32b-think") if m in S]
print("1) NAT vs NT-forced, pragmatic rate: " + "; ".join(f"{m}: A3 {d(m,'a3','inf')}, A2 {d(m,'a2','inf')}, A1 {d(m,'a1','fel')}" for m in ms))
print("2) logical-operator families (A3): " + "; ".join(f"{m}: quant {d(m,'a3-quant','inf')}, conn {d(m,'a3-conn','inf')}, modal {d(m,'a3-modal','inf')}" for m in ms))
print("3) PCI and capability: " + "; ".join(f"{m}: B2 acc {d(m,'b2','yn','nat','dacc')} (NT-forced {r(m,'b2','yn','ntforced')}), MATH-500 NAT {S[m].get('cap',{}).get('math500',{}).get('nat',{}).get('acc',{}).get('mean','n/a')}" for m in ms))
if "olmo3-32b-think" not in S: print("(32B-Think not yet in the summary)")
