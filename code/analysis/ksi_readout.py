#!/usr/bin/env python3
"""Readout of the two gate-deciding runs from the canonical summaries (TEST split): (1) A1-ctx KSI = R_full - R_partial per model/condition,
(2) Qwen NT-forced baseline: NAT minus ntforced (dRnt) on A3/A2/A1/B2. pp [95% CI]."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
K = json.load(open(R / "A1CTX/summary.json")); S = json.load(open(R / "E1S/summary.json"))
def b(x): return "n/a" if not x else f"{x['mean']:+.1f} [{x['lo']:+.0f},{x['hi']:+.0f}]"
def g(d, *k):
    for x in k:
        if not isinstance(d, dict) or x not in d: return None
        d = d[x]
    return d
print("A1-ctx KSI (R_full - R_partial, pp):")
for m in ("qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b", "gpt-oss-120b", "olmo3-32b-think"):
    for fr, fm in (("a1ctx-fel", "fel"), ("a1ctx-inf", "inf")):
        node = g(K, m, fr, fm)
        if node: print(f"  {m:16s} {fr[-3:]}: " + "; ".join(f"{c} {b(v.get('ksi'))} (Rfull {v['Rfull']['mean']:.0f}, Rpart {v['Rpart']['mean']:.0f})" for c, v in node.items() if c in ("off", "ntforced", "nat", "low", "high")))
print("Qwen NT-forced baseline: NAT minus ntforced (dRnt/daccnt):")
for m in ("qwen3-8b", "qwen3-32b"):
    print(f"  {m}: A3 {b(g(S,m,'a3','inf','nat','dRnt'))}, A2 {b(g(S,m,'a2','inf','nat','dRnt'))}, A1 {b(g(S,m,'a1','fel','nat','dRnt'))}, B2 acc {b(g(S,m,'b2','yn','nat','daccnt'))}; ntforced R: A3 {g(S,m,'a3','inf','ntforced','R','mean')}, A2 {g(S,m,'a2','inf','ntforced','R','mean')}, A1 {g(S,m,'a1','fel','ntforced','R','mean')}")
