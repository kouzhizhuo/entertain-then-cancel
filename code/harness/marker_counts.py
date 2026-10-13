#!/usr/bin/env python
"""Literalness-marker counts per family in NATURAL thinking traces (lead 2c). usage: marker_counts.py <runs dir P> <out.json>
For each model in E1_main/* and E1s/*: % of natural traces (finish == stop, seed-pooled) containing >= 1 marker and markers per 1k thinking tokens,
per set family (a1, a2-<dataset>, a3-<trigger>, b2, b1-*). Markers: technically, strictly speaking, logically, does not exclude, compatible with, not necessarily, literally, by definition."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
P, out = Path(sys.argv[1]), Path(sys.argv[2])
MARK = re.compile(r"technically|strictly speaking|logically|does not exclude|doesn't exclude|compatible with|not necessarily|literally|by definition", re.I)
items = {}
for f in (Path(__file__).resolve().parents[1] / "data" / "items").glob("[!._]*.jsonl"):
    for l in open(f): r = json.loads(l); items[r["id"]] = r
def fam(bucket, iid):
    it = items.get(iid, {})
    if bucket in ("a3_fel", "a3_nli"): return f"a3-{it.get('subset')}", "a3"
    if bucket == "a2": return f"a2-{it.get('subset')}", "a2"
    if bucket.startswith("a1_"): return "a1", "a1"
    if bucket.startswith("b1_"): return f"b1-{bucket[3:]}", "b1"
    if bucket == "b2_ludwig": return "b2", "b2"
    return None, None
def ci(v):
    v = np.asarray(v, float); rng = np.random.default_rng(0); b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1); return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))
res = {}
for grp in ("E1_main", "E1s"):
    for d in sorted((P / grp).glob("*")) if (P / grp).exists() else []:
        if not (d / "traces.jsonl").exists() or not (d / "samples.jsonl").exists(): continue
        bucket = {}
        for l in open(d / "samples.jsonl"):
            r = json.loads(l); bucket.setdefault(r["item_id"], r["bucket"])
        acc = defaultdict(list)
        for l in open(d / "traces.jsonl"):
            t = json.loads(l)
            if t["finish"] != "stop" or t.get("seed_type"): continue
            b = bucket.get(t["item_id"]); f, _ = fam(b or "", t["item_id"])
            if not f: continue
            n = len(MARK.findall(t["text"])); acc[f].append((1.0 if n else 0.0, 1000.0 * n / max(t["n_tokens"], 1)))
        mk = d.name.lower().replace(".", "p").replace("-fp8", "").replace("-a3b", "-a3b")
        for f, v in acc.items():
            a = [x[0] for x in v]; b = [x[1] for x in v]; lo, hi = ci(a); lo2, hi2 = ci(b)
            res.setdefault(mk, {}).setdefault(f, {"nat": {}})["nat"]["marker"] = {"mean": round(100 * float(np.mean(a)), 3), "lo": round(100 * lo, 3), "hi": round(100 * hi, 3), "n": len(a)}
            res[mk][f]["nat"]["marker1k"] = {"mean": round(float(np.mean(b)), 3), "lo": round(lo2, 3), "hi": round(hi2, 3), "n": len(b)}
out.parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=1); print("markers", {k: len(v) for k, v in res.items()})
