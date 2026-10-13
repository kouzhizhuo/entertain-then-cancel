#!/usr/bin/env python3
"""E7 trajectory classes (lead / red-team 1.2): per item x seed trace, the sequence of answers (pragmatic = is_gold) after plain truncation at k = 0, 32, ... 2048 tokens and at the natural end.
Only prefixes k < n_tokens are distinct points (a trace shorter than k is simply finished); the end answer is think_full. Classes:
 a = pragmatic at some k < end and literal at the end;  b = literal throughout;  c = pragmatic throughout;  d = literal at every k < end and pragmatic at the end;  e = other.
Also reports `alive`: share of traces still running at each k (n_tokens > k), and the same classes for the cohort of traces still running at k = 512 (traj512). `traj_k32` repeats the classes without the k=0 answer (traces with no prefix 0 < k < end are left out). Traces with any invalid answer along the ladder are dropped (n reported).
Shares are item-level (an item's traces averaged), 95% CI = item bootstrap (2000). usage: python e7_traj.py <runs/E7 or runs/E7_full dir>  -> <model>/traj.json"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
KS = {"think_b0_plain": 0, "think_b32_plain": 32, "think_b64_plain": 64, "think_b128_plain": 128, "think_b256_plain": 256, "think_b512_plain": 512, "think_b1024_plain": 1024, "think_b2048_plain": 2048}
def sets_of(r, full):
    pre = "a3f" if r["bucket"].startswith("a3f") else "a3"
    if r["bucket"] == "a2": return ["a2"]
    sub = {"quantifiers": "quant", "connectives": "conn", "modals": "modal"}.get(r["subset"])
    return [pre] + ([f"{pre}-{sub}"] if sub else [])
def ci(v):
    v = np.asarray(v, float); rng = np.random.default_rng(0); b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1); return round(float(np.percentile(b, 2.5)) * 100, 1), round(float(np.percentile(b, 97.5)) * 100, 1)
def cls(seq, end):
    if all(s == 0 for s in seq) and end == 0: return "b"
    if all(s == 1 for s in seq) and end == 1: return "c"
    if any(s == 1 for s in seq) and end == 0: return "a"
    if all(s == 0 for s in seq) and end == 1: return "d"
    return "e"
def blk(v, n):
    lo, hi = ci(v); return {"share": round(100 * float(np.mean(v)), 1), "lo": lo, "hi": hi, "n": n, "share__digits": 1, "lo__digits": 1, "hi__digits": 1}
for mdir in sorted(Path(sys.argv[1]).glob("*")):
    f = mdir / "samples.jsonl"
    if not f.exists(): continue
    T = defaultdict(dict); meta = {}
    for l in open(f):
        r = json.loads(l); k = KS.get(r["condition"], "end" if r["condition"] == "think_full" else None)
        if k is None: continue
        key = (r["item_id"], r["seed"]); T[key][k] = (r["is_gold"] if r["valid"] else None)
        if k == "end": T[key]["n"] = r["think_tokens"]; meta[key] = r
    per = defaultdict(lambda: defaultdict(list)); dropped = defaultdict(int)
    for key, d in T.items():
        if "end" not in d or key not in meta: continue
        n = d["n"]; ladder = [(k, d[k]) for k in sorted(k for k in d if isinstance(k, int)) if k < n or k == 0]
        vals = [v for _, v in ladder] + [d["end"]]
        for s in sets_of(meta[key], False):
            if any(v is None for v in vals): dropped[s] += 1; continue
            seq = [int(v) for _, v in ladder]; c = cls(seq, int(d["end"]))
            pos = [int(v) for k, v in ladder if k > 0]       # variant without the k=0 (no-thinking) answer: needs at least one prefix 0 < k < end
            per[s][key[0]].append({"c": c, "n": n, "c32": cls(pos, int(d["end"])) if pos else None})
    out = {}
    for s, items in per.items():
        node = {"n_traces": sum(len(v) for v in items.values()), "n_items": len(items), "n_dropped_invalid": dropped[s]}
        node["traj"] = {c: blk([np.mean([t["c"] == c for t in v]) for v in items.values()], int(sum(t["c"] == c for v in items.values() for t in v))) for c in "abcde"}
        v32 = {i: [t for t in v if t["c32"] is not None] for i, v in items.items()}; v32 = {i: v for i, v in v32.items() if v}
        node["traj_k32"] = {c: blk([np.mean([t["c32"] == c for t in v]) for v in v32.values()], int(sum(t["c32"] == c for v in v32.values() for t in v))) for c in "abcde"} if len(v32) >= 5 else None
        node["alive"] = {f"k{k:04d}": {"share": round(100 * float(np.mean([t["n"] > k for v in items.values() for t in v])), 1), "share__digits": 1} for k in (32, 64, 128, 256, 512, 1024, 2048)}
        coh = {i: [t for t in v if t["n"] > 512] for i, v in items.items()}; coh = {i: v for i, v in coh.items() if v}
        node["traj512"] = {c: blk([np.mean([t["c"] == c for t in v]) for v in coh.values()], int(sum(t["c"] == c for v in coh.values() for t in v))) for c in "abcde"} if len(coh) >= 5 else None
        node["n_traces_alive512"] = int(sum(len(v) for v in coh.values()))
        out[s] = node
    json.dump(out, open(mdir / "traj.json", "w"), indent=1)
    print(mdir.name, {s: {c: out[s]["traj"][c]["share"] for c in "abcde"} | {"n": out[s]["n_traces"], "alive512": out[s]["alive"]["k0512"]["share"]} for s in ("a3-quant", "a3-conn") if s in out})
