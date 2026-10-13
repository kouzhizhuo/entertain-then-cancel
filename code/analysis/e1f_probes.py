#!/usr/bin/env python3
"""E1F yes-bias discriminators (lead request; METHOD_CHECK s2). Inputs: results/api_prior (no-utterance prior probe, felicity question without the utterance; pragmatic = Yes) and results/api_flip (polarity-flipped question 'is it still left open?';
pragmatic = No), 160 a3-quant/conn items x 2 samples, lowest vs highest dose. Per model x dose: pragmatic rate, Yes rate, failure shares, paired cluster-bootstrap delta (items valid at both doses; seeds averaged within item) and adversarial bounds.
Merged into results/E1F/summary.json as fmt tokens infprior / infflip (set a3-quant, a3-conn, a3) and written to results/E1F/artefact_probes.json. Reads the original Yes/No format values from the main summary for the side-by-side table."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "harness"))
from common import load_items_from_config
DOSE_RANK = {"off512": -1, "eff-min": 0, "off": 0, "eff-low": 1, "bud1k": 2, "eff-high": 3, "bud4k": 3, "bud16k": 4}
def tok(n): return n.split("_", 1)[1].replace(".", "p")
def agg(rows, yes_idx):
    d = defaultdict(list)
    for r in rows: d[r["item_id"]].append(r)
    out = {}
    for k, v in d.items():
        ok = [r for r in v if r["valid"]]
        out[k] = {"valid": bool(ok), "gold": sum(float(r["is_gold"]) for r in ok) / len(ok) if ok else None, "yes": sum(1.0 * (r["choice"] == yes_idx[k]) for r in ok) / len(ok) if ok else None}
    return out
def boot(pairs, clus, n=10000):
    by = defaultdict(list)
    for k, a, b in pairs: by[clus[k]].append(a - b)
    ks = list(by); rng = random.Random(0); v = []
    for _ in range(n):
        s = c = 0
        for q in (rng.choice(ks) for _ in ks): s += sum(by[q]); c += len(by[q])
        v.append(100 * s / c)
    v.sort(); return round(v[int(.025 * n)], 1), round(v[int(.975 * n) - 1], 1), len(ks)
res = {}
for fmt, root, cfgn in (("infprior", "api_prior", "E1f_prior"), ("infflip", "api_flip", "E1f_flip")):
    R = EXP / "results" / root
    if not R.exists(): continue
    its = {i["id"]: i for i in load_items_from_config(json.load(open(EXP / "configs" / f"{cfgn}.json")))}
    yes_idx = {k: [o.lower() for o in it["options"]].index("yes") for k, it in its.items()}
    clus = {k: (it["meta"].get("premise") or k) for k, it in its.items()}
    runs = {}
    for f in sorted(R.glob("*/*/samples.jsonl")): runs.setdefault(tok(f.parent.parent.name), {})[f.parent.name] = [json.loads(l) for l in open(f)]
    for model, doses in runs.items():
        base = min(doses, key=lambda d: DOSE_RANK[d])
        for sname, bks in (("a3-quant", {"f_a3q"}), ("a3-conn", {"f_a3c"}), ("a3", {"f_a3q", "f_a3c"})):
            sel = lambda r: its[r["item_id"]]["bucket"] in bks
            B = agg([r for r in doses[base] if sel(r)], yes_idx)
            for dose, rows in doses.items():
                rs = [r for r in rows if sel(r)]
                if not rs: continue
                D = agg(rs, yes_idx); n = len(rs); val = [r for r in rs if r["valid"]]
                node = {"n_calls": n, "fail": {"valid_share": round(100 * len(val) / n, 1), "invalid": round(100 * (n - len(val)) / n, 1), "trunc": round(100 * sum(r["finish"] not in ("stop", "end_turn") for r in rs) / n, 1)},
                        "R": {"mean": round(100 * sum(float(r["is_gold"]) for r in val) / len(val), 1), "n": len(val)} if val else None,
                        "yes": {"mean": round(100 * sum(r["choice"] == yes_idx[r["item_id"]] for r in val) / len(val), 1), "n": len(val)} if val else None}
                if dose != base:
                    for key in ("gold", "yes"):
                        both = [(k, D[k][key], B[k][key]) for k in D if k in B and D[k]["valid"] and B[k]["valid"]]
                        if both:
                            lo, hi, nc = boot(both, {k: clus[k] for k, _, _ in both}); node["d" + ("R" if key == "gold" else "Yes")] = {"mean": round(100 * sum(a - b for _, a, b in both) / len(both), 1), "lo": lo, "hi": hi, "n": len(both), "n_clusters": nc}
                res.setdefault(model, {}).setdefault(sname, {}).setdefault(fmt, {})[dose] = node
json.dump(res, open(EXP / "results/E1F/artefact_probes.json", "w"), indent=1)
sm = json.load(open(EXP / "results/E1F/summary.json"))
def dg(x, k=1):
    for a in ("mean", "lo", "hi"):
        if a in x: x[a + "__digits"] = k
    return x
for m, ss in res.items():
    for s, fm in ss.items():
        for f, ds in fm.items():
            for dose, node in ds.items():
                t = sm.setdefault(m, {}).setdefault(s, {}).setdefault(f, {}).setdefault(dose, {})
                t["n_calls"] = node["n_calls"]; t["fail"] = {k: {"mean": v, "mean__digits": 1} for k, v in node["fail"].items()}
                for k in ("R", "yes"):
                    if node.get(k): t[k] = dg(dict(node[k]))
                for k in ("dR", "dYes"):
                    if k in node: t[k] = dg(dict(node[k]))
json.dump(sm, open(EXP / "results/E1F/summary.json", "w"), indent=1)
for m, ss in res.items():
    for f, ds in ss["a3"].items():
        for dose, n in ds.items():
            print(f"{f:8s} {m:18s} {dose:9s} R={n['R'] and n['R']['mean']} yes={n['yes'] and n['yes']['mean']} valid={n['fail']['valid_share']}", ("dR %+.1f [%+.1f,%+.1f]  dYes %+.1f [%+.1f,%+.1f]" % (n['dR']['mean'], n['dR']['lo'], n['dR']['hi'], n['dYes']['mean'], n['dYes']['lo'], n['dYes']['hi'])) if 'dR' in n and 'dYes' in n else "")
