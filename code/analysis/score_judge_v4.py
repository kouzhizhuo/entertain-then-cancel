#!/usr/bin/env python3
"""Judge validation (open-weight judge = gpt-oss, labels_judge_<name>.jsonl from judge_pool.py): (1) human-judge agreement on the 50 V4 coding texts (the user's labels, V4 alone), post-stratified on the labeler-A class and
bootstrapped within A strata exactly as score_v4.py; (2) consistency of the judge with labeler A, labeler B (3 orders) and the A-B agreement on the full 294-text pool. Pre-declared rule (HUMAN_AUDIT_V4_meta.json): a judge is accepted only if
human-judge agreement >= 0.70 (kappa 3-class; lower CI reported). usage: python score_judge_v4.py labels_judge_gptoss20b.jsonl  -> gold/summary_judge_<name>.json"""
import json, random, sys
from collections import Counter
from pathlib import Path
G = Path(__file__).resolve().parents[1] / "gold"; name = sys.argv[1]
HM = {"pragmatic (enriched) reading": "P", "literal reading": "L", "hedged / both / neither": "H"}
AM = {"PRAG": "P", "CORRECT": "P", "LIT": "L", "BOTH": "H", "OTHER": "H", "OTHER-NL": "H"}
J = {r["id"]: HM[r["label"]] for r in map(json.loads, open(G / name))}
A = {r["sample_id"]: AM[r["label"]] for r in map(json.loads, open(G / "labels_A.jsonl"))}
B3 = {r["id"]: HM[r["label"]] for r in map(json.loads, open(G / "labels_B_perm3.jsonl"))}
K = json.load(open(G / "HUMAN_AUDIT_V4_key.json")); H = {r["id"]: HM.get(r["label"]) for r in map(json.loads, open(G / "HUMAN_AUDIT_V4_LABELS.jsonl"))}
pool = [json.loads(l)["id"] for l in open(G / "reference_pool.jsonl")]; NS = Counter(A[i] for i in pool)
def kap(x, y, w=None):
    w = w or [1.0] * len(x); t = sum(w); po = sum(wi for xi, yi, wi in zip(x, y, w) if xi == yi) / t
    px, py = Counter(), Counter()
    for xi, yi, wi in zip(x, y, w): px[xi] += wi / t; py[yi] += wi / t
    pe = sum(px[k] * py[k] for k in set(px) | set(py)); return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po
cod = [q for q, v in K.items() if v["kind"] == "coding"]; strat = {}
for q in cod: strat.setdefault(A[K[q]["pool_id"]], []).append(q)
def hj(items):
    n_s = Counter(A[K[q]["pool_id"]] for q in items); w = [NS[A[K[q]["pool_id"]]] / n_s[A[K[q]["pool_id"]]] for q in items]
    return kap([H[q] for q in items], [J[K[q]["pool_id"]] for q in items], w)
k, po = hj(cod); rng = random.Random(0); v = []
for _ in range(10000):
    it = [q for s in strat for q in (rng.choice(strat[s]) for _ in strat[s])]; x = hj(it)[0]
    if x == x: v.append(x)
v.sort()
def pool_kappa(P, Q):
    ids = sorted(set(P) & set(Q)); x = [P[i] for i in ids]; y = [Q[i] for i in ids]; r = random.Random(0); vals = []
    for _ in range(2000):
        ix = [r.randrange(len(ids)) for _ in ids]; z = kap([x[i] for i in ix], [y[i] for i in ix])[0]
        if z == z: vals.append(z)
    vals.sort(); return {"kappa": round(kap(x, y)[0], 3), "lo": round(vals[50], 3), "hi": round(vals[1949], 3), "agree": round(100 * kap(x, y)[1], 1), "n": len(ids)}
out = {"judge_file": name, "human_judge_V4": {"kappa_w": round(k, 3), "lo": round(v[250], 3), "hi": round(v[9749], 3), "agree_w": round(100 * po, 1), "n": len(cod), "passes_0.70": bool(k >= 0.70)},
       "judge_vs_A_pool": pool_kappa(J, A), "judge_vs_B3_pool": pool_kappa(J, B3), "A_vs_B3_pool": pool_kappa(A, B3), "judge_label_dist": dict(Counter(J.values()))}
json.dump(out, open(G / f"summary_judge_{name.replace('labels_judge_', '').replace('.jsonl', '')}.json", "w"), indent=1); print(json.dumps(out, indent=1))
