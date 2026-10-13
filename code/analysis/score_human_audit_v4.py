"""Score the V4 human audit strictly by the rules pre-declared in exp/gold/HUMAN_AUDIT_V4_meta.json (pooling: V4 alone).
Coding: 3-class kappa/accuracy, human vs LLM reference A, B, B-perm3, post-stratified on the A label (w_s = N_s(pool)/n_s(sample)), bootstrap within A stratum (10,000);
per-class accuracy (A = PRAG, LIT) unweighted; unweighted overall and binary (hedged excluded) as sensitivity checks. Judges: none have coded the pool -> not evaluable.
A1-ctx: accuracy vs intended ('unclear' = wrong; sensitivity: excluded), Wilson 95% lower bound, full-minus-partial 'misleading' gap with bootstrap CI.
Writes exp/gold/summary_human_v4_tex.json (json2tex-ready, group 'v4')."""
import json, math, random
from collections import Counter, defaultdict
from pathlib import Path
G = Path(__file__).resolve().parents[1] / "gold"
K = json.load(open(G / "HUMAN_AUDIT_V4_key.json"))
H = {r["id"]: r for r in map(json.loads, open(G / "HUMAN_AUDIT_V4_LABELS.jsonl"))}
assert set(H) == set(K) and all(r["source"] == "human" for r in H.values())
O = {"pragmatic (enriched) reading": "P", "literal reading": "L", "hedged / both / neither": "H"}
AM = lambda l: {"PRAG": "P", "CORRECT": "P", "LIT": "L"}.get(l, "H")
A = {r["sample_id"]: AM(r["label"]) for r in map(json.loads, open(G / "labels_A.jsonl"))}
B1 = {r["id"]: O.get(r["label"], "H") for r in map(json.loads, open(G / "labels_B.jsonl"))}
B3 = {r["id"]: O.get(r["label"], "H") for r in map(json.loads, open(G / "labels_B_perm3.jsonl"))}
Npool = Counter({"P": "PRAG", "L": "LIT", "H": "HEDGE"}[v] for v in A.values())
cod = [(q, k) for q, k in K.items() if k["kind"] == "coding"]
nS = Counter(k["stratum_A"] for _, k in cod); W = {s: Npool[s] / nS[s] for s in nS}
def wkappa(rows):  # rows: (h, ref, w)
    tw = sum(w for *_, w in rows); po = sum(w for h, r, w in rows if h == r) / tw
    ch, cr = defaultdict(float), defaultdict(float)
    for h, r, w in rows: ch[h] += w; cr[r] += w
    pe = sum(ch[c] * cr[c] for c in ch) / tw / tw
    return ((po - pe) / (1 - pe) if pe < 1 else 0.0), 100 * po
def sboot(rows_by_s, f, R=10000):
    rs = random.Random(7); v = []
    for _ in range(R):
        smp = [x for rows in rows_by_s.values() for x in (rows[rs.randrange(len(rows))] for _ in rows)]
        v.append(f(smp))
    v.sort(); return v[int(.025 * R)], v[int(.975 * R) - 1]
def ci(m, lohi, n, d=2): return {"mean": round(m, d), "lo": round(lohi[0], d), "hi": round(lohi[1], d), "n": n, "mean__digits": d, "lo__digits": d, "hi__digits": d}
v4 = {"n-coding": len(cod), "n-a1ctx": sum(k["kind"] == "a1ctx" for k in K.values())}
for nm, ref in (("a", A), ("b", B1), ("b3", B3)):
    by = defaultdict(list)
    for q, k in cod: by[k["stratum_A"]].append((O[H[q]["label"]], ref[k["pool_id"]], W[k["stratum_A"]]))
    rows = [x for r in by.values() for x in r]
    kw, aw = wkappa(rows)
    v4[f"kappa-human-{nm}"] = ci(kw, sboot(by, lambda s: wkappa(s)[0]), len(rows))
    v4[f"acc-human-{nm}"] = ci(aw, sboot(by, lambda s: wkappa(s)[1]), len(rows), 1)
    un = [(h, r, 1.0) for h, r, _ in rows]; ku, au = wkappa(un)
    v4[f"kappa-human-{nm}-unw"] = ci(ku, sboot({"all": un}, lambda s: wkappa(s)[0]), len(un))
    bn = [x for x in un if "H" not in x[:2]]; kb, ab = wkappa(bn)
    v4[f"kappa-human-{nm}-bin"] = ci(kb, sboot({"all": bn}, lambda s: wkappa(s)[0]), len(bn))
    v4[f"conf-{nm}"] = dict(Counter(f"{h}{r}" for h, r, _ in rows))
for s, c in (("PRAG", "P"), ("LIT", "L"), ("HEDGE", "H")):   # per-class: does the human agree with A within A's class
    xs = [O[H[q]["label"]] == c for q, k in cod if k["stratum_A"] == s]
    rs = random.Random(7); bs = sorted(100 * sum(rs.choice(xs) for _ in xs) / len(xs) for _ in range(10000))
    v4[f"acc-class-{s.lower()}"] = ci(100 * sum(xs) / len(xs), (bs[250], bs[9749]), len(xs), 1)
v4["human-dist-coding"] = dict(Counter(O[H[q]["label"]] for q, _ in cod))
ctx = [(H[q]["label"], k["intended"], k["subset"]) for q, k in K.items() if k["kind"] == "a1ctx"]
def wilson(x, n, z=1.96):
    p = x / n; return 100 * (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)
def bt(xs, f, R=10000):
    rs = random.Random(7); v = sorted(f([rs.choice(xs) for _ in xs]) for _ in range(R)); return v[int(.025 * R)], v[int(.975 * R) - 1]
acc = lambda s: 100 * sum(h == t for h, t, _ in s) / len(s)
x = sum(h == t for h, t, _ in ctx)
v4["a1ctx-acc"] = {**ci(acc(ctx), bt(ctx, acc), len(ctx), 1), "wilsonlo": round(wilson(x, len(ctx)), 1)}
ex = [c for c in ctx if c[0] != "unclear"]; xe = sum(h == t for h, t, _ in ex)
v4["a1ctx-acc-exunclear"] = {**ci(acc(ex), bt(ex, acc), len(ex), 1), "wilsonlo": round(wilson(xe, len(ex)), 1)}
def gap(s):
    f = [h == "yes, misleading" for h, _, a in s if a == "felicity_full"]; p = [h == "yes, misleading" for h, _, a in s if a == "felicity_partial"]
    return 100 * (sum(f) / max(1, len(f)) - sum(p) / max(1, len(p)))
full = [c for c in ctx if c[2] == "felicity_full"]; part = [c for c in ctx if c[2] == "felicity_partial"]
rs = random.Random(7); gb = sorted(gap([rs.choice(full) for _ in full] + [rs.choice(part) for _ in part]) for _ in range(10000))
v4["a1ctx-gap"] = ci(gap(ctx), (gb[250], gb[9749]), len(ctx), 1)
v4["a1ctx-labels"] = {a: dict(Counter(h for h, _, s in ctx if s == a)) for a in ("felicity_full", "felicity_partial")}
v4["judges"] = "not evaluable: no judge has coded the reference pool (no judge codes exist)"
dec = {"a1ctx_valid": v4["a1ctx-acc"]["mean"] >= 80 and v4["a1ctx-acc"]["wilsonlo"] >= 70 and v4["a1ctx-gap"]["lo"] > 0}
for nm in ("a", "b", "b3"): dec[f"rater_{nm}_consistent_with_human"] = v4[f"kappa-human-{nm}"]["lo"] >= 0.40
dec["judge_accepted"] = False
v4["decisions"] = dec
json.dump({"v4": v4}, open(G / "summary_human_v4_full.json", "w"), indent=1)
json.dump({"v4": {k: v for k, v in v4.items() if not (k.startswith("conf-") or k in ("a1ctx-labels", "human-dist-coding", "judges", "decisions"))}}, open(G / "summary_human_v4_tex.json", "w"), indent=1)
print(json.dumps(v4, indent=1))
