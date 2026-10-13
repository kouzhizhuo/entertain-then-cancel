#!/usr/bin/env python3
"""Score the V4 human audit (HUMAN_AUDIT_V4_LABELS.jsonl, user's labels entered by an agent on the user's attestation) by the rules pre-declared in HUMAN_AUDIT_V4_meta.json (pooling_rule_predeclared).
V4 ALONE (V1 was LLM-filled; the pooled V1+V4 rule collapses to V4 only: disclosed). Coding: post-stratified on the labeler-A class, w_s = N_s(pool)/n_s, bootstrap within A strata (10,000), unweighted + binary (hedged excluded)
sensitivities, per-class accuracy; reference raters A, B (fixed order), B-perm3. A1-ctx: accuracy vs intended (unclear = wrong) + 'unclear excluded', Wilson 95%, full-minus-partial 'misleading' gap bootstrap.
Judges: no open-answer judge codes exist yet, so human-judge agreement is NOT computed (stated, not filled). Writes exp/gold/summary_human_v4.json (key format mean/lo/hi/n)."""
import json, random, math
from collections import Counter, defaultdict
from pathlib import Path
G = Path(__file__).resolve().parents[1] / "gold"
K = json.load(open(G / "HUMAN_AUDIT_V4_key.json")); H = {r["id"]: r["label"] for r in map(json.loads, open(G / "HUMAN_AUDIT_V4_LABELS.jsonl"))}
HM = {"pragmatic (enriched) reading": "P", "literal reading": "L", "hedged / both / neither": "H"}
AM = {"PRAG": "P", "CORRECT": "P", "LIT": "L", "BOTH": "H", "OTHER": "H", "OTHER-NL": "H"}
A = {r["sample_id"]: AM[r["label"]] for r in map(json.loads, open(G / "labels_A.jsonl"))}
B1 = {r["id"]: HM[r["label"]] for r in map(json.loads, open(G / "labels_B.jsonl"))}; B3 = {r["id"]: HM[r["label"]] for r in map(json.loads, open(G / "labels_B_perm3.jsonl"))}
pool_ids = [json.loads(l)["id"] for l in open(G / "reference_pool.jsonl")]; NS = Counter(A[i] for i in pool_ids if i in A)
cod = sorted(q for q, v in K.items() if v["kind"] == "coding"); ctx = sorted(q for q, v in K.items() if v["kind"] == "a1ctx")
assert set(H) == set(K), "label/key id mismatch"
rows = [(A[K[q]["pool_id"]], HM[H[q]], {"A": A, "B": B1, "B3": B3}) for q in cod]
strat = defaultdict(list)
for q in cod: strat[A[K[q]["pool_id"]]].append(q)
def kap_w(items, ref):    # items: list of (q); weights by A stratum
    n_s = Counter(A[K[q]["pool_id"]] for q in items); w = {s: NS[s] / n_s[s] for s in n_s}
    cm = defaultdict(float); tot = 0.0
    for q in items:
        s = A[K[q]["pool_id"]]; h = HM[H[q]]; r = ref[K[q]["pool_id"]]; cm[(h, r)] += w[s]; tot += w[s]
    po = sum(v for (h, r), v in cm.items() if h == r) / tot
    ph = Counter(); pr = Counter()
    for (h, r), v in cm.items(): ph[h] += v / tot; pr[r] += v / tot
    pe = sum(ph[k] * pr[k] for k in set(ph) | set(pr)); return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), po
def kap_u(items, ref, binary=False):
    pairs = [(HM[H[q]], ref[K[q]["pool_id"]]) for q in items]
    if binary: pairs = [(h, r) for h, r in pairs if h != "H" and r != "H"]
    n = len(pairs); po = sum(h == r for h, r in pairs) / n; ch = Counter(h for h, _ in pairs); cr = Counter(r for _, r in pairs)
    pe = sum(ch[k] * cr[k] for k in set(ch) | set(cr)) / n ** 2; return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), po, n
def boot(fn, n=10000, seed=0):
    rng = random.Random(seed); v = []
    for _ in range(n):
        items = [q for s in strat for q in (rng.choice(strat[s]) for _ in strat[s])]
        k = fn(items)
        if k == k: v.append(k)
    v.sort(); return round(v[int(.025 * len(v))], 3), round(v[int(.975 * len(v)) - 1], 3)
def blk(mean, ci, n, d=2): return {"mean": round(mean, d), "lo": ci[0], "hi": ci[1], "n": n, "mean__digits": d, "lo__digits": d, "hi__digits": d}
out = {"n_coding": len(cod), "human_dist": dict(Counter(HM[H[q]] for q in cod)), "strata_A": dict(Counter(A[K[q]["pool_id"]] for q in cod)), "pool_N_by_A": dict(NS), "coding": {}}
for nm, ref in (("A", A), ("B", B1), ("B3", B3)):
    k, po = kap_w(cod, ref); ci = boot(lambda it: kap_w(it, ref)[0]); acc = boot(lambda it: kap_w(it, ref)[1])
    ku, pou, _ = kap_u(cod, ref); cu = boot(lambda it: kap_u(it, ref)[0]); kb, pob, nb = kap_u(cod, ref, True); cb = boot(lambda it: kap_u(it, ref, True)[0])
    out["coding"][nm] = {"kappa_w": blk(k, ci, len(cod)), "agree_w": blk(100 * po, (round(100 * acc[0], 1), round(100 * acc[1], 1)), len(cod), 1),
                         "kappa_unweighted": blk(ku, cu, len(cod)), "kappa_binary_no_hedge": blk(kb, cb, nb),
                         "consistent_with_human_(lower_CI>=0.40)": ci[0] >= 0.40}
# per-class accuracy, reference = A, within A stratum (unweighted)
pc = {}
for s, qs in strat.items():
    acc = sum(HM[H[q]] == s for q in qs) / len(qs); r = random.Random(1); v = sorted(sum(HM[H[r.choice(qs)]] == s for _ in qs) / len(qs) for _ in range(5000))
    pc[s] = {"acc": round(100 * acc, 1), "lo": round(100 * v[125], 1), "hi": round(100 * v[4874], 1), "n": len(qs)}
out["per_class_accuracy_vs_A"] = pc
# A1-ctx
def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)); return round((c - r) / d, 3), round((c + r) / d, 3)
ok = [H[q] == K[q]["intended"] for q in ctx]; ex = [H[q] == K[q]["intended"] for q in ctx if H[q] != "unclear"]
out["a1ctx"] = {"n": len(ctx), "acc_unclear_wrong": {"mean": round(100 * sum(ok) / len(ok), 1), "wilson": [round(100 * x, 1) for x in wilson(sum(ok), len(ok))]},
                "acc_unclear_excluded": {"mean": round(100 * sum(ex) / len(ex), 1), "n": len(ex), "wilson": [round(100 * x, 1) for x in wilson(sum(ex), len(ex))]},
                "label_dist": dict(Counter(H[q] for q in ctx))}
full = [q for q in ctx if K[q]["subset"] == "felicity_full"]; part = [q for q in ctx if K[q]["subset"] == "felicity_partial"]
f = lambda qs: sum(H[q] == "yes, misleading" for q in qs) / len(qs)
rng = random.Random(2); g = sorted(f([rng.choice(full) for _ in full]) - f([rng.choice(part) for _ in part]) for _ in range(10000))
out["a1ctx"]["gap_misleading_full_minus_partial"] = {"mean": round(100 * (f(full) - f(part)), 1), "lo": round(100 * g[250], 1), "hi": round(100 * g[9749], 1)}
out["a1ctx"]["bar_pass (acc>=80 and Wilson lower>=70)"] = bool(100 * sum(ok) / len(ok) >= 80 and 100 * wilson(sum(ok), len(ok))[0] >= 70)
out["judge_agreement"] = "not computed: no open-answer judge codes exist yet (F-open judge not run); do not fill"
out["provenance"] = "user attestation relayed by the coordinator 2026-10-02: choices made by the user, typed by an agent (openpyxl); reported as human labels on that attestation"
json.dump(out, open(G / "summary_human_v4.json", "w"), indent=1); print(json.dumps(out, indent=1))
