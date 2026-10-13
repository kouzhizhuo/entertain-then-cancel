#!/usr/bin/env python3
"""P2 X3 held-out TEST (notes/DECISIONS.md X3 rows; EXPLORATORY, dev-selected). usage: python x3_test.py [dev|test]
Arms per model: NAT; <email> (routed items answer from the forced answer after the first 128 thinking tokens of the same NAT trace);
<email> (routed items answer from no-thinking); task-aware instruction (user turn; existing runs). Router v2 exactly as x_remedies_x3.py.
TEST RULE: pooled over the two models, the paired 95% CI (2.5/97.5 percentiles, 4000 item bootstraps within set) of the
equal-family score difference vs NAT excludes 0 AND every safety stratum's one-sided 95% lower bound (pooled) > -2 pp."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
X = Path(__file__).resolve().parents[1]; R = X / "results"
split = sys.argv[1] if len(sys.argv) > 1 else "test"
sys.path.insert(0, str(X / "harness")); from common import render_user
FAM = {"imppres_quant_fel": "speaker", "imppres_conn_fel": "speaker", "ludwig": "speaker", "circa": "speaker", "pub_task2": "speaker", "pub_task3": "speaker",
       "pub_task6": "speaker", "figqa": "speaker", "ix_rec": "speaker", "nope": "presup", "imppres_presup": "presup", "ix_can": "literal",
       "imppres_quant_truth": "literal", "imppres_conn_truth": "literal", "folio": "literal", "gsm8k": "literal"}
STRATA = {"ix_can": ["ix_can"], "imppres_truth": ["imppres_quant_truth", "imppres_conn_truth"], "folio": ["folio"], "gsm8k": ["gsm8k"]}
Q3 = re.compile(r"conclude that the speaker believes", re.I); UTT = re.compile(r'A speaker says:\s*"([^"]*)"')
SOME = re.compile(r"\b(some|or)\b", re.I); EXCL = re.compile(r"\b(all|both|at least|not necessarily|in fact|possibly|maybe|don't know|do not know)\b", re.I)
items = {}
for l in open(X / "data" / "items" / f"method_{split}.jsonl"):
    it = json.loads(l); items[it["id"]] = it
def routed(it):
    t = render_user(it, list(range(len(it["options"]))), "default", None).split("\n\nOptions:")[0]; m = UTT.search(t)
    return bool(Q3.search(t)) and m is not None and bool(SOME.search(m.group(1))) and not EXCL.search(m.group(1))
def find(*cands, opt=False):
    for c in cands:
        for src in ("runs", "<host>", "slurm4"):
            p = R / src / c / "samples.jsonl"
            if p.exists(): return p
    if opt: return None
    sys.exit(f"missing {cands}")
MODELS = {"q3": ("Qwen3-8B", "off"), "r1": ("DeepSeek-R1-0528-Qwen3-8B", "think_b0_plain")}
acc = {}; info = {}
for mk, (mname, NT) in MODELS.items():
    rows = defaultdict(dict)
    xdir = f"M1_{split}_X3/{mk}" if split == "test" else f"M1_dev_X/{mk}"
    for l in open(find(xdir)):
        r = json.loads(l); rows[r["condition"]][r["item_id"]] = r
    ta = {}
    tp = find(f"M1_{split}/{mname}", f"M1_{split}/{mname.replace('DeepSeek-', '')}", f"M1_dev/{mname}" if split == "dev" else "_", opt=True)
    for l in (open(tp) if tp else []):
        r = json.loads(l)
        if r["condition"] == "think_full+taskaware": ta[r["item_id"]] = r
    A = {a: defaultdict(dict) for a in ("nat", "x1", "x2", "taskaware")}; nr = defaultdict(int)
    for i, nat in rows["think_full"].items():
        it = items.get(i)
        if it is None: continue
        s = it["meta"]["set"]; g = it["gold"]; rt = routed(it); nr[s] += rt
        A["nat"][s][i] = float(nat["valid"] and nat["choice"] == g)
        for a, cond in (("x1", "think_b128_plain"), ("x2", NT)):
            r = rows[cond].get(i)
            use = rt and r is not None and r["valid"] and (a == "x2" or nat["think_tokens"] > 128)
            A[a][s][i] = float((r["choice"] if use else nat["choice"] if nat["valid"] else -1) == g)
        t = ta.get(i); A["taskaware"][s][i] = float(t is not None and t["valid"] and t["choice"] == g)
    gs = {}; gta = {}
    gp = find(f"M1_{split}_gsm/{mname}") if split == "test" else find("M1_dev_v2_gsm/Qwen3-8B" if mk == "q3" else "D_dev_gsm/r1_base")
    for l in open(gp):
        r = json.loads(l)
        if r["cond"] == "think_full": gs[r["id"]] = float(r["correct"])
        if r["cond"] == "think_full+taskaware": gta[r["id"]] = float(r["correct"])
    for a in A: A[a]["gsm8k"] = dict(gta if (a == "taskaware" and gta) else gs)
    acc[mk] = A; info[mk] = {"routed": dict(nr), "gsm_taskaware_available": bool(gta)}
rng = np.random.default_rng(0); B = 4000
def score_draws(a):
    """bootstrap draws of the equal-family score difference (arm - NAT), pooled = mean over models"""
    tot = np.zeros(B); point = 0.0
    for mk, A in acc.items():
        fam = defaultdict(list)
        for s in FAM:
            if s not in A["nat"] or not A["nat"][s]: continue
            ids = list(A["nat"][s]); d = np.array([100 * (A[a][s][i] - A["nat"][s][i]) for i in ids])
            fam[FAM[s]].append((d.mean(), d[rng.integers(0, len(d), (B, len(d)))].mean(1)))
        sp = [x for x in fam["speaker"]]; li = [x for x in fam["literal"]]
        pm = (np.mean([x[0] for x in sp]) + np.mean([x[0] for x in li])) / 2
        bd = (np.mean([x[1] for x in sp], 0) + np.mean([x[1] for x in li], 0)) / 2
        tot += bd / len(acc); point += pm / len(acc)
    return point, tot
def gate(a, strata):
    d = np.concatenate([np.array([100 * (A[a][s][i] - A["nat"][s][i]) for s in strata for i in A["nat"].get(s, {})]) for A in acc.values()])
    bm = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return float(d.mean()), float(np.percentile(bm, 5))
def fammean(A, a, f): return float(np.mean([100 * np.mean(list(A[a][s].values())) for s in FAM if FAM[s] == f and A[a].get(s)]))
out = {"split": split, "routed": info, "arms": {}}
for a in ("x1", "x2", "taskaware"):
    pt, dr = score_draws(a); lo, hi = np.percentile(dr, [2.5, 97.5]); g = {k: gate(a, v) for k, v in STRATA.items()}
    out["arms"][a] = {"score_diff": pt, "ci": [float(lo), float(hi)], "ci_lo": float(lo), "ci_hi": float(hi), "gates": {k: {"diff": v[0], "lb": v[1]} for k, v in g.items()},
                      "passes_test_rule": bool(lo > 0 and all(v[1] > -2 for v in g.values())),
                      "per_model": {mk: {f: fammean(A, a, f) for f in ("speaker", "presup", "literal")} for mk, A in acc.items()}}
out["nat_per_model"] = {mk: {f: fammean(A, "nat", f) for f in ("speaker", "presup", "literal")} for mk, A in acc.items()}
out["felicity_per_model"] = {mk: {a: {s: 100 * float(np.mean(list(A[a][s].values()))) for s in ("imppres_quant_fel", "imppres_conn_fel") if A[a].get(s)} for a in A} for mk, A in acc.items()}
json.dump(out, open(R / "M1" / f"x3_{split}.json", "w"), indent=1)
for a, v in out["arms"].items(): print(a, f"diff {v['score_diff']:+.2f} CI [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]", {k: round(g['lb'], 1) for k, g in v["gates"].items()}, "PASS" if v["passes_test_rule"] else "fail")
print("felicity:", json.dumps(out["felicity_per_model"])[:600])
