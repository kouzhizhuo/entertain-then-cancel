#!/usr/bin/env python3
"""METHOD candidate selection on DEV (METHOD_DESIGN v1): reads the Qwen3-8B DEV run (results/<src>/M1_dev/Qwen3-8B/samples.jsonl + M1_dev_gsm/Qwen3-8B/{results.json,samples.jsonl}) and applies the pre-declared rule:
 score = mean(speaker-family accuracy, literal-gold-family accuracy) (family = macro over its sets); eligible = candidates whose every literal-gold stratum (ix_can, imppres truth quant/conn pooled, folio, gsm8k) has a one-sided 95% paired-bootstrap
 lower bound of (candidate - NAT) above -2 pp; winner = highest score among eligible, tie-break (within 0.5 pp) by fewer mean tokens; none eligible -> 'no safety-qualified main method'.
Also reports the strongest BASELINE on dev (same score, among NAT-512, user-exact, user-task-aware, Grice prefix, NT). Writes results/M1/selection_dev.json and prints a table."""
import json, sys, glob, random
from collections import defaultdict
from pathlib import Path
import numpy as np
X = Path(__file__).resolve().parents[1] / "results"; OUT = X / "M1"; OUT.mkdir(exist_ok=True)
def find(g, m):
    for src in ("runs", "<host>", "slurm4"):
        p = X / src / g / m
        if (p / "samples.jsonl").exists(): return p
    sys.exit(f"missing {g}/{m}")
model = sys.argv[1] if len(sys.argv) > 1 else "Qwen3-8B"
d = find("M1_dev", model); rows = [json.loads(l) for l in open(d / "samples.jsonl")]
FAM = {"imppres_quant_fel": "speaker", "imppres_conn_fel": "speaker", "ludwig": "speaker", "circa": "speaker", "pub_task2": "speaker", "pub_task3": "speaker", "pub_task6": "speaker", "figqa": "speaker", "ix_rec": "speaker",
       "nope": "presup", "imppres_presup": "presup", "ix_can": "literal", "imppres_quant_truth": "literal", "imppres_conn_truth": "literal", "folio": "literal", "gsm8k": "literal"}
STRATA = {"ix_can": ["ix_can"], "imppres_truth": ["imppres_quant_truth", "imppres_conn_truth"], "folio": ["folio"], "gsm8k": ["gsm8k"]}
acc = defaultdict(dict); items = defaultdict(lambda: defaultdict(dict)); toks = defaultdict(list)
for r in rows:
    if not r["valid"]: y = 0.0
    else: y = float(r["is_gold"])
    st = r["bucket"][2:]; acc[r["condition"]].setdefault(st, []).append(y); items[r["condition"]][st][r["item_id"]] = y; toks[r["condition"]].append(r["total_tokens"] if "total_tokens" in r else r.get("think_tokens", 0))
gd = find("M1_dev_gsm", model); gres = json.load(open(gd / "results.json"))["results"]; gs = defaultdict(dict)
for l in open(gd / "samples.jsonl"):
    r = json.loads(l); gs[r["cond"]][r["id"]] = float(r["correct"])
for c, v in gs.items(): items[c]["gsm8k"] = v; acc[c]["gsm8k"] = list(v.values())
gtok = {c: gres[c]["tokens_mean"] for c in gres}
conds = [c for c in acc]
def famscore(c, fam): return float(np.mean([100 * np.mean(acc[c][s]) for s in FAM if FAM[s] == fam and s in acc[c]]))
nat = "think_full"; rng = np.random.default_rng(0)
def lower_bound(c, strata):
    ids = None; dd = []
    for s in strata:
        a, b = items[c][s], items[nat][s]; k = sorted(set(a) & set(b)); dd += [100 * (a[i] - b[i]) for i in k]
    dd = np.asarray(dd); bm = dd[rng.integers(0, len(dd), (4000, len(dd)))].mean(1); return float(np.percentile(bm, 5)), float(dd.mean())
tab = {}
for c in conds:
    sp, li, pr = famscore(c, "speaker"), famscore(c, "literal"), famscore(c, "presup")
    lb = {k: lower_bound(c, v) for k, v in STRATA.items()} if c != nat else {k: (0.0, 0.0) for k in STRATA}
    mt = float(np.mean(toks[c])) if c in toks and toks[c] else 0.0; tab[c] = {"speaker": sp, "literal": li, "presup": pr, "score": (sp + li) / 2, "tokens_forced_choice": mt, "tokens_gsm": gtok.get(c), "gate_lb": lb, "gates_pass": all(v[0] > -2 for v in lb.values())}
cands = [c for c in ("seed_intent", "seed_dual", "seed_route") if c in tab]; elig = [c for c in cands if tab[c]["gates_pass"]]
best = None
if elig:
    top = max(tab[c]["score"] for c in elig); close = [c for c in elig if tab[c]["score"] >= top - 0.5]; best = min(close, key=lambda c: tab[c]["tokens_forced_choice"])
bases = [c for c in ("think_b512_plain", "think_full+sitroute", "think_full+taskaware", "seed_grice", "off") if c in tab]; bbest = max(bases, key=lambda c: tab[c]["score"]) if bases else None
out = {"model": model, "winner": best, "eligible": elig, "strongest_baseline": bbest, "table": tab}
json.dump(out, open(OUT / f"selection_dev_{model}.json", "w"), indent=1)
print(f"{'cond':24s} speaker literal presup score  tok_fc tok_gsm  gates(lb: " + " ".join(STRATA) + ")")
for c in sorted(tab, key=lambda c: -tab[c]["score"]):
    t = tab[c]; print(f"{c:24s} {t['speaker']:6.1f} {t['literal']:6.1f} {t['presup']:6.1f} {t['score']:6.1f} {t['tokens_forced_choice']:7.0f} {t['tokens_gsm'] or 0:7.0f}  {'PASS' if t['gates_pass'] else 'fail'} " + " ".join(f"{v[0]:+.1f}" for v in t["gate_lb"].values()))
print("WINNER:", best or "none (no safety-qualified main method)", "| strongest baseline:", bbest)
