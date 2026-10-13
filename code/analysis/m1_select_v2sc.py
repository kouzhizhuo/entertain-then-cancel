#!/usr/bin/env python3
"""SIT v2 DEV qualification (METHOD_DESIGN AMENDMENT v2; ONE SHOT): SIT-check (late_sitcheck) qualifies only if it passes the v1 dev gates AND beats NAT (think_full) on the equal-family score. The neutral check is a control, never a fallback. Adds token accounting and paired transitions vs NAT. Reads M1_dev_v2 / M1_dev_v2_gsm; writes results/M1/selection_dev_v2_<model>.json.
(v1 docstring follows) METHOD candidate selection on DEV (METHOD_DESIGN v1): reads the Qwen3-8B DEV run (results/<src>/M1_dev/Qwen3-8B/samples.jsonl + M1_dev_gsm/Qwen3-8B/{results.json,samples.jsonl}) and applies the pre-declared rule:
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
d = find("M1_dev_SC", "q3"); rows = [json.loads(l) for l in open(d / "samples.jsonl")]
FAM = {"imppres_quant_fel": "speaker", "imppres_conn_fel": "speaker", "ludwig": "speaker", "circa": "speaker", "pub_task2": "speaker", "pub_task3": "speaker", "pub_task6": "speaker", "figqa": "speaker", "ix_rec": "speaker",
       "nope": "presup", "imppres_presup": "presup", "ix_can": "literal", "imppres_quant_truth": "literal", "imppres_conn_truth": "literal", "folio": "literal", "gsm8k": "literal"}
STRATA = {"ix_can": ["ix_can"], "imppres_truth": ["imppres_quant_truth", "imppres_conn_truth"], "folio": ["folio"], "gsm8k": ["gsm8k"]}
acc = defaultdict(dict); items = defaultdict(lambda: defaultdict(dict)); toks = defaultdict(list)
for r in rows:
    if not r["valid"]: y = 0.0
    else: y = float(r["is_gold"])
    st = r["bucket"][2:]; acc[r["condition"]].setdefault(st, []).append(y); items[r["condition"]][st][r["item_id"]] = y; toks[r["condition"]].append(r["total_tokens"] if "total_tokens" in r else r.get("think_tokens", 0))
gd = find("M1_dev_v2_gsm", model); gres = json.load(open(gd / "results.json"))["results"]; gs = defaultdict(dict)
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
SIT, CTRL = "late_sitcheck", "late_neutral"
beats = tab[SIT]["score"] > tab[nat]["score"]; gates = tab[SIT]["gates_pass"]; qualifies = bool(beats and gates)
bases = [c for c in ("think_b512_plain", "think_full+taskaware", "off") if c in tab]; bbest = max(bases, key=lambda c: tab[c]["score"])
# ---- paired transitions vs NAT (wrong->right / right->wrong), by family and for named sets
SETS = ["circa", "ix_rec", "pub_task2", "nope", "imppres_presup", "imppres_conn_truth", "folio", "gsm8k"]
def trans(c, strata):
    w2r = r2w = n = 0
    for s_ in strata:
        a, b = items[c][s_], items[nat][s_]
        for i_ in set(a) & set(b):
            n += 1; w2r += (b[i_] == 0 and a[i_] == 1); r2w += (b[i_] == 1 and a[i_] == 0)
    return {"n": n, "wrong_to_right": w2r, "right_to_wrong": r2w}
trn = {c: {"family": {f: trans(c, [s_ for s_ in FAM if FAM[s_] == f and s_ in items[c]]) for f in ("speaker", "presup", "literal")}, "sets": {s_: trans(c, [s_]) for s_ in SETS if s_ in items[c]}} for c in (SIT, CTRL, "think_full+taskaware") if c in items}
# ---- tokens before/after insertion and cap share (forced-choice rows carry inj_tokens / stage1_finish)
tk = defaultdict(lambda: defaultdict(list))
for r in rows:
    if r["condition"] in (nat, SIT, CTRL): tk[r["condition"]]["total"].append(r["total_tokens"]); tk[r["condition"]]["inj"].append(r.get("inj_tokens", 0)); tk[r["condition"]]["capped"].append(r.get("stage1_finish") != "stop")
tokacc = {c: {"total_mean": float(np.mean(v["total"])), "injected_mean": float(np.mean(v["inj"])), "before_insertion_mean": float(np.mean(v["total"]) - np.mean(v["inj"])), "cap_share_pct": 100 * float(np.mean(v["capped"])), "insertion_applied_pct": 100 * float(np.mean([x > 0 for x in v["inj"]]))} for c, v in tk.items()}
gsmtok = {c: {k: gres[c].get(k) for k in ("tokens_mean", "inj_tokens_mean", "pre_tokens_mean", "unfinished_share")} for c in gres}
out = {"model": model, "qualifies": qualifies, "beats_nat_on_score": bool(beats), "passes_v1_gates": bool(gates), "strongest_baseline": bbest, "neutral_is_never_fallback": True, "table": tab, "transitions_vs_nat": trn, "tokens_forced_choice": tokacc, "tokens_gsm": gsmtok}
json.dump(out, open(OUT / f"selection_dev_v2sc_{model}.json", "w"), indent=1)
print(f"{'cond':24s} speaker literal presup score  tok_fc tok_gsm  gates(lb: " + " ".join(STRATA) + ")")
for c in sorted(tab, key=lambda c: -tab[c]["score"]):
    t = tab[c]; print(f"{c:24s} {t['speaker']:6.1f} {t['literal']:6.1f} {t['presup']:6.1f} {t['score']:6.1f} {t['tokens_forced_choice']:7.0f} {t['tokens_gsm'] or 0:7.0f}  {'PASS' if t['gates_pass'] else 'fail'} " + " ".join(f"{v[0]:+.1f}" for v in t["gate_lb"].values()))
print("tokens (forced choice):", json.dumps(tokacc)); print("tokens (gsm):", json.dumps(gsmtok))
for c, v in trn.items(): print("transitions vs NAT", c, json.dumps(v))
print("SIT-check passes v1 gates:", gates, "| beats NAT on score:", beats, "| QUALIFIES:", qualifies, "| strongest baseline:", bbest)
