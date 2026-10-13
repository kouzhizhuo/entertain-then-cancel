#!/usr/bin/env python3
"""D-CONTINGENCY dev qualification (METHOD_DESIGN "D-CONTINGENCY"; ONE SHOT per model): the trained model in NAT mode must pass the v1 dev gates (lower bound of trained - base NAT above -2 pp on ix_can, imppres truth, folio, gsm8k) AND beat base NAT on the equal-family score. The control (equal-data unselected) is reported, never a fallback.
usage: python m1_select_D.py q3|r1     reads M1_dev_D/<name>_{base,selected,control} (+ M1_dev_D_gsm via D_dev_gsm), q3 base = M1_dev_v2/Qwen3-8B think_full; writes results/M1/selection_dev_D_<name>.json"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
X = Path(__file__).resolve().parents[1] / "results"; OUT = X / "M1"; OUT.mkdir(exist_ok=True)
name = sys.argv[1]
def find(g, m):
    for src in ("runs", "<host>", "slurm4"):
        p = X / src / g / m
        if (p / "samples.jsonl").exists(): return p
    sys.exit(f"missing {g}/{m}")
FAM = {"imppres_quant_fel": "speaker", "imppres_conn_fel": "speaker", "ludwig": "speaker", "circa": "speaker", "pub_task2": "speaker", "pub_task3": "speaker", "pub_task6": "speaker", "figqa": "speaker", "ix_rec": "speaker",
       "nope": "presup", "imppres_presup": "presup", "ix_can": "literal", "imppres_quant_truth": "literal", "imppres_conn_truth": "literal", "folio": "literal", "gsm8k": "literal"}
STRATA = {"ix_can": ["ix_can"], "imppres_truth": ["imppres_quant_truth", "imppres_conn_truth"], "folio": ["folio"], "gsm8k": ["gsm8k"]}
def load(fc_dir, gsm_dir, cond="think_full"):
    it = defaultdict(dict); tok = []
    for l in open(fc_dir / "samples.jsonl"):
        r = json.loads(l)
        if r["condition"] != cond: continue
        it[r["bucket"][2:]][r["item_id"]] = float(r["valid"] and r["is_gold"]); tok.append(r["total_tokens"])
    g = json.load(open(gsm_dir / "results.json"))["results"][cond]
    for l in open(gsm_dir / "samples.jsonl"):
        r = json.loads(l)
        if r["cond"] == cond: it["gsm8k"][r["id"]] = float(r["correct"])
    return it, float(np.mean(tok)), g["tokens_mean"]
if name == "q3": base = load(find("M1_dev_v2", "Qwen3-8B"), find("M1_dev_v2_gsm", "Qwen3-8B"))
else: base = load(find("M1_dev_D", "r1_base"), find("D_dev_gsm", "r1_base"))
def fam(it, f): return float(np.mean([100 * np.mean(list(it[s].values())) for s in FAM if FAM[s] == f and s in it]))
rng = np.random.default_rng(0)
def lb(a, b, strata):
    d = np.asarray([100 * (a[s][i] - b[s][i]) for s in strata for i in set(a[s]) & set(b[s])]); bm = d[rng.integers(0, len(d), (4000, len(d)))].mean(1); return float(np.percentile(bm, 5)), float(d.mean())
def row(it, t1, t2, ref=None):
    sp, li, pr = fam(it, "speaker"), fam(it, "literal"), fam(it, "presup")
    g = {k: lb(it, base[0], v) for k, v in STRATA.items()} if ref else {k: (0.0, 0.0) for k in STRATA}
    return {"speaker": sp, "literal": li, "presup": pr, "score": (sp + li) / 2, "tokens_forced_choice": t1, "tokens_gsm": t2, "gate_lb": g, "gates_pass": all(v[0] > -2 for v in g.values())}
tab = {"base_NAT": row(base[0], base[1], base[2])}; trn = {}
for kind in ("selected", "control", "nobal"):
    try: it, t1, t2 = load(find("M1_dev_D", f"{name}_{kind}"), find("D_dev_gsm", f"{name}_{kind}"))
    except SystemExit: continue
    tab[{"selected": "SIT-train_selected", "control": "control_unselected", "nobal": "SIT-nobal_ablation"}[kind]] = row(it, t1, t2, True)
    w2r = r2w = 0; per = {}
    for s in it:
        a, b = it[s], base[0][s]; k = set(a) & set(b); x = sum(b[i] == 0 and a[i] == 1 for i in k); y = sum(b[i] == 1 and a[i] == 0 for i in k); per[s] = [int(x), int(y), len(k)]
    trn[kind] = per
sel = tab.get("SIT-train_selected"); qual = bool(sel and sel["gates_pass"] and sel["score"] > tab["base_NAT"]["score"])
out = {"model": name, "qualifies": qual, "passes_v1_gates": bool(sel and sel["gates_pass"]), "beats_base_on_score": bool(sel and sel["score"] > tab["base_NAT"]["score"]), "control_is_never_fallback": True, "table": tab, "transitions_wrong_to_right_right_to_wrong_n": trn}
json.dump(out, open(OUT / f"selection_dev_D_{name}.json", "w"), indent=1)
print(f"{'arm':20s} speaker literal presup score tok_fc tok_gsm gates(ix_can imppres_truth folio gsm8k)")
for c, t in tab.items(): print(f"{c:20s} {t['speaker']:6.1f} {t['literal']:6.1f} {t['presup']:6.1f} {t['score']:6.1f} {t['tokens_forced_choice']:6.0f} {t['tokens_gsm']:6.0f} {'PASS' if t['gates_pass'] else 'fail'} " + " ".join(f"{v[0]:+.1f}" for v in t["gate_lb"].values()))
print("QUALIFIES:", qual)

if name == "q3":    # v3 export for the lead's tab:ablation (arms sit / control / nobal / nat; same fields as selection_dev_v2)
    ren = {"base_NAT": "nat", "SIT-train_selected": "sit", "control_unselected": "control", "SIT-nobal_ablation": "nobal"}
    v3 = {"model": "Qwen3-8B", "qualifies": qual, "qualification_rule": "SIT-train only (v1 dev gates AND beats base NAT on the equal-family score); control and nobal are never fallbacks", "table": {ren[k]: v for k, v in tab.items()},
          "transitions_wrong_to_right_right_to_wrong_n": {{"selected": "sit"}.get(k, k): v for k, v in trn.items()}}
    json.dump(v3, open(OUT / "selection_dev_v3_Qwen3-8B.json", "w"), indent=1); print("wrote selection_dev_v3_Qwen3-8B.json")
