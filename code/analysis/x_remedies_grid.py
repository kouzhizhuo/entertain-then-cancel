#!/usr/bin/env python3
"""P2 EXPLORATORY ROUND 2 (X1′/X2′ weight grid, DECISIONS.md 2026-10-08) -- derived from x_remedies.py; dev (notes/DECISIONS.md "P2 EXPLORATORY", frozen; NOT pre-registered). usage: python x_remedies.py q3|r1
Inputs: results/*/M1_dev_X/<name>/samples.jsonl (conditions NT [off | think_b0_plain], think_full, think_b128_plain; p_canon = readout probabilities in canonical option order) + data/items/method_dev.jsonl
        + the cached NAT GSM8K dev run (GSM8K is a math task: always NAT).
Routing (RE-FROZEN, INPUT-ONLY, DECISIONS.md 2026-10-07 12:27): regexes on the rendered prompt minus the answer options; the question asks mean/imply/intend/suggest/conclude/infer, NOT true/false/entail/follows/necessarily/valid/calculate/how many, contains whole-word some|or, and none of the exclusion phrases. No set/category/gold field is read for routing. Everything else (and GSM8K, which has no forced-choice prompt) = cached NAT.
X1: argmax 0.8 p_NAT + 0.2 p_128 (p_128 = forced answer after the first 128 thinking tokens of the SAME cached NAT trace, plain native close; NAT closed before 128 tokens -> p_128 = p_NAT by construction -> NAT). Control x1_w0: weight 0.
X2: argmax 0.9 log p_NAT + 0.1 log p_NT, floor 1e-6. Control x2_w0: weight 0. Invalid readouts (no label probabilities) fall back to NAT.
Gate (as m1_select): beat NAT on the equal-family score = mean(speaker, literal) AND one-sided 95% lower bounds of (arm - NAT) > -2 pp on ix_can, imppres truth, folio, gsm8k, over ALL items. Output results/M1/selection_dev_X_<name>.json"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
X = Path(__file__).resolve().parents[1]; R = X / "results"; OUT = R / "M1"; OUT.mkdir(exist_ok=True)
name = sys.argv[1]; NT = "off" if name == "q3" else "think_b0_plain"
def find(g, m):
    for src in ("runs", "<host>", "slurm4"):
        p = R / src / g / m
        if (p / "samples.jsonl").exists(): return p
    sys.exit(f"missing {g}/{m}")
items = {}
for l in open(X / "data" / "items" / "method_dev.jsonl"):
    it = json.loads(l); items[it["id"]] = it
rows = defaultdict(dict)
for l in open(find("M1_dev_X", name) / "samples.jsonl"):
    r = json.loads(l); rows[r["condition"]][r["item_id"]] = r
FAM = {"imppres_quant_fel": "speaker", "imppres_conn_fel": "speaker", "ludwig": "speaker", "circa": "speaker", "pub_task2": "speaker", "pub_task3": "speaker", "pub_task6": "speaker", "figqa": "speaker", "ix_rec": "speaker",
       "nope": "presup", "imppres_presup": "presup", "ix_can": "literal", "imppres_quant_truth": "literal", "imppres_conn_truth": "literal", "folio": "literal", "gsm8k": "literal"}
STRATA = {"ix_can": ["ix_can"], "imppres_truth": ["imppres_quant_truth", "imppres_conn_truth"], "folio": ["folio"], "gsm8k": ["gsm8k"]}
# ROUTER (re-frozen INPUT-ONLY, notes/DECISIONS.md 2026-10-07 12:27): regexes on the RENDERED PROMPT text only; no dataset / set / category / gold field is read.
sys.path.insert(0, str(X / "harness")); from common import render_user
Q_YES = re.compile(r"\b(mean|means|meant|imply|implies|implied|intend|intends|intended|suggest|suggests|suggesting|conclude|infer)\b", re.I)
Q_NO = re.compile(r"\b(true|false|entail|entails|follows|necessarily|valid|calculate|how many)\b", re.I)
SOME = re.compile(r"\b(some|or)\b", re.I); EXCL = re.compile(r"\b(all|both|at least|not necessarily|in fact|possibly|maybe|don't know|do not know)\b", re.I)
def prompt_minus_options(it):
    p = render_user(it, list(range(len(it["options"]))), "default", None)
    return p.split("\n\nOptions:")[0]
def routed(it):
    t = prompt_minus_options(it)
    return bool(Q_YES.search(t)) and not Q_NO.search(t) and bool(SOME.search(t)) and not EXCL.search(t)
def pick(p):   # canonical option index with the highest probability
    return int(np.argmax(p))
def decide(arm, i):
    nat = rows["think_full"][i]; it = items[i]
    if not nat["valid"] or not routed(it): return nat["choice"] if nat["valid"] else None, False
    pn = np.asarray(nat["p_canon"], float)
    if arm == "nat": return nat["choice"], False
    kind, w = arm.split("@"); w = float(w)
    if kind == "x1":
        r = rows["think_b128_plain"].get(i)
        if r is None or not r["valid"] or nat["think_tokens"] <= 128: return nat["choice"], False
        return pick((1 - w) * pn + w * np.asarray(r["p_canon"], float)), True
    r = rows[NT].get(i)
    if r is None or not r["valid"]: return nat["choice"], False
    f = lambda p: np.log(np.maximum(np.asarray(p, float), 1e-6))
    return pick((1 - w) * f(pn) + w * f(r["p_canon"])), True
GRID = [0.25, 0.5, 0.75, 1.0]
ARMS = ["nat"] + [f"x1@{w}" for w in GRID] + [f"x2@{w}" for w in GRID]
acc = {a: defaultdict(dict) for a in ARMS}; changed = {a: defaultdict(int) for a in ARMS}; n_routed = defaultdict(int); n_set = defaultdict(int)
for i, nat in rows["think_full"].items():
    it = items[i]; s = it["meta"]["set"]; n_set[s] += 1; n_routed[s] += routed(it)
    for a in ARMS:
        ch, used = decide(a, i); acc[a][s][i] = float(ch is not None and ch == it["gold"])
        changed[a][s] += int(ch is not None and nat["valid"] and ch != nat["choice"])
gd = find("M1_dev_v2_gsm", "Qwen3-8B") if name == "q3" else find("D_dev_gsm", "r1_base")
gs = {}
for l in open(gd / "samples.jsonl"):
    r = json.loads(l)
    if r["cond"] == "think_full": gs[r["id"]] = float(r["correct"])
for a in ARMS: acc[a]["gsm8k"] = dict(gs)
def fam(a, f): return float(np.mean([100 * np.mean(list(acc[a][s].values())) for s in FAM if FAM[s] == f and s in acc[a]]))
rng = np.random.default_rng(0)
def lb(a, strata):
    d = np.asarray([100 * (acc[a][s][i] - acc["nat"][s][i]) for s in strata for i in acc[a][s]]); bm = d[rng.integers(0, len(d), (4000, len(d)))].mean(1); return float(np.percentile(bm, 5)), float(d.mean())
tab = {}
for a in ARMS:
    sp, li, pr = fam(a, "speaker"), fam(a, "literal"), fam(a, "presup"); g = {k: lb(a, v) for k, v in STRATA.items()}
    tab[a] = {"speaker": sp, "literal": li, "presup": pr, "score": (sp + li) / 2, "gate_lb": g, "gates_pass": all(v[0] > -2 for v in g.values()), "changed_choices": {s: int(v) for s, v in changed[a].items() if v}}
nat_tok = float(np.mean([r["total_tokens"] for r in rows["think_full"].values()])); nt_tok = float(np.mean([r["total_tokens"] for r in rows[NT].values()]))
for a in ARMS: tab[a]["beats_nat"] = bool(tab[a]["score"] > tab["nat"]["score"]); tab[a]["passes_dev_gate"] = bool(a != "nat" and tab[a]["beats_nat"] and tab[a]["gates_pass"])
trn = {}
for a in ARMS[1:]:
    trn[a] = {s: [int(sum(acc["nat"][s][i] == 0 and acc[a][s][i] == 1 for i in acc[a][s])), int(sum(acc["nat"][s][i] == 1 and acc[a][s][i] == 0 for i in acc[a][s])), len(acc[a][s])] for s in n_set}
out = {"model": name, "spec": "notes/DECISIONS.md P2 X1′/X2′ exploratory round 2 (not pre-registered)", "table": tab, "routed_items": {s: [int(n_routed[s]), int(n_set[s])] for s in sorted(n_set)}, "routed_total": int(sum(n_routed.values())),
       "transitions_wrong_to_right_right_to_wrong_n": trn, "tokens_forced_choice": {"nat": nat_tok, "x1_extra_readout_tokens_unused": float(np.mean([r["answer_tokens"] for r in rows["think_b128_plain"].values()])), "nt_readout": nt_tok}}
json.dump(out, open(OUT / f"selection_dev_X2_{name}.json", "w"), indent=1)
print(f"{'arm':7s} speaker literal presup score  gates(ix_can imppres_truth folio gsm8k)  changed")
for a in ARMS: t = tab[a]; print(f"{a:7s} {t['speaker']:6.1f} {t['literal']:6.1f} {t['presup']:6.1f} {t['score']:6.2f}  {'PASS' if t['gates_pass'] else 'fail'} " + " ".join(f"{v[0]:+.1f}" for v in t["gate_lb"].values()) + f"  {sum(t['changed_choices'].values())}")
print("routed per set:", out["routed_items"]); print("DEV GATE PASS:", {a: tab[a]["passes_dev_gate"] for a in ARMS[1:]})
