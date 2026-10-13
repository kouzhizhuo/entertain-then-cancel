#!/usr/bin/env python3
"""METHOD TEST report (METHOD_DESIGN v1). Inputs: results/<src>/M1_test/<model>/samples.jsonl (forced-choice arms), results/<src>/M1_test_gsm/<model>/{results,samples} (GSM8K arms), frozen SIT arm name via --sit seed_X.
Per model and method: accuracy per set, macro over the speaker-meaning sets / literal-gold sets / presupposition sets, context-reversal PAIR accuracy (both members right), total generated tokens (mean, p95; the seed is counted).
Statistics: paired cluster bootstrap (4,000 draws; clusters = scenario / item group), SIT - NAT and SIT - strongest baseline (strongest = highest dev score, passed with --base), per-stratum one-sided 95% lower bound for the safety strata
(ix_can, imppres truth quant+conn, folio, gsm8k) vs -2 pp, pooled primary difference over models (per-model differences averaged inside each draw = model as a fixed effect), cost ratio SIT/NAT.
Writes results/M1/test_report.json and results/M1/summary.json (keys m1.<model>.<stat>.<method>.*)."""
import json, sys, argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--sit", required=True); ap.add_argument("--base", default="think_full+taskaware"); ap.add_argument("--models", nargs="*"); a = ap.parse_args()
X = Path(__file__).resolve().parents[1] / "results"; OUT = X / "M1"; OUT.mkdir(exist_ok=True); rng = np.random.default_rng(0)
FAM = {"imppres_quant_fel": "speaker", "imppres_conn_fel": "speaker", "ludwig": "speaker", "circa": "speaker", "pub_task2": "speaker", "pub_task3": "speaker", "pub_task6": "speaker", "figqa": "speaker", "ix_rec": "speaker",
       "nope": "presup", "imppres_presup": "presup", "ix_can": "literal", "imppres_quant_truth": "literal", "imppres_conn_truth": "literal", "folio": "literal", "gsm8k": "literal"}
STRATA = {"ix_can": ["ix_can"], "imppres_truth": ["imppres_quant_truth", "imppres_conn_truth"], "folio": ["folio"], "gsm8k": ["gsm8k"]}
NAT = "think_full"; NT_ALIASES = ("off", "think_b0_plain")
def find(g, m):
    for src in ("runs", "<host>", "slurm4"):
        p = X / src / g / m
        if (p / "samples.jsonl").exists(): return p
def cluster(r):
    m = r.get("meta") or {}
    return str(m.get("pair", m.get("ix_id", m.get("qid", m.get("scenario", m.get("premise", r["item_id"]))))))
def load_model(model):
    d = find("M1_test", model); items = defaultdict(lambda: defaultdict(dict)); toks = defaultdict(list); clus = {}
    if d is None: return None
    for l in open(d / "samples.jsonl"):
        r = json.loads(l); st = r["bucket"][2:]; c = r["condition"]; y = float(r["is_gold"]) if r["valid"] else 0.0
        items[c][st][r["item_id"]] = y; clus[(st, r["item_id"])] = r["item_id"]; toks[c].append(r.get("total_tokens", r.get("think_tokens", 0)))
    g = find("M1_test_gsm", model)
    if g:
        for l in open(g / "samples.jsonl"):
            r = json.loads(l); items[r["cond"]]["gsm8k"][r["id"]] = float(r["correct"]); clus[("gsm8k", r["id"])] = r["id"]
        res = json.load(open(g / "results.json"))["results"]
    else: res = {}
    return items, toks, clus, res
def macro(it, c, fam): return float(np.mean([100 * np.mean(list(it[c][s].values())) for s in FAM if FAM[s] == fam and s in it[c] and it[c][s]]))
def pair_acc(it, c):
    d = it[c].get("ctxrev", {}); pairs = defaultdict(dict)
    for iid, y in d.items():
        k = iid.split(".")[2]; pairs[k][iid.split(".")[3]] = y
    return 100 * float(np.mean([min(v.values()) for v in pairs.values() if len(v) == 2])) if pairs else None
def diffs(it, c1, c0, sets):
    out = []
    for s in sets:
        a, b = it[c1].get(s, {}), it[c0].get(s, {}); out += [(s, i, 100 * (a[i] - b[i])) for i in sorted(set(a) & set(b))]
    return out
def boot_ci(dd, n=4000):
    arr = np.asarray([x[2] for x in dd]); cl = np.asarray([hash(x[0] + "|" + x[1].rsplit(".", 1)[0]) for x in dd]); u, inv = np.unique(cl, return_inverse=True)
    sums = np.bincount(inv, weights=arr); cnt = np.bincount(inv); idx = rng.integers(0, len(u), (n, len(u))); bm = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(arr.mean()), float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5)), float(np.percentile(bm, 5))
models = a.models or [p.name for src in ("runs", "<host>") for p in sorted((X / src / "M1_test").glob("*")) if (p / "samples.jsonl").exists()]
models = sorted(set(models)); rep = {}; pooled = defaultdict(list)
for m in models:
    L = load_model(m)
    if not L: continue
    it, toks, clus, gres = L; nt = next((c for c in NT_ALIASES if c in it), None)
    meths = {"nt": nt, "nat": NAT, "nat512": "think_b512_plain", "userexact": "think_full+sitroute", "taskaware": "think_full+taskaware", "grice": "seed_grice", "sit": a.sit}
    meths = {k: v for k, v in meths.items() if v and v in it}
    if NAT not in it: continue
    node = {"methods": meths, "acc": {}, "paired": {}}
    for k, c in meths.items():
        tk = np.asarray(toks[c]) if c in toks and toks[c] else np.asarray([0]); gt = gres.get(c, {})
        node["acc"][k] = {"speaker": macro(it, c, "speaker"), "literal": macro(it, c, "literal"), "presup": macro(it, c, "presup"), "reversal_pair": pair_acc(it, c),
                          "tokens_mean": float(tk.mean()), "tokens_p95": float(np.percentile(tk, 95)), "gsm_tokens_mean": gt.get("tokens_mean"), "sets": {s: 100 * float(np.mean(list(it[c][s].values()))) for s in it[c] if it[c][s]}}
    if "sit" in meths:
        for ref in ("nat", "nt", "nat512", "userexact", "taskaware", "grice"):
            if ref not in meths: continue
            sp = [s for s in FAM if FAM[s] == "speaker" and s in it[meths["sit"]]]; dd = diffs(it, meths["sit"], meths[ref], sp)
            if dd:
                mean, lo, hi, _ = boot_ci(dd); node["paired"][f"speaker_vs_{ref}"] = {"diff_pp_pooled_items": mean, "lo": lo, "hi": hi}
        sp = [s for s in FAM if FAM[s] == "speaker" and s in it[meths["sit"]]]
        # equal-set macro difference with a paired bootstrap per set
        def macro_diff(c1, c0):
            vals = []
            for s in sp:
                a_, b_ = it[c1][s], it[c0][s]; ks = sorted(set(a_) & set(b_)); vals.append(np.array([100 * (a_[i] - b_[i]) for i in ks]))
            return vals
        for ref in ("nat", "userexact", "taskaware", "grice"):
            if ref not in meths: continue
            vals = macro_diff(meths["sit"], meths[ref]); md = float(np.mean([v.mean() for v in vals])); bs = []
            for _ in range(2000): bs.append(np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in vals]))
            node["paired"][f"speaker_macro_vs_{ref}"] = {"diff_pp": md, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5))}
            if ref == "nat": pooled["speaker_vs_nat"].append(bs)
        for sname, sets in STRATA.items():
            sets = [s for s in sets if s in it[meths["sit"]]]
            if not sets: continue
            dd = diffs(it, meths["sit"], NAT, sets); mean, lo, hi, lb = boot_ci(dd); node["paired"][f"safety_{sname}_vs_nat"] = {"diff_pp": mean, "lo95_two_sided": lo, "hi": hi, "lower_bound_one_sided_95": lb, "non_inferior_m2pp": bool(lb > -2.0), "n": len(dd)}
        node["cost_ratio_sit_over_nat"] = {"mean": node["acc"]["sit"]["tokens_mean"] / max(1e-9, node["acc"]["nat"]["tokens_mean"]), "p95": node["acc"]["sit"]["tokens_p95"] / max(1e-9, node["acc"]["nat"]["tokens_p95"])}
    rep[m] = node
# pooled primary (model as fixed effect): mean over models of the per-model equal-set macro difference, bootstrap draws aligned by index
if pooled["speaker_vs_nat"]:
    k = min(len(b) for b in pooled["speaker_vs_nat"]); P = np.mean([np.asarray(b[:k]) for b in pooled["speaker_vs_nat"]], axis=0)
    rep["_pooled"] = {"speaker_macro_sit_minus_nat": {"diff_pp": float(np.mean([rep[m]["paired"]["speaker_macro_vs_nat"]["diff_pp"] for m in rep if m != "_pooled" and "speaker_macro_vs_nat" in rep[m]["paired"]])), "lo": float(np.percentile(P, 2.5)), "hi": float(np.percentile(P, 97.5)), "n_models": len(pooled["speaker_vs_nat"])}}
json.dump(rep, open(OUT / "test_report.json", "w"), indent=1)
print(json.dumps({m: {"speaker": {k: round(v["speaker"], 1) for k, v in n["acc"].items()}, "literal": {k: round(v["literal"], 1) for k, v in n["acc"].items()}, "reversal": {k: (round(v["reversal_pair"], 1) if v["reversal_pair"] is not None else None) for k, v in n["acc"].items()}} for m, n in rep.items() if m != "_pooled"}, indent=1))
if "_pooled" in rep: print("POOLED speaker macro SIT-NAT:", rep["_pooled"])
