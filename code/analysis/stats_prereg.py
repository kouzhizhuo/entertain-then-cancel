#!/usr/bin/env python3
"""Pre-registered statistics (lead request, finish plan): from the per-item, per-seed answers (merge tree /tmp/p2_merge built by merge_results.sh; test split only).
 (1) glmm: per Qwen3 model, E1 budget runs: logit P(pragmatic) ~ log2(B+1) x class (gci = a1/a2/a3 sets; ref = pci/psp = b1/b2/a4 sets), B = realised thinking tokens of the row, caps + natural arms (no 'off'),
     fitted as a logistic model with item-cluster-robust SEs (population-averaged, GEE-independence equivalent). DEVIATION from the pre-registered random-slope GLMM ((1+log B | item)): no mixed-model software
     (R/lme4) is available; the cluster-robust logit estimates the same fixed interaction with a sandwich CI. Reported: interaction coefficient, 95% CI, p, n rows, n clusters, per-class slopes.
 (2) tost: equivalence test (TOST, +-3 pp) on the paired item-level difference natural thinking - baseline for lexical scales (a2) and LUDWIG (b2), every open model with those sets (t-based, df = items - 1).
 (3) holm: Holm correction of the paired-bootstrap p-values (two-sided, 20,000 resamples) of delta R on some (quantifiers) and or (connectives), within each model family, separately for the core test sets (a3_fel) and the full sets (a3f_fel).
Outputs: results/STATS/summary.json (stats.<model>.<test>.*), results/STATS/stats_{glmm,tost,holm}.csv."""
import json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats as st
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "analysis"))
from export_e1 import MODEL, base_of
T = Path("/tmp/p2_merge"); OUT = EXP / "results" / "STATS"; OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
def mk(name): return MODEL.get(name, name.lower().replace(".", "p"))
def load(group, model_dir, buckets=None):
    f = Path(model_dir) / "samples.jsonl"; rows = []
    for l in open(f):
        r = json.loads(l)
        if r.get("split") != "test" or not r["valid"] or r["is_gold"] is None: continue
        if buckets and not any(r["bucket"].startswith(b) for b in buckets): continue
        rows.append((r["item_id"], r["bucket"], r.get("subset"), r["condition"], r["seed"], float(r["is_gold"]), r.get("think_tokens", 0)))
    return pd.DataFrame(rows, columns=["item", "bucket", "subset", "cond", "seed", "y", "tok"])
def meta_conds(model_dir): return json.load(open(Path(model_dir) / "results.json"))["meta"]["conditions"]
def nat_and_base(conds):
    nat = "effort_high" if any(c.startswith("effort_") for c in conds) else "think_full"
    base = base_of(conds); return nat, base
def item_diff(df, nat, base):
    a = df[df.cond == nat].groupby("item").y.mean(); b = df[df.cond == base].groupby("item").y.mean(); k = a.index.intersection(b.index)
    return (a[k] - b[k]).values * 100, len(k)
def boot_p(d, n=20000):
    d = np.asarray(d, float); m = d.mean(); idx = rng.integers(0, len(d), (n, len(d))); bm = d[idx].mean(1)
    p = (1 + 2 * min((bm <= 0).sum(), (bm >= 0).sum())) / (n + 1); return m, min(1.0, p), np.percentile(bm, [2.5, 97.5])
CLASS = lambda b: "gci" if b.split("_")[0] in ("a1", "a2", "a3", "a3f") else "pci"
res = defaultdict(dict); rows_glmm, rows_tost, rows_holm = [], [], []
# ------------------------------------------------------------------ (1) GLMM-style interaction, Qwen3 E1 budget runs
BUD = {"think_b0": 0, "think_b0_plain": 0, "think_b64": 64, "think_b128": 128, "think_b256": 256, "think_b512": 512, "think_b1024": 1024, "think_b2048": 2048, "think_full": None}
for grp, mdl in (("E1_main", "Qwen3-4B"), ("E1_main", "Qwen3-8B"), ("E1_main", "Qwen3-14B"), ("E1s_nt", "Qwen3-32B-FP8"), ("E1s", "Qwen3-30B-A3B-FP8"), ("E1s", "Qwen3-1.7B"), ("E1s", "Qwen3-0.6B")):
    d = T / grp / mdl
    if not d.exists(): continue
    df = load(grp, d); df = df[df.cond.isin(list(BUD) + ["think_b512_plain", "think_b2048_plain"])].copy()
    df["cls"] = df.bucket.map(CLASS); df["gci"] = (df.cls == "gci").astype(float); df["lb"] = np.log2(df.tok + 1); df["lbc"] = df.lb - df.lb.mean(); df["ix"] = df.lbc * df.gci
    df["cl"] = df.bucket + "|" + df.item; X = sm.add_constant(df[["lbc", "gci", "ix"]])
    fit = sm.Logit(df.y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(df.cl)[0]})
    ci = fit.conf_int().loc["ix"]; slopes = {}
    for c in ("gci", "pci"):
        s = df[df.cls == c]; Xs = sm.add_constant(s[["lbc"]]); f2 = sm.Logit(s.y, Xs).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(s.cl)[0]})
        slopes[c] = (f2.params["lbc"], f2.conf_int().loc["lbc"].tolist(), f2.pvalues["lbc"])
    k = mk(mdl); node = {"coef_interaction": fit.params["ix"], "lo": ci[0], "hi": ci[1], "p": fit.pvalues["ix"], "n_obs": int(len(df)), "n_clusters": int(df.cl.nunique()),
                         "slope_gci": slopes["gci"][0], "slope_gci_lo": slopes["gci"][1][0], "slope_gci_hi": slopes["gci"][1][1], "slope_pci": slopes["pci"][0], "slope_pci_lo": slopes["pci"][1][0], "slope_pci_hi": slopes["pci"][1][1]}
    res[k]["glmm"] = node; rows_glmm.append({"model": k, **node}); print("GLMM", k, {a: round(b, 4) if isinstance(b, float) else b for a, b in node.items() if a in ("coef_interaction", "lo", "hi", "p", "n_obs")}, flush=True)
# ------------------------------------------------------------------ (2) TOST on a2 and LUDWIG (b2), every open model
SRC = [g for g in ("E1_main", "E1s", "E1s_nt", "E1p_pop", "E1b", "E1q", "E3_gptoss", "E1_extra", "E1_pangu", "E1R1_core") if (T / g).exists()]
for grp in SRC:
    for d in sorted((T / grp).glob("*")):
        if not (d / "samples.jsonl").exists() or not (d / "results.json").exists(): continue
        try: conds = meta_conds(d)
        except Exception: continue
        nat, base = nat_and_base(conds)
        if base == "think_b0_plain" and "think_b512_plain" in conds: pass
        if nat not in conds or base not in conds or nat == base: continue
        df = load(grp, d, ["a2", "b2_ludwig", "b2"])
        k = mk(d.name)
        for tag, pref in (("a2", ("a2",)), ("b2", ("b2_ludwig", "b2"))):
            s = df[df.bucket.isin(pref)] if tag == "b2" else df[df.bucket == "a2"]
            if s.empty: continue
            dd, n = item_diff(s, nat, base)
            if n < 10: continue
            m, sd = dd.mean(), dd.std(ddof=1); se = sd / math.sqrt(n); mar = 3.0
            p_lo = 1 - st.t.cdf((m + mar) / se, n - 1); p_hi = st.t.cdf((m - mar) / se, n - 1); p = max(p_lo, p_hi); ci90 = (m - st.t.ppf(0.95, n - 1) * se, m + st.t.ppf(0.95, n - 1) * se)
            node = {"diff_pp": m, "lo90": ci90[0], "hi90": ci90[1], "p_tost": p, "equivalent_3pp": bool(p < 0.05), "n_items": int(n), "margin_pp": mar, "source": grp}
            if "tost_" + tag in res[k] and res[k]["tost_" + tag]["n_items"] >= n: continue    # keep the source with the most items
            res[k]["tost_" + tag] = node; rows_tost.append({"model": k, "set": tag, **node})
# ------------------------------------------------------------------ (3) Holm on some / or delta R (core and full sets)
FAM = lambda k: ("Qwen3" if k.startswith("qwen3-") else "OLMo-3" if k.startswith("olmo3") else "gpt-oss" if k.startswith("gptoss") or k.startswith("gpt-oss") else "newest" if k.startswith(("qwen3p5", "qwen3p8")) else "further open")
tests = []   # (family, model, trigger, source_kind, p, mean, lo, hi, n)
for kind, groups, bkt in (("core", ("E1_main", "E1s", "E1s_nt", "E1p_pop", "E1b", "E1q", "E3_gptoss", "E1_pangu", "E1R1_core"), ("a3_fel",)), ("full", ("E1_scalar_full", "E1_sfx", "E1_gptoss_sf", "E1R1_sfx"), ("a3f_fel",))):
    seen = set()
    for grp in groups:
        if not (T / grp).exists(): continue
        for d in sorted((T / grp).glob("*")):
            if not (d / "samples.jsonl").exists() or not (d / "results.json").exists(): continue
            k = mk(d.name)
            if (kind, k) in seen: continue
            try: conds = meta_conds(d)
            except Exception: continue
            nat, base = nat_and_base(conds)
            if nat not in conds or base not in conds or nat == base: continue
            df = load(grp, d, list(bkt)); df = df[df.bucket.isin(bkt)]
            ok = False
            for trig, lab in (("quantifiers", "some"), ("connectives", "or")):
                s = df[df.subset == trig]
                if s.empty: continue
                dd, n = item_diff(s, nat, base)
                if n < 10: continue
                m, p, ci = boot_p(dd); tests.append((FAM(k), k, lab, kind, p, m, ci[0], ci[1], n)); ok = True
            if ok: seen.add((kind, k))
for kind in ("core", "full"):
    for fam in sorted({t[0] for t in tests}):
        T_ = [t for t in tests if t[0] == fam and t[3] == kind]; T_.sort(key=lambda t: t[4]); m_ = len(T_)
        run = 0.0
        for i, t in enumerate(T_):
            run = max(run, min(1.0, (m_ - i) * t[4])); k = t[1]
            node = {"delta_pp": t[5], "lo": t[6], "hi": t[7], "p_raw": t[4], "p_holm": run, "survives_holm_05": bool(run < 0.05), "n_items": t[8], "family": fam, "family_size": m_}
            res[k][f"holm_{t[2]}_{kind}"] = node; rows_holm.append({"model": k, "trigger": t[2], "sets": kind, **node})
# ------------------------------------------------------------------ write
def tolist(o):
    if isinstance(o, dict): return {k: tolist(v) for k, v in o.items()}
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    return o
def add_digits(node):
    for k, v in list(node.items()):
        if isinstance(v, dict): add_digits(v)
        elif isinstance(v, float) and not k.endswith("__digits"): node[k + "__digits"] = 4 if k.startswith("p") or "coef" in k or "slope" in k else 1
out = tolist({m: dict(v) for m, v in res.items()}); add_digits(out)
json.dump(out, open(OUT / "summary.json", "w"), indent=1)
for name, rows in (("glmm", rows_glmm), ("tost", rows_tost), ("holm", rows_holm)): pd.DataFrame(rows).to_csv(OUT / f"stats_{name}.csv", index=False)
print("models", len(out), "tests: glmm", len(rows_glmm), "tost", len(rows_tost), "holm", len(rows_holm))
