#!/usr/bin/env python3
"""Secondary format-robustness count (coordinator 2026-10-05): among models with a full-set three-way (nli) run,
how many show a significant drop (95% CI < 0) on some OR or, and how many a significant rise. Writes exp/results/DERIVED/summary.json."""
import json
from pathlib import Path
R = Path(__file__).resolve().parents[1] / "results"
blocks = [json.load(open(R / "E1SF/summary.json")), json.load(open(R / "E1SFX/summary.json")), json.load(open(R / "E1PANGU/summary.json"))]
drop, rise, tested = [], [], []
for b in blocks:
    for m, v in b.items():
        if m.endswith(('-elow', '-emed')): continue  # effort variants of one model are not separate models
        if m == 'openpangu-embedded-1b-v1p1': continue  # fast-thinking only (no thinking arm)
        es = [v.get(f"a3f-{t}", {}).get("nli", {}).get("nat", {}).get("dR") for t in ("quant", "conn")]
        es = [e for e in es if e and "lo" in e]
        if not es: continue
        tested.append(m)
        if any(e["hi"] < 0 for e in es): drop.append(m)
        if any(e["lo"] > 0 for e in es): rise.append(m)
out = {"fullsets": {"nli": {"tested": len(tested), "drop": len(drop), "rise": len(rise)}}}
json.dump({"drop_models": drop, "rise_models": rise}, open(R / "DERIVED/models_lists.json", "w"), indent=1)
(R / "DERIVED").mkdir(exist_ok=True); json.dump(out, open(R / "DERIVED/summary.json", "w"), indent=1)
print(json.dumps(out["fullsets"]["nli"]), drop, rise)

# --- MAIN COUNT (pre-declared polarity rule, DECISIONS 2026-10-05): over the full OPEN roster (closed models are reported separately and not counted) ---
def blk(n): return json.load(open(R / f"{n}/summary.json"))
PRIMARY = [("E1", "nat"), ("E1R1", "nat"), ("E1S", "nat"), ("E1P", "nat"), ("E1B", "nat"), ("E1Q", "nat"), ("E1PANGU", "nat"), ("E3", "high")]      # primary Yes/No inference format: thinking on - off (forced-empty / low effort where no switch)
NLI = [("E1SF", "nat"), ("E1SFX", "nat"), ("E1PANGU", "nat"), ("E1R1", "nat"), ("E1GSF", "high")]                                          # polarity-free full-set formats (3-way NLI)
EXCL = ("pragrest", "-elow", "-emed", "openpangu-embedded-1b", "mistral-small-24b-instruct", "qwen3-4b-instruct-2507")      # trained variants, effort variants, no-thinking controls, fast-thinking-only
EXCL_EXACT = ("phi4",)
def keep(m): return not m.startswith(EXCL_EXACT) or m != "phi4" and not any(x in m for x in EXCL)
def keep(m): return m not in EXCL_EXACT and not any(x in m for x in EXCL)
def eff(v, f, arm, t, fmt):
    e = v.get(f"{f}-{t}", {}).get(fmt, {}).get(arm, {}).get("dR"); return e if e and "lo" in e else None
prim, nli, probe = {}, {}, {}
for b, arm in PRIMARY:
    for m, v in blk(b).items():
        if not keep(m) or m in prim: continue
        es = {t: eff(v, "a3", arm, t, "inf") for t in ("quant", "conn")}; es = {t: e for t, e in es.items() if e}
        if es: prim[m] = es
for b, arm in NLI:
    for m, v in blk(b).items():
        if not keep(m) or m in nli: continue
        es = {t: eff(v, "a3f", arm, t, "nli") for t in ("quant", "conn")}; es = {t: e for t, e in es.items() if e}
        if es: nli[m] = es
for m, v in blk("E1PROBE").items():
    es = {t: v.get(f"a3-{t}", {}).get("infflip", {}).get("nat", {}).get("dR") for t in ("quant", "conn")}; es = {t: e for t, e in es.items() if e and "lo" in e}
    if es: probe[m] = es
def se(e): return (e["hi"] - e["lo"]) / (2 * 1.96)
def balanced_lo(o, f):    # mean of the original and the polarity-flipped (pragmatic-aligned) shifts; CI by combining the two CIs as independent (approximation, disclosed)
    mean = (o["mean"] + f["mean"]) / 2; s = (se(o) ** 2 + se(f) ** 2) ** 0.5 / 2; return mean, mean - 1.96 * s
tested, counted, formatdep, rise, flat, detail = [], [], [], [], [], {}
for m, es in sorted(prim.items()):
    tested.append(m); drop = any(e["hi"] < 0 for e in es.values()); up = any(e["lo"] > 0 for e in es.values())
    nl = nli.get(m); nli_ok = None if nl is None else any(e["hi"] < 0 for e in nl.values())
    pol_ok = None
    if m in probe:
        pol_ok = True
        for t, f in probe[m].items():
            if t in es:
                mean, lo = balanced_lo(es[t], f)
                if lo > 0: pol_ok = False
    detail[m] = {"primary_drop": drop, "primary_rise": up, "nli_drop": nli_ok, "polarity_ok": pol_ok}
    if up and not drop: rise.append(m)
    if not drop: flat.append(m) if not up else None; continue
    if nli_ok is False or pol_ok is False: formatdep.append(m)
    else: counted.append(m)
out["main"] = {"tested": len(tested), "counted": len(counted), "formatdep": len(formatdep), "rise": len(rise), "no_sig_change": len(flat)}
ml = json.load(open(R / "DERIVED/models_lists.json")); ml.update({"main_tested": tested, "main_counted": counted, "main_formatdep": formatdep, "main_rise": rise, "main_no_sig_change": flat, "main_detail": detail})
json.dump(ml, open(R / "DERIVED/models_lists.json", "w"), indent=1); json.dump(out, open(R / "DERIVED/summary.json", "w"), indent=1)
print("MAIN", out["main"]); print(" counted:", counted); print(" formatdep:", formatdep); print(" rise:", rise); print(" no sig change:", flat)

# --- pre-registered statistics, display-ready (STATS block -> DERIVED) ---
S = json.load(open(R / "STATS/summary.json"))
def pstr(p): return "<.001" if p < 0.001 else f"{p:.3f}".lstrip("0")
glmm = {}
for m in ("qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"):
    g = S[m]["glmm"]
    glmm[m] = {"coef": round(g["coef_interaction"], 3), "lo": round(g["lo"], 3), "hi": round(g["hi"], 3), "p": pstr(g["p"]),
               "coef__digits": 3, "lo__digits": 3, "hi__digits": 3}
fam = {}
for m, v in S.items():
    if m.endswith(("-elow", "-emed")): continue
    for t in ("some", "or"):
        h = v.get(f"holm_{t}_full")
        if not h: continue
        f = fam.setdefault(h["family"], {"tests": 0, "survive": 0}); f["tests"] += 1; f["survive"] += bool(h["survives_holm_05"])
tot = {"tests": sum(f["tests"] for f in fam.values()), "survive": sum(f["survive"] for f in fam.values())}
tost = {m: v for m, v in S.items() if not m.endswith(("-elow", "-emed"))}
eq_a2 = sum(bool(v.get("tost_a2", {}).get("equivalent_3pp")) for v in tost.values()); n_a2 = sum("tost_a2" in v for v in tost.values())
out.update({"glmm": glmm, "holm": {"full": tot}, "tost": {"a2": {"equivalent": eq_a2, "tested": n_a2}}})
json.dump(out, open(R / "DERIVED/summary.json", "w"), indent=1)
print("holm full", tot, {k: v for k, v in fam.items()}, "tost a2", eq_a2, n_a2)
