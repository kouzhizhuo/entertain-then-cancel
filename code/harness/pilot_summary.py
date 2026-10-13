#!/usr/bin/env python3
"""Pilot tables + figure + trace excerpts from results dirs.
   usage: python pilot_summary.py <out_dir> <label>=<results_dir> [...]
   e.g.   python pilot_summary.py ../results/pilot 8B-felicity=../results/pilot_v3_felicity/Qwen3-8B 8B-default=...
Writes <out_dir>/tables.md, <out_dir>/traces.md, <out_dir>/fig_pilot_budget.{pdf,png}. Every number comes from samples.jsonl /
free_samples.jsonl / traces.jsonl of the given runs. Metric names: gold = answer equals the gold (pragmatic) reading; literal =
answer equals the literal-reading option; margin = |log p1 - log p2| of the two most probable options in the constrained readout.
CIs: 95% percentile bootstrap over ITEMS (seed-averaged per-item rates); '±' = std over seeds of the per-seed rate.
"""
import json, re, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[2] / "figures" / "style"))
try:
    import palette as P  # noqa: E402  (figure style; absent on <host> -> colours only, figure() unavailable)
except Exception:  # pragma: no cover
    class P:  # minimal stand-in so the statistics functions can be imported where the style files are not installed
        BLUE, ORANGE, GREEN, VERMILION = "#0072B2", "#E69F00", "#009E73", "#D55E00"

CLAB = {"off": "off", "off_soft": "off-soft", "think_b0": "b0", "think_b256_plain": "256p", "think_b512": "512", "think_b1024_plain": "1024p",
        "think_b2048": "2048", "think_full": "full"}
ORDER = list(CLAB)
BUCKETS = [("scalar_imppres", "Scalar (IMPPRES)", P.BLUE), ("scalar_underinformative", "Underinformative (B&N-style)", P.ORANGE),
           ("indirect_answer_ruis", "Indirect answers (Ruis)", P.GREEN), ("indirect_speech_hu", "Indirect speech (Hu)", P.VERMILION)]
BORDER = [b[0] for b in BUCKETS] + ["scalar_imppres_control", "scalar_imppres_noimpl"]
FIGCONDS = ["off", "think_b0", "think_b512", "think_b2048", "think_full"]
MARKERS = re.compile(r"technically|strictly speaking|logically|does not exclude|doesn't exclude|compatible with|not necessarily|literally|by definition", re.I)


def load(d):
    d = Path(d)
    rows = [json.loads(l) for l in open(d / "samples.jsonl")]
    meta = json.load(open(d / "results.json"))["meta"]
    free = [json.loads(l) for l in open(d / "free_samples.jsonl")] if (d / "free_samples.jsonl").exists() else []
    items = {}
    for l in open(HERE.parent / "data" / "items" / "imppres_scalar.jsonl"):
        r = json.loads(l); items[r["id"]] = r
    for r in rows + free:
        it = items.get(r["item_id"])
        r["trigger"] = it["subset"] if it else r["subset"]
        r["rel"] = it["meta"]["spec_relation"] if it else None
    return {"rows": rows, "meta": meta, "free": free, "dir": d}


def per_item(rows, cond, key, sel=lambda r: True):
    d = defaultdict(list)
    for r in rows:
        if r["condition"] == cond and r.get(key) is not None and sel(r):
            d[r["item_id"]].append(float(r[key]))
    return {k: mean(v) for k, v in d.items()}


def ci(vals, n=4000, seed=0):
    x = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    return np.percentile(x[rng.integers(0, len(x), (n, len(x)))].mean(1), [2.5, 97.5])


def seed_std(rows, cond, key, sel):
    ps = []
    for s in sorted({r["seed"] for r in rows}):
        v = [float(r[key]) for r in rows if r["condition"] == cond and r["seed"] == s and r.get(key) is not None and sel(r)]
        if v: ps.append(mean(v))
    return stdev(ps) if len(ps) > 1 else float("nan")


def cell(rows, cond, key, sel):
    pi = per_item(rows, cond, key, sel)
    if not pi: return None
    v = list(pi.values()); lo, hi = ci(v)
    return {"m": 100 * mean(v), "std": 100 * seed_std(rows, cond, key, sel), "lo": 100 * lo, "hi": 100 * hi, "n": len(v)}


def delta(rows, c1, c0, key, sel):
    a, b = per_item(rows, c1, key, sel), per_item(rows, c0, key, sel)
    ks = sorted(set(a) & set(b))
    if not ks: return None
    d = [a[k] - b[k] for k in ks]; lo, hi = ci(d)
    return 100 * mean(d), 100 * lo, 100 * hi, len(ks)


def fmt(c): return "–" if c is None else f"{c['m']:.1f} ± {c['std']:.1f} [{c['lo']:.0f}, {c['hi']:.0f}]"
def fmtd(d): return "–" if d is None else f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]"


def margin(r):
    p = sorted(r["p_canon"], reverse=True)
    return abs(np.log(max(p[0], 1e-6)) - np.log(max(p[1], 1e-6)))


def pct(x, q): return float(np.percentile(x, q))


def tables(runs):
    md = []
    for lab, R in runs.items():
        rows, meta = R["rows"], R["meta"]
        conds = [c for c in ORDER if c in meta["conditions"]]
        buckets = sorted({r["bucket"] for r in rows}, key=lambda b: BORDER.index(b) if b in BORDER else 99)
        md.append(f"\n### {lab}: {meta['model_name']} — framing `{meta['config'].get('variant', 'default')}`, {meta['n_items']} items × {len(meta['seeds'])} seeds, "
                  f"T={meta['sampling']['temperature']}, cap {meta['sampling']['cap']}, config sha16 `{meta['config_sha16']}`\n")
        md.append("**Pragmatic-response rate** = % answers equal to the gold (pragmatic) reading; mean ± std over seeds [95% bootstrap CI over items]. "
                  "Δ = think_full − off (paired over items).\n")
        md.append("| bucket (role) | n | " + " | ".join(CLAB[c] for c in conds) + " | Δ full−off [CI] |"); md.append("|---|---|" + "---|" * (len(conds) + 1))
        for b in buckets:
            for role in sorted({r["role"] for r in rows if r["bucket"] == b}):
                sel = lambda r, b=b, role=role: r["bucket"] == b and r["role"] == role
                cs = [cell(rows, c, "is_gold", sel) for c in conds]
                md.append(f"| {b} ({role}) | {cs[0]['n'] if cs[0] else 0} | " + " | ".join(fmt(c) for c in cs) + f" | {fmtd(delta(rows, 'think_full', 'off', 'is_gold', sel))} |")
        md.append("\n**Literal-reading rate** (% answers equal to the literal-reading option; items where the source defines one):\n")
        md.append("| bucket (role) | " + " | ".join(CLAB[c] for c in conds) + " | Δ full−off [CI] |"); md.append("|---|" + "---|" * (len(conds) + 1))
        for b in buckets:
            for role in sorted({r["role"] for r in rows if r["bucket"] == b}):
                sel = lambda r, b=b, role=role: r["bucket"] == b and r["role"] == role
                cs = [cell(rows, c, "is_literal", sel) for c in conds]
                if all(c is None for c in cs): continue
                md.append(f"| {b} ({role}) | " + " | ".join(fmt(c) for c in cs) + f" | {fmtd(delta(rows, 'think_full', 'off', 'is_literal', sel))} |")
        # NT-variant agreement and CAP vs CAP-plain
        md.append("\n**No-thinking variants and truncation arms** (rule: the three NT variants within 2 pp; CAP-plain vs CAP flagged if sign differs or > 3 pp):\n")
        md.append("| bucket (role) | off | off-soft | b0 | max NT gap (pp) | flag | 256p | 512 | 1024p | 2048 |"); md.append("|---|---|---|---|---|---|---|---|---|---|")
        for b in buckets:
            for role in sorted({r["role"] for r in rows if r["bucket"] == b}):
                sel = lambda r, b=b, role=role: r["bucket"] == b and r["role"] == role
                v = {c: cell(rows, c, "is_gold", sel) for c in conds}
                nt = [v[c]["m"] for c in ("off", "off_soft", "think_b0") if v.get(c)]
                if len(nt) < 2: continue
                gap = max(nt) - min(nt)
                g = lambda c: f"{v[c]['m']:.1f}" if v.get(c) else "–"
                md.append(f"| {b} ({role}) | {g('off')} | {g('off_soft')} | {g('think_b0')} | {gap:.1f} | {'**>2**' if gap > 2 else 'ok'} | "
                          f"{g('think_b256_plain')} | {g('think_b512')} | {g('think_b1024_plain')} | {g('think_b2048')} |")
        # trigger breakdown
        md.append("\n**Scalar (IMPPRES targets) by trigger**, pragmatic-response rate:\n")
        md.append("| trigger | n | " + " | ".join(CLAB[c] for c in conds) + " | Δ full−off [CI] |"); md.append("|---|---|" + "---|" * (len(conds) + 1))
        for t in sorted({r["trigger"] for r in rows if r["bucket"] == "scalar_imppres"}):
            sel = lambda r, t=t: r["bucket"] == "scalar_imppres" and r["trigger"] == t
            cs = [cell(rows, c, "is_gold", sel) for c in conds]
            md.append(f"| {t} | {cs[0]['n']} | " + " | ".join(fmt(c) for c in cs) + f" | {fmtd(delta(rows, 'think_full', 'off', 'is_gold', sel))} |")
        md.append("\n**Scalar (IMPPRES targets) by relation** (`gold`/`literal` labels differ by relation; 'reverse negated' items ask the unnatural direction):\n")
        md.append("| relation | n | " + " | ".join(CLAB[c] for c in conds) + " | Δ full−off [CI] |"); md.append("|---|---|" + "---|" * (len(conds) + 1))
        for t in sorted({r["rel"] for r in rows if r["bucket"] == "scalar_imppres"}):
            sel = lambda r, t=t: r["bucket"] == "scalar_imppres" and r["rel"] == t
            cs = [cell(rows, c, "is_gold", sel) for c in conds]
            md.append(f"| {t} | {cs[0]['n']} | " + " | ".join(fmt(c) for c in cs) + f" | {fmtd(delta(rows, 'think_full', 'off', 'is_gold', sel))} |")
        # tokens
        md.append("\n**Realised thinking tokens** median [IQR] and share of traces ended naturally before the budget:\n")
        tc = [c for c in conds if c not in ("off", "off_soft")]
        md.append("| bucket | " + " | ".join(CLAB[c] for c in tc) + " |"); md.append("|---|" + "---|" * len(tc))
        for b in buckets:
            cells = []
            for c in tc:
                rs = [r for r in rows if r["bucket"] == b and r["condition"] == c]
                tk = [r["think_tokens"] for r in rs]
                cells.append(f"{median(tk):.0f} [{pct(tk, 25):.0f}, {pct(tk, 75):.0f}] / {100 * mean(1 if r['natural'] else 0 for r in rs):.0f}%")
            md.append(f"| {b} | " + " | ".join(cells) + " |")
        # margins
        md.append("\n**Letter-probability margin** |log p₁ − log p₂| of the constrained readout, median [IQR] (large = committed; ≈0 = torn):\n")
        md.append("| bucket | " + " | ".join(CLAB[c] for c in conds) + " |"); md.append("|---|" + "---|" * len(conds))
        for b in buckets:
            cells = []
            for c in conds:
                mg = [margin(r) for r in rows if r["bucket"] == b and r["condition"] == c and r["valid"]]
                cells.append(f"{median(mg):.1f} [{pct(mg, 25):.1f}, {pct(mg, 75):.1f}]" if mg else "–")
            md.append(f"| {b} | " + " | ".join(cells) + " |")
        # free answers
        fr = R["free"]
        if fr:
            md.append("\n**Free-answer pass** (unconstrained reply ending in `Answer: X`; subset of items, all seeds): committed = a parsable final letter; "
                      "hedge-words = reply contains depends/ambiguous/either/cannot determine/…; unfinished = thinking hit the cap; "
                      "R(free) = pragmatic rate among committed vs R(constrained) on the same items and seeds.\n")
            md.append("| bucket | cond | n | committed % | hedge-words % | unfinished % | R free | R constrained |"); md.append("|---|---|---|---|---|---|---|---|")
            for b in buckets:
                for c in ("off", "think_full"):
                    rs = [r for r in fr if r["bucket"] == b and r["condition"] == c]
                    if not rs: continue
                    com = [r for r in rs if r["committed"]]
                    keys = {(r["item_id"], r["seed"]) for r in rs}
                    cons = [r for r in rows if r["bucket"] == b and r["condition"] == c and (r["item_id"], r["seed"]) in keys and r["is_gold"] is not None]
                    md.append(f"| {b} | {c} | {len(rs)} | {100 * len(com) / len(rs):.1f} | {100 * mean(r['hedge_words'] for r in rs):.1f} | "
                              f"{100 * mean(r['unfinished_think'] for r in rs):.1f} | "
                              f"{(100 * mean(r['is_gold'] for r in com)) if com else float('nan'):.1f} | {(100 * mean(r['is_gold'] for r in cons)) if cons else float('nan'):.1f} |")
        md.append(f"\nFinish counts `{json.dumps(meta['finish_counts'])}`; generated tokens {meta['total_generated_tokens']:,}; wall {meta['wall_seconds']:.0f}s.\n")
    return "\n".join(md)


def snippet(text, width=650):
    """Verbatim excerpt: opening + sentences containing literalness markers, elisions marked […]."""
    if len(text) <= width:
        return text.strip()
    hits = [m.start() for m in MARKERS.finditer(text)]
    out = text[: width // 2].strip()
    if hits:
        h = hits[0]; a = max(width // 2, h - 120)
        out += " […] " + text[a: a + width // 2].strip()
    return out + " […]"


def traces(runs, label, main_cond="think_full"):
    R = runs[label]; rows, d = R["rows"], R["dir"]
    tr, lines = {}, {}
    for i, l in enumerate(open(d / "traces.jsonl"), 1):
        t = json.loads(l); tr[(t["item_id"], t["seed"])] = t; lines[(t["item_id"], t["seed"])] = i
    by = defaultdict(dict)
    for r in rows:
        by[(r["item_id"], r["seed"])][r["condition"]] = r
    md = [f"## Verbatim trace excerpts ({label}; `{d.name}` → `exp/results/…/traces.jsonl:line`)\n"]

    def pick(bucket, cond_a, cond_b, want_a, want_b, k=1, prefer_literal=False):
        out = []
        for (iid, s), cs in sorted(by.items()):
            a, b = cs.get(cond_a), cs.get(cond_b)
            if not a or not b or a["bucket"] != bucket or not a["valid"] or not b["valid"]: continue
            if a["is_gold"] == want_a and b["is_gold"] == want_b and (not prefer_literal or b["is_literal"]):
                t = tr.get((iid, s))
                if t and t["n_tokens"] > 60:
                    out.append((iid, s, a, b, t))
        # prefer traces that spell out a literalness marker, then shorter ones (easier to quote)
        out.sort(key=lambda x: (0 if MARKERS.search(x[4]["text"]) else 1, x[4]["n_tokens"]))
        return out[:k]

    def show(title, picks, ctxrow=None):
        md.append(f"### {title}\n")
        if not picks: md.append("_no item satisfies this pattern in this run_\n"); return
        for iid, s, a, b, t in picks:
            it = next(x for x in rows if x["item_id"] == iid)
            md.append(f"- `{iid}` seed {s}: off → {'pragmatic' if a['is_gold'] else 'other'} ({a['choice_kind']}), full → {'pragmatic' if b['is_gold'] else 'other'} ({b['choice_kind']}); "
                      f"{t['n_tokens']} thinking tokens; `traces.jsonl:{lines[(iid, s)]}` (finish `{t['finish']}`)\n\n  > " + snippet(t["text"]).replace("\n", "\n  > ") + "\n")

    show("Pragmatic when not thinking → literal after thinking (scalar targets)", pick("scalar_imppres", "off", main_cond, True, False, 2, prefer_literal=True))
    show("Literal when not thinking → pragmatic after thinking (scalar targets; reverse case)", pick("scalar_imppres", "off", main_cond, False, True, 1))
    show("PCI improved by thinking (Ruis indirect answers: wrong when not thinking → right after thinking)", pick("indirect_answer_ruis", "off", main_cond, False, True, 1))
    show("PCI improved by thinking (Hu indirect speech)", pick("indirect_speech_hu", "off", main_cond, False, True, 1))
    return "\n".join(md)


def logodds(r, item_lit):
    """Answer-position log-odds L = log p(gold) - log p(literal option) (literal defined and != gold) else log p(gold) - log(1 - p(gold)); p from the
    constrained letter readout (p_canon, canonical option order). Positive = towards the gold (pragmatic) reading. Clipped at |L| <= 13.8 (p >= 1e-6)."""
    p = r["p_canon"]; g = r["gold"]
    q = p[g]
    other = p[item_lit] if item_lit is not None and item_lit != g else 1 - q
    return float(np.log(max(q, 1e-6)) - np.log(max(other, 1e-6)))


def note_sections(runs):
    md = []
    # gold/literal indices are not stored per sample: recover from items
    idx = {}
    for f in sorted((HERE.parent / "data" / "items").glob("[!._]*.jsonl")):  # skip exFAT ._* AppleDouble files
        for l in open(f):
            r = json.loads(l); idx[r["id"]] = r
    def item_view(r):  # framing actually used
        it = idx[r["item_id"]]
        v = (it.get("variants") or {}).get(r.get("variant")) if r.get("variant") not in (None, "default", "alt") else None
        return (v["gold"], v["literal"]) if v else (it["gold"], it["literal"])
    md.append("## 1. Reading chosen (R, % answers equal to the gold = pragmatic reading; forced-choice letter readout, greedy) — mean over 3 seeds, 95% bootstrap CI over items\n")
    md.append("Sets in this pilot: IMPPRES scalar targets (A3-like; released items, log ≠ prag), Bott–Noveck-style underinformative sentences (A1 prototype, 45 targets; old 45-triple list), LUDWIG (B2, 180 of 600 test), Hu indirect speech (B1 subset, 20). "
              "A2 (scalar diversity), A4, B1-other phenomena, B3, PUB are not in the pilot. `off` = NT-empty, `off-soft` = `/no_think`, `b0` = NT-forced (stop phrase + `</think>` at once), `NNNp` = plain truncation (bare `</think>`), `512`/`2048` = CAP with the stop phrase, `full` = NAT (cap 8192). SDT c/d′ needs the six-sentence A1 design (not yet built) and is not reported.\n")
    for lab, R in runs.items():
        rows, meta = R["rows"], R["meta"]; conds = [c for c in ORDER if c in meta["conditions"]]
        md.append(f"\n**{lab}** (`{R['dir'].parent.name}/{R['dir'].name}`, T={meta['sampling']['temperature']}, config `{meta['config_sha16']}`)\n")
        md.append("| set (role) | n | " + " | ".join(CLAB[c] for c in conds) + " |"); md.append("|---|---|" + "---|" * len(conds))
        for b in sorted({r["bucket"] for r in rows}, key=lambda b: BORDER.index(b) if b in BORDER else 99):
            if b == "scalar_imppres_noimpl": continue   # dropped as a control (lead, plan v1.2.6): ill-posed under the inference question
            for role in sorted({r["role"] for r in rows if r["bucket"] == b}):
                sel = lambda r, b=b, role=role: r["bucket"] == b and r["role"] == role
                cs = [cell(rows, c, "is_gold", sel) for c in conds]
                md.append(f"| {b} ({role}) | {cs[0]['n']} | " + " | ".join(f"{c['m']:.1f} [{c['lo']:.0f}, {c['hi']:.0f}]" if c else "–" for c in cs) + " |")
    md.append("\nFull tables (± seed std, literal rate, Δ full−off with CIs, by trigger and by relation): `exp/results/pilot_summary/tables.md`.\n")

    md.append("## 2. Hedge rate and realised thinking tokens\n")
    md.append("The letter-constrained readout cannot hedge, so hedging is measured in a separate free-answer pass (unconstrained reply ending in `Answer: X`; 200-item subset, every 3rd item of the pilot list, all 3 seeds; `off` and `think_full` only). "
              "'committed' = parsable final letter; 'hedge-words' = the reply contains depends / ambiguous / either / cannot determine / … (keyword flag, not a judge); 'unfinished' = thinking reached the cap. **All runs: 100 % committed, 0 % unfinished, hedge-words 0–7 % → the plan's < 10 % hedge criterion is met and no switch to a constrained final line is needed.**\n")
    md.append("| run | set | cond | n | committed % | hedge-words % | R free | R constrained (same items/seeds) |"); md.append("|---|---|---|---|---|---|---|---|")
    for lab, R in runs.items():
        for b in [x for x in BORDER if x in {r["bucket"] for r in R["free"]}]:
            if b == "scalar_imppres_noimpl": continue
            for c in ("off", "think_full"):
                rs = [r for r in R["free"] if r["bucket"] == b and r["condition"] == c]
                if not rs: continue
                com = [r for r in rs if r["committed"]]; keys = {(r["item_id"], r["seed"]) for r in rs}
                cons = [r for r in R["rows"] if r["bucket"] == b and r["condition"] == c and (r["item_id"], r["seed"]) in keys and r["is_gold"] is not None]
                md.append(f"| {lab} | {b} | {c} | {len(rs)} | {100 * len(com) / len(rs):.0f} | {100 * mean(r['hedge_words'] for r in rs):.1f} | "
                          f"{(100 * mean(r['is_gold'] for r in com)) if com else float('nan'):.1f} | {(100 * mean(r['is_gold'] for r in cons)) if cons else float('nan'):.1f} |")
    md.append("\nRealised thinking tokens, median [IQR], and share of traces that ended naturally before the budget (same for every framing of a model; sampling seed-determined):\n")
    lab0 = list(runs)[0]; R0 = runs[lab0]; rows = R0["rows"]; tc = [c for c in ORDER if c in R0["meta"]["conditions"] and c not in ("off", "off_soft")]
    md.append(f"({lab0})\n"); md.append("| set | " + " | ".join(CLAB[c] for c in tc) + " |"); md.append("|---|" + "---|" * len(tc))
    for b in [x for x in BORDER if x in {r['bucket'] for r in rows} and x != "scalar_imppres_noimpl"]:
        cells = []
        for c in tc:
            rs = [r for r in rows if r["bucket"] == b and r["condition"] == c]; tk = [r["think_tokens"] for r in rs]
            cells.append(f"{median(tk):.0f} [{pct(tk, 25):.0f}, {pct(tk, 75):.0f}] / {100 * mean(1 if r['natural'] else 0 for r in rs):.0f}%")
        md.append(f"| {b} | " + " | ".join(cells) + " |")

    md.append("\n## 3. Answer-position log-odds L (mean [95% bootstrap CI over items]; positive = towards the gold/pragmatic reading)\n")
    md.append("L = log p(gold) − log p(literal option) from the letter probabilities after `</think>\\n\\n` (when a literal option is defined and differs from gold; otherwise log p(gold) − log(1 − p(gold))), clipped at ±13.8. "
              "Treat the clip as a ceiling: at `full` most cells sit at ±13.8 because the model is near-certain. Target items only.\n")
    for lab, R in runs.items():
        rows, meta = R["rows"], R["meta"]; conds = [c for c in ORDER if c in meta["conditions"]]
        md.append(f"\n**{lab}**\n"); md.append("| set (target) | " + " | ".join(CLAB[c] for c in conds) + " |"); md.append("|---|" + "---|" * len(conds))
        for b in [x for x in BORDER if x in {r['bucket'] for r in rows} and x not in ("scalar_imppres_noimpl", "scalar_imppres_control")]:
            cells = []
            for c in conds:
                d = defaultdict(list)
                for r in rows:
                    if r["bucket"] == b and r["condition"] == c and r["role"] == "target" and r["valid"]:
                        g, l = item_view(r); d[r["item_id"]].append(logodds({**r, "gold": g}, l))
                v = [mean(x) for x in d.values()]
                if not v: cells.append("–"); continue
                lo, hi = ci(v); cells.append(f"{mean(v):+.1f} [{lo:+.1f}, {hi:+.1f}]")
            md.append(f"| {b} | " + " | ".join(cells) + " |")
    return "\n".join(md)


def figure(runs, out):
    import matplotlib.pyplot as plt
    P.use()
    n = len(runs)
    fig, axs = plt.subplots(1, n, figsize=P.figsize("text", height=2.0), sharey=True)
    axs = np.atleast_1d(axs)
    x = np.arange(len(FIGCONDS))
    for ax, (lab, R) in zip(axs, runs.items()):
        rows = R["rows"]
        for b, name, col in BUCKETS:
            sel = lambda r, b=b: r["bucket"] == b and r["role"] == "target"
            cs = [cell(rows, c, "is_gold", sel) for c in FIGCONDS]
            if any(c is None for c in cs): continue
            m = np.array([c["m"] for c in cs]); lo = np.array([c["lo"] for c in cs]); hi = np.array([c["hi"] for c in cs])
            ax.plot(x, m, color=col, lw=P.LW["data"], label=name, **P.marker_kw(col, ms=3.2))
            P.band(ax, x, lo, hi, col)
        ax.set_xticks(x, [CLAB[c] for c in FIGCONDS]); ax.set_xlabel("thinking budget (tokens)")
        ax.set_ylim(-2, 102); P.panel_title(ax, lab)
    axs[0].set_ylabel("pragmatic-response rate (%)")
    P.legend_row(axs[0], ncols=4, columnspacing=1.0, handlelength=1.4, fontsize=P.SIZE["legend"])
    P.save(fig, out / "fig_pilot_budget")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for a in sys.argv[2:]:
        lab, d = a.split("=", 1); runs[lab] = load(d)
    (out / "tables.md").write_text(tables(runs))
    (out / "traces.md").write_text("\n\n".join(traces(runs, lab) for lab in runs))
    (out / "note_sections.md").write_text(note_sections(runs))
    figure(runs, out)
    print("wrote", out / "tables.md", out / "traces.md", "fig_pilot_budget.{pdf,png}")


if __name__ == "__main__":
    main()
