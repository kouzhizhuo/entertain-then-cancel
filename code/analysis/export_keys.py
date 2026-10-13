#!/usr/bin/env python3
"""Export the P2 pilot runs to the json2tex key scheme of notes/RESULTS_KEYS.md  ->  exp/results/pilot/summary_pilot.json (+ meta.json).

    python3 exp/analysis/export_keys.py            # regenerate (CPU only, ~1-2 min)

Key grammar  <prefix>.<model>.<set>.<format>.<condition>.<metric>.<stat>   (prefix `pilot` is added by json2tex:  summary_pilot.json:pilot)
  percentages 0-100, deltas in pp; std = std over the 3 seeds of the per-seed rate; lo/hi = 95% percentile bootstrap over ITEMS of the
  seed-averaged per-item rate; n = number of items.  L (log-odds) keys carry a sibling "<key>__digits": 2.
Numbers are computed by the SAME functions as exp/harness/pilot_summary.py (cell / delta / logodds / ci are imported, not re-implemented).

Which run feeds which keys (pilot only; nothing here is reportable: dev-like item sampling, no held-out split, T=0.6 (v3) / 0.7 (v2)):
  qwen3-{4b,8b,14b}       pilot_v3_felicity/*  -> a1 fmt `fel`, a3 fmt `inf` (yes/no inference question), b1-indirect fmt `mc`, b2 fmt `yn`
  qwen3-{8b,14b}          pilot_v3_default/*   -> a1 fmt `tf`,  a3 fmt `nli`   (b1/b2 are the same prompts as in the felicity run: not repeated)
  qwen3-{4b,8b}-t07       pilot_v2_default/*   -> a1 `tf`, a3 `nli`, b1-indirect `mc`, b2 `yn`   (T=0.7; conditions off/b0/c0512/c2048/nat only)
  qwen3-{4b,8b}-t07       pilot_v2_alt/*       -> a1 `alt`, a3 `alt` (speaker framing; NOT in the RESULTS_KEYS format list, see README_keys.md)
Sets: a1 (Bott-Noveck-style underinformative, targets), a1-ctrl (its controls), a3 (IMPPRES scalar targets), a3-<trigger>, a3r-<relation>
(extension: by IMPPRES relation), a3-ctrl (IMPPRES log=prag controls), b1-indirect (Hu), b2 (LUDWIG).  The numerals `no_impl` bucket is dropped
(lead, plan v1.2.6: ill-posed under the inference question).
"""
import json, sys, time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "harness"))
import pilot_summary as PS  # noqa: E402  (cell, delta, logodds, ci, load)

RES = EXP / "results"
OUT = RES / "pilot" / "summary_pilot.json"

MODEL = {"Qwen3-4B": "qwen3-4b", "Qwen3-8B": "qwen3-8b", "Qwen3-14B": "qwen3-14b"}
TRIG = {"quantifiers": "quant", "numerals_2_3": "num23", "numerals_10_100": "num10", "connectives": "conn",
        "gradable_adjective": "gradadj", "gradable_verb": "gradverb", "modals": "modal"}
REL = {"implicature_PtoN": "ptn", "implicature_NtoP": "ntp", "negated implicature_P": "negp", "negated implicature_N": "negn",
       "reverse negated implicature_P": "revnegp", "reverse negated implicature_N": "revnegn"}
TARGET_SETS = lambda s: s == "a1" or s.startswith("a3-") and s != "a3-ctrl" or s == "a3" or s.startswith("a3r-")
CTRL_SETS = {"a1-ctrl", "a3-ctrl"}
PCI_SETS = {"b1-indirect", "b2"}


def cond_key(c):
    """harness condition name -> RESULTS_KEYS condition token"""
    if c == "off": return "off"
    if c == "off_soft": return "offsoft"
    if c == "think_b0": return "b0"
    if c == "think_full": return "nat"
    if c.startswith("think_b") and c.endswith("_plain"): return f"p{int(c[7:-6]):04d}"
    if c.startswith("think_b"): return f"c{int(c[7:]):04d}"
    raise ValueError(c)


def sets_of(r):
    b, role = r["bucket"], r["role"]
    if b == "scalar_imppres" and role == "target":
        s = ["a3"]
        if r["trigger"] in TRIG: s.append(f"a3-{TRIG[r['trigger']]}")
        if r["rel"] in REL: s.append(f"a3r-{REL[r['rel']]}")
        return s
    if b == "scalar_imppres_control": return ["a3-ctrl"]
    if b == "scalar_underinformative": return ["a1"] if role == "target" else ["a1-ctrl"]
    if b == "indirect_answer_ruis": return ["b2"]
    if b == "indirect_speech_hu": return ["b1-indirect"]
    return []  # scalar_imppres_noimpl: dropped


def load_items():
    idx = {}
    for f in sorted((EXP / "data" / "items").glob("[!._]*.jsonl")):
        for l in open(f):
            r = json.loads(l); idx[r["id"]] = r
    return idx


ITEMS = load_items()


def gold_lit(r):
    it = ITEMS[r["item_id"]]
    v = (it.get("variants") or {}).get(r.get("variant")) if r.get("variant") not in (None, "default", "alt") else None
    return (v["gold"], v["literal"]) if v else (it["gold"], it["literal"])


def rnd(x, k=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def rate_block(rows, cond, key):
    """{mean,std,lo,hi,n} of a rate (pp) via pilot_summary.cell (rows already restricted to one set)."""
    c = PS.cell(rows, cond, key, lambda r: True)
    if c is None: return None
    return {"mean": rnd(c["m"]), "std": rnd(c["std"]), "lo": rnd(c["lo"]), "hi": rnd(c["hi"]), "n": int(c["n"])}


def delta_block(rows, cond, key):
    """cond minus off, pp: paired over items (pilot_summary.delta) + std over seeds of the per-seed paired delta."""
    d = PS.delta(rows, cond, "off", key, lambda r: True)
    if d is None: return None
    ps = []
    for s in sorted({r["seed"] for r in rows}):
        rs = [r for r in rows if r["seed"] == s]
        a, b = PS.per_item(rs, cond, key), PS.per_item(rs, "off", key)
        ks = set(a) & set(b)
        if ks: ps.append(100 * mean(a[k] - b[k] for k in ks))
    return {"mean": rnd(d[0]), "std": rnd(stdev(ps)) if len(ps) > 1 else None, "lo": rnd(d[1]), "hi": rnd(d[2]), "n": int(d[3])}


def L_block(rows, cond):
    per_item, per_seed = defaultdict(list), defaultdict(list)
    for r in rows:
        if r["condition"] != cond or not r["valid"]: continue
        g, l = gold_lit(r)
        v = PS.logodds({**r, "gold": g}, l)
        per_item[r["item_id"]].append(v); per_seed[r["seed"]].append(v)
    if not per_item: return None
    v = [mean(x) for x in per_item.values()]
    lo, hi = PS.ci(v)
    ss = [mean(x) for x in per_seed.values()]
    return {"mean": rnd(mean(v), 2), "std": rnd(stdev(ss), 2) if len(ss) > 1 else None, "lo": rnd(lo, 2), "hi": rnd(hi, 2), "n": len(v)}


def les_block(rows, cond):
    """literal-error share (%) = literal answers / wrong answers; CI: resample items, ratio of totals."""
    per_item = defaultdict(lambda: [0, 0]); per_seed = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["condition"] != cond or not r["valid"] or r["is_gold"]: continue
        lit = 1 if r["is_literal"] else 0
        per_item[r["item_id"]][0] += lit; per_item[r["item_id"]][1] += 1
        per_seed[r["seed"]][0] += lit; per_seed[r["seed"]][1] += 1
    tot = sum(v[1] for v in per_item.values())
    if tot == 0: return None
    arr = np.array(list(per_item.values()), float)
    rng = np.random.default_rng(0); bs = []
    for _ in range(4000):
        s = arr[rng.integers(0, len(arr), len(arr))]
        bs.append(s[:, 0].sum() / max(s[:, 1].sum(), 1))
    ss = [100 * v[0] / v[1] for v in per_seed.values() if v[1]]
    return {"mean": rnd(100 * arr[:, 0].sum() / arr[:, 1].sum()), "std": rnd(stdev(ss)) if len(ss) > 1 else None,
            "lo": rnd(100 * np.percentile(bs, 2.5)), "hi": rnd(100 * np.percentile(bs, 97.5)), "n": int(tot)}


def token_block(rows, cond):
    per_item, per_seed, allv = defaultdict(list), defaultdict(list), []
    nat = []
    for r in rows:
        if r["condition"] != cond: continue
        per_item[r["item_id"]].append(r["think_tokens"]); per_seed[r["seed"]].append(r["think_tokens"]); allv.append(r["think_tokens"])
        if r["natural"] is not None: nat.append(1 if r["natural"] else 0)
    if not allv: return None, None
    v = [mean(x) for x in per_item.values()]; lo, hi = PS.ci(v)
    ss = [mean(x) for x in per_seed.values()]
    tok = {"mean": rnd(mean(allv), 1), "median": rnd(median(allv), 1), "q25": rnd(np.percentile(allv, 25), 1), "q75": rnd(np.percentile(allv, 75), 1),
           "std": rnd(stdev(ss), 1) if len(ss) > 1 else None, "lo": rnd(lo, 1), "hi": rnd(hi, 1), "n": len(v)}
    return tok, ({"mean": rnd(100 * mean(nat))} if nat else None)


def free_blocks(free_rows, setkey, cond):
    rs = [r for r in free_rows if setkey in sets_of(r) and r["condition"] == cond]
    if not rs: return {}
    def per_item_rate(vals_by_row):
        d = defaultdict(list)
        for r, v in vals_by_row: d[r["item_id"]].append(v)
        v = [mean(x) for x in d.values()]; lo, hi = PS.ci(v)
        ss = []
        for s in sorted({r["seed"] for r, _ in vals_by_row}):
            x = [v_ for r, v_ in vals_by_row if r["seed"] == s]
            ss.append(100 * mean(x))
        return {"mean": rnd(100 * mean(v)), "std": rnd(stdev(ss)) if len(ss) > 1 else None, "lo": rnd(100 * lo), "hi": rnd(100 * hi), "n": len(v)}
    out = {"hedge": per_item_rate([(r, 0.0 if r["committed"] else 1.0) for r in rs]),
           "hedgew": per_item_rate([(r, 1.0 if r["hedge_words"] else 0.0) for r in rs])}
    com = [r for r in rs if r["committed"]]
    if com:
        m = "Rfree" if TARGET_SETS(setkey) else "accfree"
        out[m] = per_item_rate([(r, 1.0 if r["is_gold"] else 0.0) for r in com])
    return out


def export_run(model, fmt_of, run_dir, sets_wanted=None):
    R = PS.load(run_dir)
    rows = R["rows"]
    by_set = defaultdict(list)
    for r in rows:
        for s in sets_of(r):
            by_set[s].append(r)
    free_by = R["free"]
    conds = [c for c in PS.ORDER if c in R["meta"]["conditions"]]
    out = {}
    for s, srows in by_set.items():
        fmt = fmt_of(s)
        if fmt is None: continue
        node_f = out.setdefault(s, {}).setdefault(fmt, {})
        for c in conds:
            ck = cond_key(c); node = node_f.setdefault(ck, {})
            if TARGET_SETS(s):
                node["R"] = rate_block(srows, c, "is_gold")
                lit = rate_block(srows, c, "is_literal")
                if lit: node["lit"] = lit
                if c != "off":
                    node["dR"] = delta_block(srows, c, "is_gold")
                    if lit: node["dlit"] = delta_block(srows, c, "is_literal")
            else:
                node["acc"] = rate_block(srows, c, "is_gold")
                if c != "off": node["dacc"] = delta_block(srows, c, "is_gold")
                if s == "b1-indirect":
                    node["lit"] = rate_block(srows, c, "is_literal")
                    if c != "off": node["dlit"] = delta_block(srows, c, "is_literal")
                    node["les"] = les_block(srows, c)
            if s not in CTRL_SETS:
                node["L"] = L_block(srows, c)
            tok, nat = token_block(srows, c)
            node["tokens"] = tok
            if nat: node["natshare"] = nat
            for k in [k for k, v in node.items() if v is None]: del node[k]
        for c in ("off", "think_full"):
            fb = free_blocks(free_by, s, c)
            for k, v in fb.items():
                node_f.setdefault(cond_key(c), {})[k] = v
    return out, R


def add_digits(node):
    """for every metric named L add '<stat>__digits': 2 next to each stat key (json2tex convention)."""
    if isinstance(node, dict):
        for k in list(node):
            if k == "L" and isinstance(node[k], dict):
                for st in list(node[k]):
                    if st != "n": node[k][f"{st}__digits"] = 2
            else:
                add_digits(node[k])


def main():
    t0 = time.time()
    tree, sources = {}, {}

    def merge(model, part, src):
        node = tree.setdefault(model, {})
        for s, fm in part.items():
            for fmt, cm in fm.items():
                dst = node.setdefault(s, {}).setdefault(fmt, {})
                for c, ms in cm.items():
                    assert c not in dst or not (set(dst[c]) & set(ms) - {"hedge", "hedgew", "Rfree", "accfree"}), (model, s, fmt, c)
                    dst.setdefault(c, {}).update(ms)
        sources.setdefault(model, []).append(src)

    fel = lambda s: {"a1": "fel", "a1-ctrl": "fel"}.get(s, "inf" if s.startswith("a3") else "mc" if s == "b1-indirect" else "yn")
    dft = lambda s: {"a1": "tf", "a1-ctrl": "tf"}.get(s) or ("nli" if s.startswith("a3") else None)      # b1/b2 not repeated
    v2d = lambda s: {"a1": "tf", "a1-ctrl": "tf"}.get(s) or ("nli" if s.startswith("a3") else "mc" if s == "b1-indirect" else "yn")
    alt = lambda s: "alt" if (s.startswith("a1") or s.startswith("a3")) else None
    plan = []
    for m in ("Qwen3-4B", "Qwen3-8B", "Qwen3-14B"): plan.append((MODEL[m], fel, f"pilot_v3_felicity/{m}"))
    for m in ("Qwen3-8B", "Qwen3-14B"): plan.append((MODEL[m], dft, f"pilot_v3_default/{m}"))
    for m in ("Qwen3-4B", "Qwen3-8B"):
        plan.append((MODEL[m] + "-t07", v2d, f"pilot_v2_default/{m}"))
        plan.append((MODEL[m] + "-t07", alt, f"pilot_v2_alt/{m}"))
    runs_meta = {}
    for model, fmt_of, rd in plan:
        d = RES / rd
        if not (d / "samples.jsonl").exists():
            print("skip (missing):", rd); continue
        part, R = export_run(model, fmt_of, d)
        merge(model, part, rd)
        m = R["meta"]
        runs_meta[rd.replace("/", "-")] = {"model": m["model_name"], "framing": m["config"].get("variant", "default"), "config_sha16": m["config_sha16"],
                         "temperature": m["sampling"]["temperature"], "n_items": m["n_items"], "n_samples": m["n_samples"],
                         "seeds": m["seeds"], "conditions": m["conditions"], "tokens": m["total_generated_tokens"],
                         "wall_s": round(m["wall_seconds"]), "vllm": m["vllm"], "finished": m["finished"]}
        print(f"exported {rd}  ({time.time() - t0:.0f}s)")
    add_digits(tree)
    tree["_meta"] = {
        "status": "PILOT numbers, draft-only: dev-like item sampling (stratified/random draws from the released sets, item_seed 1234), no dev/test split, "
                  "3 seeds, T=0.6 (pilot_v3) or 0.7 (pilot_v2, model suffix -t07), cap 8192 tokens, letter-constrained greedy readout. NOT reportable.",
        "units": "percent 0-100 (deltas in pp); std over seeds; lo/hi 95% bootstrap over items; L = answer-position log-odds gold:literal (clipped +-13.8)",
        "runs": runs_meta, "sources": sources,
        "dropped": "scalar_imppres_noimpl numerals control (ill-posed under the inference question)",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "script": "exp/analysis/export_keys.py"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tree, indent=1, sort_keys=True))
    print("wrote", OUT, f"({OUT.stat().st_size / 1e6:.1f} MB)")

    # ---- meta.json (prefix `meta`) ------------------------------------------------------------------------------------------
    man = json.load(open(EXP / "data" / "manifest.json"))
    tok = sum(v["tokens"] for v in runs_meta.values()); wall = sum(v["wall_s"] for v in runs_meta.values())
    meta = {"items": {k: v["n"] for k, v in sorted(man.items())},
            "pilot": {"runs": dict(runs_meta),
                      "seeds": [0, 1, 2], "generated_tokens_total": tok, "wall_seconds_total": wall, "gpu_hours": round(wall / 3600, 2)},
            "compute": {"gpu": "NVIDIA H800 80GB", "host": "<host>", "cuda_visible_devices": "4", "gpus_per_run": 1,
                        "vllm": sorted({v["vllm"] for v in runs_meta.values()})[0]},
            "_note": "pilot runs only; GPU-hours = summed run wall time (includes model load) x 1 GPU. Item counts are the normalised sets in exp/data/items (manifest.json), not the pilot subsets (see pilot.<run>.n_items)."}
    (RES / "meta.json").write_text(json.dumps(meta, indent=1, sort_keys=True))
    print("wrote", RES / "meta.json")


if __name__ == "__main__":
    main()
