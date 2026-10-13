#!/usr/bin/env python3
"""E1F artefact checks (lead request): per model x dose x set, from the raw API runs (results/api/<provider_model>/<dose>/samples.jsonl):
 fail.{invalid,empty,refusal,drift,trunc}  failure shares in % of calls (drift = answer starts with neither a bare option letter nor 'LETTER. option text');
 extract_mismatch  rows whose stored `valid` differs from re-running common.letter_from_text on the stored answer text (same function for every dose; 0 expected);
 dRx  paired delta of the pragmatic rate (pp) vs the lowest dose over items valid at BOTH doses, with a paired CLUSTER bootstrap (10,000; cluster = shared IMPPRES premise / A2 scale / item), n items and n clusters;
 dRlo/dRhi  worst/best-case delta with failures imputed adversarially (failure at the dose = non-pragmatic and at the base = pragmatic, and the reverse).
Merges the blocks into results/E1F/summary.json (same node layout e1f.<model>.<set>.inf.<dose>.*) and writes results/E1F/artefact.json."""
import json, random, re, sys
from collections import defaultdict
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "harness"))
from common import labels_of, letter_from_text, load_items_from_config
API = EXP / "results" / "api"; DOSE_RANK = {"think": 3, "off512": -1, "eff-min": 0, "off": 0, "bud0": 0, "bud128": 1, "eff-low": 1, "bud1k": 2, "eff-med": 2, "bud4k": 3, "eff-high": 3, "bud8k": 3, "bud16k": 4}
items = {i["id"]: i for i in load_items_from_config(json.load(open(EXP / "configs/E1f_items.json")))}
SETS = {"a3": {"f_a3q", "f_a3c", "f_a3m", "f_a3l"}, "a3-quant": {"f_a3q"}, "a3-conn": {"f_a3c"}, "a3-modal": {"f_a3m"}, "a3-gradadj": {"f_a3l"}, "a2": {"f_a2"}, "b2": {"f_b2"}}
REFUSE = re.compile(r"\b(i can'?t|i cannot|i can not|unable to|i'?m sorry|i apologi[sz]e|not able to|as an ai|i won'?t)\b", re.I)
BARE = re.compile(r"^\(?[A-F]\)?[\.\):]?\s*$"); ECHO = re.compile(r"^\(?[A-F]\)?[\.\):]\s+\S")
def cluster_of(iid):
    m = items[iid]["meta"]; return m.get("premise") or (("scale:" + m["scale"]) if m.get("scale") else iid)
V4 = {"deepseek_deepseek-v4-pro": "deepseek-v4-pro", "deepseek_deepseek-v4-flash-0731": "deepseek-v4-flash"}
def tok(name):
    m = name.split("_", 1)[1]
    return V4.get(m, m.replace(".", "p"))
def kind(r):
    t = (r.get("answer_text") or "").strip()
    if not t: return "empty"
    if not r["valid"]: return "refusal" if REFUSE.search(t) else "invalid"
    if BARE.match(t) or ECHO.match(t): return "ok"
    return "drift"
runs = {}
for f in sorted(API.glob("*/*/samples.jsonl")):
    runs.setdefault(tok(f.parent.parent.name), {})[f.parent.name] = [json.loads(l) for l in open(f)]
def agg(rows):
    """one record per item: valid if any sample parsed; is_gold = mean over the valid samples (seeds are averaged within an item)."""
    d = defaultdict(list)
    for r in rows: d[r["item_id"]].append(r)
    out = {}
    for k, v in d.items():
        ok = [float(r["is_gold"]) for r in v if r["valid"]]
        out[k] = {"valid": bool(ok), "is_gold": (sum(ok) / len(ok)) if ok else None}
    return out
def pct(x, n): return round(100 * x / n, 1) if n else None
def boot_paired(pairs, clus, n=10000, seed=0):
    by = defaultdict(list)
    for (iid, a, b) in pairs: by[clus[iid]].append(a - b)
    keys = list(by); rng = random.Random(seed); vals = []
    for _ in range(n):
        s = c = 0
        for k in (rng.choice(keys) for _ in keys): s += sum(by[k]); c += len(by[k])
        vals.append(100 * s / c)
    vals.sort(); return round(vals[int(.025 * n)], 1), round(vals[int(.975 * n) - 1], 1), len(keys)
art = {}; mism = defaultdict(int)
for model, doses in runs.items():
    base = min(doses, key=lambda d: DOSE_RANK[d])
    for sname, buckets in SETS.items():
        sel = lambda r: items[r["item_id"]]["bucket"] in buckets
        B = agg([r for r in doses[base] if sel(r)])
        for dose, rows in doses.items():
            rs = [r for r in rows if sel(r)]; n = len(rs)
            if not n: continue
            ks = [kind(r) for r in rs]
            for r in rs:
                it = items[r["item_id"]]; pos = letter_from_text(r["answer_text"], len(it["options"]), labels_of(it))
                if (pos is not None) != bool(r["valid"]): mism[(model, dose)] += 1
            node = {"fail": {k: pct(ks.count(k), n) for k in ("invalid", "empty", "refusal", "drift")}, "n_calls": n}
            node["fail"]["trunc"] = pct(sum(r["finish"] not in ("stop", "end_turn") for r in rs), n)
            node["fail"]["any"] = pct(sum(k != "ok" for k in ks), n); node["fail"]["valid_share"] = pct(sum(bool(r["valid"]) for r in rs), n)
            if dose != base:
                D = agg(rs); common = [k for k in D if k in B]
                both = [(k, float(D[k]["is_gold"]), float(B[k]["is_gold"])) for k in common if D[k]["valid"] and B[k]["valid"]]
                if both:
                    clus = {k: cluster_of(k) for k, _, _ in both}; lo, hi, nc = boot_paired(both, clus)
                    node["dRx"] = {"mean": round(100 * sum(a - b for _, a, b in both) / len(both), 1), "lo": lo, "hi": hi, "n": len(both), "n_clusters": nc, "n_dropped": len(common) - len(both)}
                    adv = lambda lo_case: 100 * sum(((0.0 if lo_case else 1.0) if not D[k]["valid"] else float(D[k]["is_gold"])) - ((1.0 if lo_case else 0.0) if not B[k]["valid"] else float(B[k]["is_gold"])) for k in common) / len(common)
                    node["dRlo"] = round(adv(True), 1); node["dRhi"] = round(adv(False), 1)
            art.setdefault(model, {}).setdefault(sname, {}).setdefault("inf", {})[dose] = node
json.dump(art, open(EXP / "results/E1F/artefact.json", "w"), indent=1)
# merge into summary.json keys
sm = json.load(open(EXP / "results/E1F/summary.json"))
def dg(d, k=1):
    for a in list(d):
        if isinstance(d[a], (int, float)) and not a.endswith("__digits") and not isinstance(d[a], bool) and a in ("mean", "lo", "hi"): d[a + "__digits"] = k
for m, sets in art.items():
    for s, fm in sets.items():
        for f, ds in fm.items():
            for dose, node in ds.items():
                tgt = sm.get(m, {}).get(s, {}).get(f, {}).get(dose)
                if tgt is None: continue
                tgt["fail"] = {k: {"mean": v, "mean__digits": 1} for k, v in node["fail"].items() if v is not None}; tgt["n_calls"] = node["n_calls"]
                if "dRx" in node:
                    x = dict(node["dRx"]); x.update({"mean__digits": 1, "lo__digits": 1, "hi__digits": 1}); tgt["dRx"] = x
                    tgt["dRbound"] = {"lo": {"mean": node["dRlo"], "mean__digits": 1}, "hi": {"mean": node["dRhi"], "mean__digits": 1}}
json.dump(sm, open(EXP / "results/E1F/summary.json", "w"), indent=1)
print("extract mismatches:", dict(mism) or 0)
for m, sets in art.items():
    for dose, node in sets["a3"]["inf"].items():
        f = node["fail"]; x = node.get("dRx"); print(f"{m:20s} {dose:9s} n={node['n_calls']} valid={f['valid_share']} inval={f['invalid']} empty={f['empty']} refuse={f['refusal']} drift={f['drift']} trunc={f['trunc']}", ("dRx %+.1f [%+.1f,%+.1f] n=%d cl=%d drop=%d bounds [%+.1f,%+.1f]" % (x['mean'], x['lo'], x['hi'], x['n'], x['n_clusters'], x['n_dropped'], node['dRlo'], node['dRhi'])) if x else "")

# ---- second format (api_mc): 3-way NLI forced choice on the same a3-quant / a3-conn items; fmt token 'nli3'; lowest vs highest dose only
MC = EXP / "results" / "api_mc"
if MC.exists():
    items.update({i["id"]: i for i in load_items_from_config(json.load(open(EXP / "configs/E1f_mc.json")))})
    mc = {}
    for f in sorted(MC.glob("*/*/samples.jsonl")): mc.setdefault(tok(f.parent.parent.name), {})[f.parent.name] = [json.loads(l) for l in open(f)]
    art2 = {}
    for model, doses in mc.items():
        base = min(doses, key=lambda d: DOSE_RANK[d])
        for sname, buckets in (("a3-quant", {"f_a3q"}), ("a3-conn", {"f_a3c"}), ("a3", {"f_a3q", "f_a3c"})):
            sel = lambda r: items[r["item_id"]]["bucket"] in buckets
            B = agg([r for r in doses[base] if sel(r)])
            for dose, rows in doses.items():
                rs = [r for r in rows if sel(r)]; n = len(rs)
                if not n: continue
                ks = [kind(r) for r in rs]
                node = {"fail": {k: pct(ks.count(k), n) for k in ("invalid", "empty", "refusal", "drift")}, "n_calls": n}
                node["fail"]["trunc"] = pct(sum(r["finish"] not in ("stop", "end_turn") for r in rs), n); node["fail"]["valid_share"] = pct(sum(bool(r["valid"]) for r in rs), n)
                vals = [float(r["is_gold"]) for r in rs if r["valid"]]; node["R"] = {"mean": round(100 * sum(vals) / len(vals), 1), "n": len(vals)} if vals else None
                if dose != base:
                    D = agg(rs); common = [k for k in D if k in B]
                    both = [(k, float(D[k]["is_gold"]), float(B[k]["is_gold"])) for k in common if D[k]["valid"] and B[k]["valid"]]
                    if both:
                        clus = {k: cluster_of(k) for k, _, _ in both}; lo, hi, nc = boot_paired(both, clus)
                        node["dRx"] = {"mean": round(100 * sum(a - b for _, a, b in both) / len(both), 1), "lo": lo, "hi": hi, "n": len(both), "n_clusters": nc, "n_dropped": len(common) - len(both)}
                        adv = lambda lo_case: 100 * sum(((0.0 if lo_case else 1.0) if not D[k]["valid"] else float(D[k]["is_gold"])) - ((1.0 if lo_case else 0.0) if not B[k]["valid"] else float(B[k]["is_gold"])) for k in common) / len(common)
                        node["dRlo"] = round(adv(True), 1); node["dRhi"] = round(adv(False), 1)
                art2.setdefault(model, {}).setdefault(sname, {}).setdefault("nli3", {})[dose] = node
    json.dump(art2, open(EXP / "results/E1F/artefact_nli3.json", "w"), indent=1)
    sm = json.load(open(EXP / "results/E1F/summary.json"))
    for m, sets in art2.items():
        for sname, fm in sets.items():
            for f, ds in fm.items():
                for dose, node in ds.items():
                    tgt = sm.setdefault(m, {}).setdefault(sname, {}).setdefault(f, {}).setdefault(dose, {})
                    tgt["fail"] = {k: {"mean": v, "mean__digits": 1} for k, v in node["fail"].items() if v is not None}; tgt["n_calls"] = node["n_calls"]
                    if node.get("R"): tgt["R"] = {"mean": node["R"]["mean"], "n": node["R"]["n"], "mean__digits": 1}
                    if "dRx" in node:
                        x = dict(node["dRx"]); x.update({"mean__digits": 1, "lo__digits": 1, "hi__digits": 1}); tgt["dRx"] = x
                        tgt["dRbound"] = {"lo": {"mean": node["dRlo"], "mean__digits": 1}, "hi": {"mean": node["dRhi"], "mean__digits": 1}}
    json.dump(sm, open(EXP / "results/E1F/summary.json", "w"), indent=1)
    for m, sets in art2.items():
        for dose, node in sets["a3"]["nli3"].items():
            f = node["fail"]; x = node.get("dRx"); print("NLI3", f"{m:20s} {dose:9s} n={node['n_calls']} valid={f['valid_share']} drift={f['drift']} trunc={f['trunc']} R={node['R']}", ("dRx %+.1f [%+.1f,%+.1f] n=%d bounds [%+.1f,%+.1f]" % (x['mean'], x['lo'], x['hi'], x['n'], node['dRlo'], node['dRhi'])) if x else "")
