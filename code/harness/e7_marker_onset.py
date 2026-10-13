#!/usr/bin/env python
"""E7 marker onset (lead question): per natural-thinking trace, where does the first literalness marker (technically / strictly / logically) sit relative to the first prefix at which the answer-position log-odds L turns negative?
usage: python e7_marker_onset.py <E7 run dir with samples.jsonl + traces.jsonl> <tokenizer dir> <out.json> [config name, default E7]
L(k) = log p(gold) - log p(literal option) at the answer position after the SAME trace is plain-truncated at k tokens (k = 0, 32, ... 2048, then the natural end = n_tokens): the E7 conditions. Per (item, seed) trace:
 turn trace = L(0) >= 0 and some later prefix has L < 0; k* = smallest such prefix; m = token position of the first marker in the trace text (tokens before it, Qwen tokenizer; None if the trace has no marker).
Reported per scalar set (a3-quant, a3-conn, a3-modal, a3, a2): over turn traces: pre = share with a marker before k* (m < k*), post = marker only after k*, none = no marker in the trace; control = share of NON-turn traces containing a marker.
Shares are item-level (traces of an item averaged), CI = item bootstrap (2000)."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_items_from_config
from pilot_summary import logodds
run, tokdir, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(tokdir)
MARK = re.compile(r"\b(technically|strictly|logically)\b", re.I)
cfg = json.load(open(Path(__file__).resolve().parents[1] / "configs" / ((sys.argv[4] if len(sys.argv) > 4 else "E7") + ".json"))); its = {i["id"]: i for i in load_items_from_config(cfg)}
KS = {"think_b0_plain": 0, "think_b32_plain": 32, "think_b64_plain": 64, "think_b128_plain": 128, "think_b256_plain": 256, "think_b512_plain": 512, "think_b1024_plain": 1024, "think_b2048_plain": 2048}
L = defaultdict(dict)                      # (item, seed) -> {k or 'full': L}
for l in open(run / "samples.jsonl"):
    r = json.loads(l); it = its.get(r["item_id"])
    if it is None or not r["valid"]: continue
    k = KS.get(r["condition"], "full" if r["condition"] == "think_full" else None)
    if k is None: continue
    L[(r["item_id"], r["seed"])][k] = logodds({**r, "gold": it["gold"]}, it.get("literal"))
def sets_of(it):
    b = it["bucket"]
    if b == "a2": return ["a2"]
    sub = {"quantifiers": "a3-quant", "connectives": "a3-conn", "modals": "a3-modal"}.get(it.get("subset"))
    pre = "a3f" if b.startswith("a3f") else "a3"
    return [pre] + ([sub.replace("a3", pre, 1)] if sub else [])
per = defaultdict(lambda: defaultdict(list))   # set -> item -> list of per-trace dicts
for l in open(run / "traces.jsonl"):
    t = json.loads(l)
    if t.get("finish") != "stop" or t["item_id"] not in its: continue          # natural traces only
    ls = L.get((t["item_id"], t["seed"]));
    if not ls or 0 not in ls or "full" not in ls: continue
    n = t["n_tokens"]; ladder = sorted([(k, v) for k, v in ls.items() if k != "full" and k < n]) + [(n, ls["full"])]
    turn = ls[0] >= 0 and any(v < 0 for k, v in ladder if k > 0); kstar = next((k for k, v in ladder if k > 0 and v < 0), None) if turn else None
    m = MARK.search(t["text"]); mpos = len(tok.encode(t["text"][: m.start()], add_special_tokens=False)) if m else None
    rec = {"turn": turn, "pre": (turn and mpos is not None and mpos < kstar), "post": (turn and mpos is not None and mpos >= kstar), "none": (turn and mpos is None), "marker": mpos is not None}
    for s in sets_of(its[t["item_id"]]): per[s][t["item_id"]].append(rec)
def ci(v):
    v = np.asarray(v, float); rng = np.random.default_rng(0); b = v[rng.integers(0, len(v), (2000, len(v)))].mean(1); return round(float(np.percentile(b, 2.5)) * 100, 1), round(float(np.percentile(b, 97.5)) * 100, 1)
res = {}
for s, d in per.items():
    node = {}
    turn_items = {i: [r for r in rs if r["turn"]] for i, rs in d.items()}; turn_items = {i: rs for i, rs in turn_items.items() if rs}
    for key in ("pre", "post", "none"):
        v = [np.mean([r[key] for r in rs]) for rs in turn_items.values()]
        if len(v) >= 5: lo, hi = ci(v); node[key] = {"mean": round(100 * float(np.mean(v)), 1), "lo": lo, "hi": hi, "n": len(v), "mean__digits": 1, "lo__digits": 1, "hi__digits": 1}
    ctrl = {i: [r for r in rs if not r["turn"]] for i, rs in d.items()}; v = [np.mean([r["marker"] for r in rs]) for rs in ctrl.values() if rs]
    if len(v) >= 5: lo, hi = ci(v); node["ctrl_marker"] = {"mean": round(100 * float(np.mean(v)), 1), "lo": lo, "hi": hi, "n": len(v), "mean__digits": 1, "lo__digits": 1, "hi__digits": 1}
    node["n_turn_traces"] = int(sum(len(v) for v in turn_items.values())); node["n_traces"] = int(sum(len(v) for v in d.values()))
    res[s] = node
json.dump(res, open(out, "w"), indent=1); print({s: {k: (v["mean"] if isinstance(v, dict) else v) for k, v in n.items()} for s, n in res.items()})
