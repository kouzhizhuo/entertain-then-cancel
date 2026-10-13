#!/usr/bin/env python3
"""D-CONTINGENCY step (c): build the SELECTED set (SIT-train) and the EQUAL-DATA CONTROL from K=4 natural-thinking samples per train item (notes/METHOD_DESIGN.md "D-CONTINGENCY").
usage: python build_sit_sets.py --fc <SIT_train/<model>> --gsm <SIT_train_gsm/<model>> --model <name> --out data/sit_train/<model>
  inputs : <fc>/samples.jsonl (per item x seed: choice/valid/is_gold/perm/stage1_finish), <fc>/traces.jsonl (thought text), <gsm>/samples.jsonl (written with --save_text --reps 4)
  selected: <= 1 trace per item whose answer equals the ORIGINAL gold (and whose thought finished; a sample whose thought hit the cap counts as NOT correct for the pass rate, because it cannot be trained on); equal quota N per family (speaker / presup / literal-gold), items with 1-3 of 4 correct fill the quota
            first, items 4/4 fill the rest; N = min over families of (#mixed + #4of4), capped at 1500.
  control : the SAME items and quotas, ONE trace per item drawn at random from the finished and valid samples regardless of correctness (unfinished or invalid samples cannot be trained on).
  output  : selected.jsonl / control.jsonl  {id, family, set, messages:[{role:user,content}], target (= thought + close tag + answer, to follow "<think>\\n" after the generation prompt), correct, n_think_tokens, seed}
            and stats.json (pass-rate distribution, quotas, 4/4 share, token means of both sets, per-set composition). Seeds are fixed (0); nothing is trimmed silently."""
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import labels_of, perm_for, render_user

ap = argparse.ArgumentParser(); ap.add_argument("--fc", required=True); ap.add_argument("--gsm", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cap", type=int, default=1500); ap.add_argument("--nobal_only", action="store_true", help="ablation SIT-nobal: write ONLY nobal.jsonl + stats_nobal.json (selected/control untouched)"); ap.add_argument("--close", default="</think>\n\n"); a = ap.parse_args()
EXP = Path(__file__).resolve().parents[1]
spl = json.load(open(EXP / "splits" / "sit_train_families.json")); FAM = {s: f for f, ss in spl["families"].items() for s in ss}
items = {json.loads(l)["id"]: json.loads(l) for l in open(EXP / "data" / "items" / "sit_train.jsonl")}
gsm_items = {json.loads(l)["id"]: json.loads(l) for l in open(EXP / "data" / "items" / "sit_train_gsm.jsonl")}
cand = defaultdict(list)    # item id -> list of sample dicts {correct, finished, valid, target, n_think, seed}
thoughts = {}
for l in open(Path(a.fc) / "traces.jsonl"):
    r = json.loads(l); thoughts[(r["item_id"], r["seed"])] = r
for l in open(Path(a.fc) / "samples.jsonl"):
    r = json.loads(l); it = items[r["item_id"]]; tr = thoughts[(r["item_id"], r["seed"])]
    finished = r.get("stage1_finish") == "stop" and tr["finish"] == "stop"; valid = bool(r["valid"]); ch = r["choice"]
    perm = perm_for(it["id"], r["seed"], len(it["options"]), it.get("shuffle", True)); assert perm == r["perm"], "permutation mismatch"
    lab = labels_of(it)[perm.index(ch)] if valid else None
    cand[it["id"]].append({"correct": bool(valid and r["is_gold"] and finished), "finished": finished, "valid": valid, "seed": r["seed"], "n_think": tr["n_tokens"],
                           "target": (tr["text"] + a.close + lab) if (finished and valid) else None, "user": render_user(it, perm, "default", None)})
for l in open(Path(a.gsm) / "samples.jsonl"):
    r = json.loads(l)
    if r["cond"] != "think_full": continue
    base, k = r["id"].rsplit("#r", 1); it = gsm_items[base]
    fin = not r["unfinished"]
    cand[base].append({"correct": bool(r["correct"]) and fin, "finished": fin, "valid": fin, "seed": int(k), "n_think": r["tokens"], "target": r["text"] if fin else None,
                       "user": it["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."})
meta = {i: (items.get(i) or {"meta": gsm_items[i]["meta"]})["meta"] for i in cand}
byfam = defaultdict(lambda: {"mixed": [], "all4": [], "zero": [], "k": Counter()})
for i, ss in sorted(cand.items()):
    k = len(ss); c = sum(s["correct"] for s in ss); fam = meta[i]["family"]; byfam[fam]["k"][f"{c}/{k}"] += 1
    if c == 0: byfam[fam]["zero"].append(i)
    elif c == k: byfam[fam]["all4"].append(i)
    else: byfam[fam]["mixed"].append(i)
rng = random.Random(0)
N = min(a.cap, min(len(v["mixed"]) + len(v["all4"]) for v in byfam.values()))
sel_ids = {}
for fam, v in byfam.items():
    m, f4 = sorted(v["mixed"]), sorted(v["all4"]); rng.shuffle(m); rng.shuffle(f4)
    sel_ids[fam] = (m + f4)[:N]
if a.nobal_only:    # ABLATION SIT-nobal: same selection rule (one correct finished trace per item, mixed-pass-rate items first, then 4/4) over the UNIFIED pool, i.e. NO family quota (natural family mix); same total N as the selected set
    total = sum(len(v) for v in sel_ids.values()); r2 = random.Random(1)
    allm = sorted(i for v in byfam.values() for i in v["mixed"]); all4 = sorted(i for v in byfam.values() for i in v["all4"]); r2.shuffle(allm); r2.shuffle(all4); nb = (allm + all4)[:total]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); rows_nb = []
    for i in sorted(nb):
        good = [s for s in cand[i] if s["correct"] and s["target"]]; s = random.Random(f"selected|{i}").choice(good)    # the same trace the selected set uses for this item
        rows_nb.append({"id": i, "family": meta[i]["family"], "set": meta[i]["set"], "messages": [{"role": "user", "content": s["user"]}], "target": s["target"], "correct": True, "n_think_tokens": s["n_think"], "seed": s["seed"]})
    with open(out / "nobal.jsonl", "w") as f:
        for r in rows_nb: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mixed_set = set(allm); t = [r["n_think_tokens"] for r in rows_nb]
    st = {"model": a.model, "n": len(rows_nb), "family_counts": dict(Counter(r["family"] for r in rows_nb)), "share_4of4": sum(1 for r in rows_nb if r["id"] not in mixed_set) / len(rows_nb), "mean_think_tokens": sum(t) / len(t), "total_tokens": sum(t), "sets": dict(Counter(r["set"] for r in rows_nb))}
    json.dump(st, open(out / "stats_nobal.json", "w"), indent=1); print(json.dumps(st, indent=1)); sys.exit(0)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); stats = {"model": a.model, "quota_N_per_family": N, "cap": a.cap, "families": {}}
rows = {"selected": [], "control": []}
for fam, ids in sel_ids.items():
    for i in sorted(ids):
        ss = cand[i]
        good = [s for s in ss if s["correct"] and s["target"]]; assert good, i
        trainable = [s for s in ss if s["target"]]
        for kind, pool in (("selected", good), ("control", trainable)):
            s = random.Random(f"{kind}|{i}").choice(pool)
            rows[kind].append({"id": i, "family": fam, "set": meta[i]["set"], "messages": [{"role": "user", "content": s["user"]}], "target": s["target"], "correct": s["correct"], "n_think_tokens": s["n_think"], "seed": s["seed"]})
for kind in rows:
    with open(out / f"{kind}.jsonl", "w") as f:
        for r in rows[kind]: f.write(json.dumps(r, ensure_ascii=False) + "\n")
for fam, v in byfam.items():
    n4 = sum(1 for i in sel_ids[fam] if i in set(v["all4"]))
    stats["families"][fam] = {"items": len(v["mixed"]) + len(v["all4"]) + len(v["zero"]), "mixed": len(v["mixed"]), "all4": len(v["all4"]), "zero": len(v["zero"]), "pass_k_histogram": dict(v["k"]), "selected": len(sel_ids[fam]),
                              "selected_share_4of4": n4 / max(1, len(sel_ids[fam])), "selected_sets": dict(Counter(meta[i]["set"] for i in sel_ids[fam]))}
for kind in rows:
    t = [r["n_think_tokens"] for r in rows[kind]]
    stats[kind] = {"n": len(t), "mean_think_tokens": sum(t) / max(1, len(t)), "share_correct": sum(r["correct"] for r in rows[kind]) / max(1, len(t)), "total_tokens": sum(t), "by_family_mean_tokens": {fam: (lambda x: sum(x) / max(1, len(x)))([r["n_think_tokens"] for r in rows[kind] if r["family"] == fam]) for fam in byfam}}
json.dump(stats, open(out / "stats.json", "w"), indent=1); print(json.dumps(stats, indent=1))
