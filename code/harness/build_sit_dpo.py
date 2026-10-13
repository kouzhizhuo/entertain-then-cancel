#!/usr/bin/env python3
"""ATTEMPT 4 (SIT-DPO) pairs, notes/METHOD_DESIGN.md "ATTEMPT 4". From the D train pool and the SAME K=4 natural-thinking samples (no new sampling): mixed-pass items only (1-3 of 4 correct against the ORIGINAL gold; correct = valid, equals gold, thought finished);
chosen = one correct trace, rejected = one incorrect FINISHED trace with a valid wrong answer (a trace that hit the cap or has no valid answer has no answer to be wrong about and is not used as rejected), both drawn at random (seed 0). Equal quota per family (speaker / presup / literal incl. FOLIO + GSM8K),
N = the smallest family's pair pool, capped at 1500. Control = label-shuffled: the SAME pairs, chosen/rejected orientation assigned by a fair coin (seed 0).
usage: python build_sit_dpo.py --fc <SIT_train/<model>> --gsm <SIT_train_gsm/<model>> --model <name> --out data/sit_dpo/<model>      writes pairs.jsonl, pairs_shuffled.jsonl, stats.json"""
import argparse, json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import labels_of, perm_for, render_user
ap = argparse.ArgumentParser(); ap.add_argument("--fc", required=True); ap.add_argument("--gsm", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cap", type=int, default=1500); ap.add_argument("--close", default="</think>\n\n"); a = ap.parse_args()
EXP = Path(__file__).resolve().parents[1]
items = {json.loads(l)["id"]: json.loads(l) for l in open(EXP / "data" / "items" / "sit_train.jsonl")}
gsm_items = {json.loads(l)["id"]: json.loads(l) for l in open(EXP / "data" / "items" / "sit_train_gsm.jsonl")}
cand = defaultdict(list); thoughts = {}
for l in open(Path(a.fc) / "traces.jsonl"):
    r = json.loads(l); thoughts[(r["item_id"], r["seed"])] = r
for l in open(Path(a.fc) / "samples.jsonl"):
    r = json.loads(l); it = items[r["item_id"]]; tr = thoughts[(r["item_id"], r["seed"])]
    finished = r.get("stage1_finish") == "stop" and tr["finish"] == "stop"; valid = bool(r["valid"]); ch = r["choice"]
    perm = perm_for(it["id"], r["seed"], len(it["options"]), it.get("shuffle", True)); assert perm == r["perm"]
    lab = labels_of(it)[perm.index(ch)] if valid else None; ok = bool(valid and r["is_gold"] and finished)
    cand[it["id"]].append({"correct": ok, "wrong": bool(finished and valid and not r["is_gold"]), "seed": r["seed"], "n_think": tr["n_tokens"], "target": (tr["text"] + a.close + lab) if (finished and valid) else None, "user": render_user(it, perm, "default", None)})
for l in open(Path(a.gsm) / "samples.jsonl"):
    r = json.loads(l)
    if r["cond"] != "think_full": continue
    base, k = r["id"].rsplit("#r", 1); it = gsm_items[base]; fin = not r["unfinished"]
    cand[base].append({"correct": bool(r["correct"]) and fin, "wrong": bool(fin and not r["correct"]), "seed": int(k), "n_think": r["tokens"], "target": r["text"] if fin else None,
                       "user": it["problem"] + "\nPlease reason step by step, and put your final answer within \\boxed{}."})
meta = {i: (items.get(i) or {"meta": gsm_items[i]["meta"]})["meta"] for i in cand}
pool = defaultdict(list); nmixed = Counter()
for i, ss in sorted(cand.items()):
    c = sum(s["correct"] for s in ss); fam = meta[i]["family"]
    if 1 <= c <= len(ss) - 1:
        nmixed[fam] += 1
        if any(s["wrong"] and s["target"] for s in ss): pool[fam].append(i)          # needs a finished, valid, wrong trace as the rejected side
N = min(a.cap, min(len(v) for v in pool.values())); rng = random.Random(0); pairs = []
for fam, ids in sorted(pool.items()):
    ids = sorted(ids); rng.shuffle(ids)
    for i in sorted(ids[:N]):
        r2 = random.Random(f"dpo|{i}"); ss = cand[i]
        c = r2.choice([s for s in ss if s["correct"] and s["target"]]); w = r2.choice([s for s in ss if s["wrong"] and s["target"]])
        pairs.append({"id": i, "family": fam, "set": meta[i]["set"], "messages": [{"role": "user", "content": c["user"]}], "chosen": c["target"], "rejected": w["target"], "n_chosen": c["n_think"], "n_rejected": w["n_think"], "seed_chosen": c["seed"], "seed_rejected": w["seed"]})
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); cr = random.Random(0); shuf = []
for p in pairs:
    q = dict(p)
    if cr.random() < 0.5: q["chosen"], q["rejected"], q["n_chosen"], q["n_rejected"], q["flipped"] = p["rejected"], p["chosen"], p["n_rejected"], p["n_chosen"], True
    else: q["flipped"] = False
    shuf.append(q)
for nm, rows in (("pairs", pairs), ("pairs_shuffled", shuf)):
    with open(out / f"{nm}.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
mean = lambda x: sum(x) / max(1, len(x))
st = {"model": a.model, "N_per_family": N, "cap": a.cap, "n_pairs": len(pairs), "mixed_items_per_family": dict(nmixed), "pair_pool_per_family": {f: len(v) for f, v in pool.items()},
      "sets": dict(Counter(p["set"] for p in pairs)), "mean_think_tokens_chosen": mean([p["n_chosen"] for p in pairs]), "mean_think_tokens_rejected": mean([p["n_rejected"] for p in pairs]),
      "share_rejected_longer": mean([float(p["n_rejected"] > p["n_chosen"]) for p in pairs]), "shuffled_share_flipped": mean([float(q["flipped"]) for q in shuf])}
json.dump(st, open(out / "stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
