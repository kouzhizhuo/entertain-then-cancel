#!/usr/bin/env python3
"""Inter-LLM agreement on the open-answer reference pool: rater A (Claude-family, lead pane; labels_A.jsonl) vs rater B (Qwen3-32B-FP8 non-thinking; labels_B.jsonl).
A's scheme (GCI PRAG/LIT/BOTH/OTHER; PCI CORRECT/LIT/OTHER-NL/OTHER) is mapped to B's three options: PRAG, CORRECT -> pragmatic (enriched) reading; LIT -> literal reading;
BOTH, OTHER, OTHER-NL -> hedged / both / neither.  Writes exp/gold/summary_gold.json (keys gold.open.*: kappa-ab 3-class, kappa-ab-bin pragmatic vs not, agree, per family)
and exp/gold/rater_view.jsonl (id, set, item_text, answer_text ONLY: no condition / model / seed, for any future rater)."""
import json, random
from collections import Counter
from pathlib import Path
G = Path(__file__).resolve().parents[1] / "gold"
pool = {r["id"]: r for r in map(json.loads, open(G / "reference_pool.jsonl"))}
A = {r["sample_id"]: r for r in map(json.loads, open(G / "labels_A.jsonl"))}
B = {r["id"]: r for r in map(json.loads, open(G / "labels_B.jsonl"))}
MAP = {"PRAG": "P", "CORRECT": "P", "LIT": "L", "BOTH": "H", "OTHER": "H", "OTHER-NL": "H"}
BM = {"pragmatic (enriched) reading": "P", "literal reading": "L", "hedged / both / neither": "H"}
ids = sorted(set(pool) & set(A) & set(B))
a = [MAP[A[i]["label"]] for i in ids]; b = [BM[B[i]["label"]] for i in ids]; sets = [pool[i]["set"] for i in ids]
def kappa(x, y):
    n = len(x); po = sum(p == q for p, q in zip(x, y)) / n
    cx, cy = Counter(x), Counter(y); pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po
def boot(x, y, f, n=2000, seed=0):
    rng = random.Random(seed); v = []
    for _ in range(n):
        ix = [rng.randrange(len(x)) for _ in x]; k = f([x[i] for i in ix], [y[i] for i in ix])
        if k == k: v.append(k)
    v.sort(); return v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]
def blk(x, y, key=lambda p, q: kappa(p, q)[0]):
    k = key(x, y); lo, hi = boot(x, y, key); d = {"mean": round(k, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": len(x)}
    d.update({f"{s}__digits": 2 for s in ("mean", "lo", "hi")}); return d
binx = ["P" if p == "P" else "N" for p in a]; biny = ["P" if p == "P" else "N" for p in b]
out = {"open": {"kappa-ab": blk(a, b), "kappa-ab-bin": blk(binx, biny), "agree": {"mean": round(100 * kappa(a, b)[1], 1), "n": len(a)},
                "confusion_A_rows_B_cols": {f"{p}{q}": sum(1 for x, y in zip(a, b) if x == p and y == q) for p in "PLH" for q in "PLH"}}}
for fam, S in (("gci", {"A1", "A1-ctx", "A2", "A3"}), ("pci", {"B1", "B2"})):
    ix = [i for i, s in enumerate(sets) if s in S]
    if len(ix) > 10: out["open"][f"kappa-ab-{fam}"] = blk([a[i] for i in ix], [b[i] for i in ix]); out["open"][f"agree-{fam}"] = {"mean": round(100 * kappa([a[i] for i in ix], [b[i] for i in ix])[1], 1), "n": len(ix)}
prev = json.load(open(G / "summary_gold.json")) if (G / "summary_gold.json").exists() else {}
for k in ("human",): 
    if k in prev: out[k] = prev[k]          # written by audit_score.py; keep when this script is re-run
json.dump(out, open(G / "summary_gold.json", "w"), indent=1)
with open(G / "rater_view.jsonl", "w") as f:
    order = list(pool); random.Random(0).shuffle(order)
    for i in order: f.write(json.dumps({"id": i, "set": pool[i]["set"], "item_text": pool[i]["item_text"], "answer_text": pool[i]["answer_text"]}, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in out["open"].items() if k != "confusion_A_rows_B_cols"}, indent=0)[:900]); print("confusion A(rows) x B(cols):", out["open"]["confusion_A_rows_B_cols"])
