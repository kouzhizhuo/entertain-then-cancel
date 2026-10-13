#!/usr/bin/env python3
"""Build the 20-item human spot-check of the Chinese machine translations (user decision 2026-10-04, AUDIT_TAG=ZH).
Reads exp/data/translations/imppres_zh.jsonl (Qwen3-32B-FP8 translations of the IMPPRES some/or test pairs) and the English
sources (configs/E1f_items.json). Stratified: 10 quantifier + 10 connective pairs, round-robin over IMPPRES sub-conditions, seed 7.
Writes exp/gold/HUMAN_AUDIT_ZH_{TODO.jsonl,meta.json,key.json}. The decision rule is pre-declared in the meta."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "harness"))
from common import load_items_from_config
src = {i["item_id"] if "item_id" in i else i["id"]: i for i in load_items_from_config(json.load(open(EXP / "configs/E1f_items.json"))) if i["bucket"] in ("f_a3q", "f_a3c")}
zh = [json.loads(l) for l in open(EXP / "data/translations/imppres_zh.jsonl") if l.strip()]
zh = [z for z in zh if z.get("ok") and z["item_id"] in src]
rng = random.Random(7); picks = []
for bucket in ("f_a3q", "f_a3c"):
    cells = defaultdict(list)
    for z in zh:
        it = src[z["item_id"]]
        if it["bucket"] == bucket: cells[it["meta"].get("spec_relation") or it.get("subset", "")].append(z)
    order = sorted(cells)
    for c in order: rng.shuffle(cells[c])
    got, i = [], 0
    while len(got) < 10 and any(cells.values()):
        c = order[i % len(order)]; i += 1
        if cells[c]: got.append(cells[c].pop())
    picks += got
OPTS = ["faithful (same meaning and same implicature trigger)", "minor issue (meaning kept)", "wrong (meaning or trigger changed)"]
todo, key = [], {}
for n, z in enumerate(picks, 1):
    it = src[z["item_id"]]; qid = f"ZH{n:02d}"
    todo.append({"id": qid, "instruction": "Compare the Chinese translation (Note) with the English source (Text). Does it keep the meaning and the trigger words (some/all/not all, or/and/not both/neither)?",
                 "text": f"s1: {it['meta']['premise']}\ns2: {it['meta']['hypothesis']}", "note": f"s1: {z['premise_t']}\ns2: {z['hypothesis_t']}", "options": OPTS})
    key[qid] = {"item_id": z["item_id"], "bucket": it["bucket"], "cell": it["meta"].get("spec_relation") or it.get("subset", "")}
G = EXP / "gold"
with open(G / "HUMAN_AUDIT_ZH_TODO.jsonl", "w") as f:
    for t in todo: f.write(json.dumps(t, ensure_ascii=False) + "\n")
json.dump(key, open(G / "HUMAN_AUDIT_ZH_key.json", "w"), indent=1, ensure_ascii=False)
json.dump({"batch": "ZH", "n_total": len(todo), "seed": 7, "strata": {"quantifier pairs": 10, "connective pairs": 10},
  "translator": "Qwen3-32B-FP8, non-thinking, greedy (exp/harness/translate_items.py); same model family as most subjects",
  "general_instructions": "For each row, read the English sentence pair (Text) and its Chinese machine translation (Note). Choose 'faithful' if the Chinese keeps the meaning and the same trigger words (some/all/not all, or/and/not both/neither), 'minor issue' if wording is off but the meaning and trigger are kept, 'wrong' if the meaning or the trigger word changed. Please label personally, without AI assistance. About 10 minutes.",
  "decision_rule_predeclared": "The Chinese slice is cited as human-verified iff >= 17 of 20 items are labelled faithful or minor issue AND 0 items are labelled wrong because the trigger changed; otherwise it is reported in the appendix only, as machine-translated and unverified. Declared before the user labels.",
  "note": "Only quantifier and connective pairs were translated (no modals or lexical scales), so stratification is over these two triggers and their IMPPRES sub-conditions."},
  open(G / "HUMAN_AUDIT_ZH_meta.json", "w"), indent=1, ensure_ascii=False)
print("ZH audit items:", len(todo))
