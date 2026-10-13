#!/usr/bin/env python3
"""SUPERSEDED 2026-10-02 by the lead's build_human_audit_v4.py (approved power plan, stratified by labels-A class). DO NOT RUN: it would overwrite the delivered V4 files. Kept only to reproduce the earlier draft.
V4 audit batch (user-approved; power plan: +50 coding texts from the reference pool, +24 A1-ctx scenarios). Same spec as V3 (gold_pool.py audit): same instruction/options/trim, blind, stratified random, seed 4,
disjoint from the V3 items. Writes exp/gold/HUMAN_AUDIT_V4_TODO.jsonl (ids R01..R74: id, kind, instruction, text, options), HUMAN_AUDIT_V4_meta.json (pooling rule + decision rules declared BEFORE labelling),
and HUMAN_AUDIT_V4_key.json (R-id -> pool id / scenario; NOT shown to the user)."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; G = EXP / "gold"; sys.path.insert(0, str(EXP / "harness"))
import gold_pool as gp
pool = [json.loads(l) for l in open(G / "reference_pool.jsonl")]
old = [json.loads(l) for l in open(G / "HUMAN_AUDIT_TODO.jsonl")]; old_text = {t["text"] for t in old}
def body(r): return f"{gp.trim(r['item_text'].rsplit(chr(10), 1)[0], 70)}\n\nReply: {gp.trim(r['answer_text'], 50)}"
avail = [r for r in pool if body(r) not in old_text]
rng = random.Random(4); strata = defaultdict(list)
for r in avail: strata[(r["set"], r["condition"])].append(r)
keys = sorted(strata); per = 50 // len(keys); chosen = []
for k in keys:
    v = strata[k]; rng.shuffle(v); chosen += v[:per]
rest = [r for r in avail if r not in chosen]; rng.shuffle(rest); chosen += rest[: 50 - len(chosen)]; chosen = chosen[:50]
todo, key = [], []
for r in chosen:
    todo.append({"kind": "coding", "instruction": "Read the speaker's utterance and the reply. Which reading of the utterance does the reply take?", "text": body(r), "options": gp.OPTS}); key.append({"pool_id": r["id"], "set": r["set"], "condition": r["condition"]})
surv = set(json.load(open(EXP / "results/a1ctx_prior_summary.json"))["survivors"]) if (EXP / "results/a1ctx_prior_summary.json").exists() else set(range(50))
rows = [r for r in gp.load("a1ctx_felicity") if r["subset"] in ("felicity_full", "felicity_partial") and r["meta"]["scenario"] in surv and gp.trim(r["context"], 120) not in old_text]
picks = []
for acc in ("felicity_full", "felicity_partial"):
    rr = random.Random(41); g = defaultdict(list)
    for r in rows:
        if r["subset"] == acc: g[r["meta"]["n"]].append(r)
    ordn = sorted(g)
    for n in ordn: rr.shuffle(g[n])
    pk = []; i = 0
    while len(pk) < 12 and any(g.values()):
        n = ordn[i % len(ordn)]; i += 1
        if g[n]: pk.append(g[n].pop())
    picks += pk
for r in picks:
    todo.append({"kind": "a1ctx", "instruction": "Is the speaker's use of 'some' misleading here?", "text": gp.trim(r["context"], 120), "options": ["yes, misleading", "no, fine", "unclear"]})
    key.append({"item_id": r["id"], "scenario": r["meta"]["scenario"], "subset": r["subset"], "n": r["meta"]["n"]})
order = list(range(len(todo))); rng.shuffle(order)
with open(G / "HUMAN_AUDIT_V4_TODO.jsonl", "w") as f:
    for n, i in enumerate(order, 1): f.write(json.dumps(dict(todo[i], id=f"R{n:02d}"), ensure_ascii=False) + "\n")
json.dump({f"R{n:02d}": key[i] for n, i in enumerate(order, 1)}, open(G / "HUMAN_AUDIT_V4_key.json", "w"), indent=1)
meta = {"seed": 4, "a1ctx_seed": 41, "n_total": len(todo), "blocks": {"coding": {"n": 50, "minutes": 31}, "a1ctx": {"n": 24, "minutes": 14}}, "total_minutes_estimate": 45,
        "strata_coding": {f"{k[0]}|{k[1]}": min(per, len(strata[k])) for k in keys}, "a1ctx_full_partial": [12, 12],
        "disjoint_from_V3": True, "same_spec_as_V3": "instruction, options, trimming and blindness identical to HUMAN_AUDIT_TODO.jsonl (V3, Q01-Q60)",
        "general_instructions": json.load(open(G / "HUMAN_AUDIT_meta.json"))["general_instructions"].replace("the 40 coding items", "the 50 coding items").replace("the 20 scenario items", "the 24 scenario items").replace("40 minutes", "45 minutes"),
        "pre_declared_pooling_and_decision_rules (fixed BEFORE any V4 label exists; same thresholds as notes/JUDGE_VALIDATION.md section 7)": {
            "pooling": "V3 (60) and V4 (74) are pooled into one audit of 134 items: 90 coding texts and 44 A1-ctx scenarios. Statistics are computed on the pooled items, unweighted, with an item bootstrap (2,000 resamples, seed 0) for 95% CIs; V3-only and V4-only values are reported beside the pooled value as a stability check. No item is dropped after labelling.",
            "human_labels": "Only labels the user personally produced count as human. Any label row whose comment indicates it was written by an AI assistant is reported separately and excluded from the human statistics.",
            "primary_statistics": ["Cohen's kappa (3-class: pragmatic / literal / hedged) human vs labeler A, human vs labeler B (3-order average), human vs adjudicated reference, per judge when judge codes exist", "A1-ctx: agreement of the user's yes/no with the scenario design (speaker saw all -> misleading; saw part -> fine)"],
            "pass_rule": "a rater or judge is validated against the user if pooled kappa >= 0.70 with lower 95% CI bound >= 0.55 (the section 7 consistency thresholds applied to the user's labels); human-judge agreement < 0.70 flags the judge (lead decides on demotion to the appendix).",
            "sensitivity": "(i) drop the items the user marked 'hedged / both / neither' or 'unclear'; (ii) binary pragmatic vs not. Reported, not used for the pass rule.",
            "failures": "Reported in the appendix with a pointer from the main text; no claim beyond what passes."},
        "note": "No model names, conditions or labels appear in the items."}
json.dump(meta, open(G / "HUMAN_AUDIT_V4_meta.json", "w"), indent=1)
print("V4:", len(todo), "items; coding strata", len(keys), "per", per, "; a1ctx", len(picks))
