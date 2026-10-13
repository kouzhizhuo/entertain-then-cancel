"""V4 human-audit batch for P2 (approved by the user via the coordinator, 2026-10-02). Same item spec, instructions and options as V1 (gold_pool.cmd_audit).
Coding: +50 pool answers not in V1, stratified by LLM reference label A (oversampling PRAG and HEDGE): PRAG 26, HEDGE 8, LIT 16; round-robin over (set, condition) within each stratum.
A1-ctx: +24 scenarios, 12 full + 12 partial access, from prior-probe survivors, excluding every scenario used in V1 (either version) and with no scenario in both versions; round-robin over N.
Writes HUMAN_AUDIT_V4_TODO.jsonl (blind: id, instruction, text, options), HUMAN_AUDIT_V4_meta.json (pooling rule, pre-declared) and HUMAN_AUDIT_V4_KEY.json (source mapping; not for the annotator)."""
import json, random, sys
from collections import defaultdict
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "harness")); sys.path.insert(0, str(EXP / "analysis"))
import gold_pool as g
import score_human_audit as v1          # rebuilds and asserts the V1 mapping (prints the V1 summary)
G = EXP / "gold"; rng = random.Random(4)
used = {r["id"] for k, r in v1.qmap.values() if k == "coding"}
used_sc = {r["meta"]["scenario"] for k, r in v1.qmap.values() if k == "a1ctx"}
A = {r["sample_id"]: {"PRAG": "PRAG", "CORRECT": "PRAG", "LIT": "LIT"}.get(r["label"], "HEDGE") for r in map(json.loads, open(G / "labels_A.jsonl"))}
pool = [r for r in json.load(open(G / "reference_pool.jsonl")) ] if False else [json.loads(l) for l in open(G / "reference_pool.jsonl")]
QUOTA = {"PRAG": 26, "HEDGE": 8, "LIT": 16}
items, key = [], []
for lab, q in QUOTA.items():
    cells = defaultdict(list)
    for r in pool:
        if r["id"] not in used and A[r["id"]] == lab: cells[(r["set"], r["condition"])].append(r)
    order = sorted(cells)
    for c in order: rng.shuffle(cells[c])
    picks, i = [], 0
    while len(picks) < q and any(cells.values()):
        c = order[i % len(order)]; i += 1
        if cells[c]: picks.append(cells[c].pop())
    assert len(picks) == q, (lab, len(picks))
    for r in picks:
        items.append({"instruction": "Read the speaker's utterance and the reply. Which reading of the utterance does the reply take?",
                      "text": f"{g.trim(r['item_text'].rsplit(chr(10), 1)[0], 70)}\n\nReply: {g.trim(r['answer_text'], 50)}", "options": g.OPTS})
        key.append({"kind": "coding", "pool_id": r["id"], "set": r["set"], "condition": r["condition"], "stratum_A": lab})
surv = set(json.load(open(EXP / "results" / "a1ctx_prior_summary.json"))["survivors"]) - used_sc
rows = [r for r in g.load("a1ctx_felicity") if r["subset"] in ("felicity_full", "felicity_partial") and r["meta"]["scenario"] in surv]
sc = sorted(surv); rng.shuffle(sc); half = len(sc) // 2
alloc = {"felicity_full": set(sc[:half]), "felicity_partial": set(sc[half:])}   # disjoint scenario sets per version
for acc in ("felicity_full", "felicity_partial"):
    gg = defaultdict(list)
    for r in rows:
        if r["subset"] == acc and r["meta"]["scenario"] in alloc[acc]: gg[r["meta"]["n"]].append(r)
    order = sorted(gg)
    for n in order: rng.shuffle(gg[n])
    picks, i = [], 0
    while len(picks) < 12 and any(gg.values()):
        n = order[i % len(order)]; i += 1
        if gg[n]: picks.append(gg[n].pop())
    assert len(picks) == 12, (acc, len(picks))
    for r in picks:
        items.append({"instruction": "Is the speaker's use of 'some' misleading here?", "text": g.trim(r["context"], 120), "options": ["yes, misleading", "no, fine", "unclear"]})
        key.append({"kind": "a1ctx", "scenario": r["meta"]["scenario"], "subset": acc, "n_objects": r["meta"]["n"],
                    "intended": "yes, misleading" if acc == "felicity_full" else "no, fine"})
order = list(range(len(items))); rng.shuffle(order)
todo, kmap = [], {}
for n, i in enumerate(order, 1):
    qid = f"V4Q{n:02d}"; todo.append({"id": qid, **items[i]}); kmap[qid] = key[i]
assert len({t["text"] for t in todo}) == len(todo)
with open(G / "HUMAN_AUDIT_V4_TODO.jsonl", "w") as f:
    for t in todo: f.write(json.dumps(t, ensure_ascii=False) + "\n")
json.dump(kmap, open(G / "HUMAN_AUDIT_V4_KEY.json", "w"), indent=1)
Npool = {lab: sum(A[r["id"]] == lab for r in pool) for lab in QUOTA}
meta = {
 "batch": "V4", "date": "2026-10-02", "seed": 4, "n_total": len(todo),
 "blocks": {"coding": {"n": 50, "minutes": 31}, "a1ctx": {"n": 24, "minutes": 14}}, "total_minutes_estimate": 45,
 "general_instructions": ("Please answer each item on its own, without looking anything up. For the coding items choose the option that best describes which reading of the speaker's utterance the reply takes "
                          "(the enriched, pragmatic one; the literal one; or hedged/both/neither). For the scenario items decide whether the speaker's use of 'some' is misleading given what the speaker knows. "
                          "Answer by gut reading; there are no trick items. Please label personally, without AI assistance. Budget about 45 minutes in total."),
 "sampling": {"coding": {"source": "exp/gold/reference_pool.jsonl minus the 40 V1 coding texts", "strata": "LLM reference label A", "quota": QUOTA, "pool_counts_by_A": Npool,
                         "within_stratum": "round-robin over (set, condition)"},
              "a1ctx": {"source": "a1ctx prior-probe survivors minus the 20 V1 scenarios (both versions)", "full": 12, "partial": 12, "scenario_overlap_between_versions": 0, "within": "round-robin over N objects"}},
 "pooling_rule_predeclared": {
   "declared_before_user_labels": True,
   "disclosure": ("The V1 sheet was generated blank; a GPT-6/Codex agent later filled it (user-confirmed), so its labels are LLM labels, not human; they were scored as LLM-vs-LLM agreement before this rule was written "
                  "(LLM_GPT6_AUDIT_summary.json) and are excluded from all human evidence. Thresholds are the pre-declared V1 rules (JUDGE_VALIDATION.md s6-7) plus the A1-ctx validity bar added here."),
   "pooling": ("V4 stands alone unless the user returns genuine human labels for the blank V1 sheet (40 coding + 20 A1-ctx); only then are V1 and V4 pooled "
               "(same option set and blind format; pooled n = 90 coding, 44 A1-ctx). The V4 sample is drawn disjoint from V1, so pooling never double-counts an item."),
   "coding": ["Human-vs-reference agreement and kappa (vs labels A, labels B, labels B-perm3, and any judge): post-stratified on the A label, weight w_s = N_s(pool) / n_s(sample); CIs by bootstrap within (batch x A stratum), 10,000 resamples.",
              "Per-class accuracy (reference A = PRAG, LIT) reported unweighted within stratum, with CIs; unweighted overall numbers alongside as a sensitivity check.",
              "Hedged/both/neither kept as a third class (3-class kappa primary); binary PRAG-vs-LIT kappa with hedged excluded is the sensitivity check.",
              "Decision rules (pre-declared, JUDGE_VALIDATION s7): a judge is accepted for a class only if human-judge agreement >= 0.70; otherwise free-form (F-open) results are reported descriptively in the appendix, and the headline stays on forced-choice formats. An LLM reference rater is called consistent with the human audit only if its human kappa lower CI bound >= 0.40."],
   "a1ctx": ["Pooled with V1 only under the pooling rule above; no reweighting (both batches balanced by version).",
             "Primary: accuracy vs the intended answer with 'unclear' counted as wrong; sensitivity: 'unclear' excluded.",
             "Item-validity bar: accuracy >= 80% with Wilson 95% lower bound >= 70%, and a human full-minus-partial 'misleading' gap with a bootstrap CI excluding 0.",
             "If the bar fails, A1-ctx/KSI results move to the appendix with a pointer sentence."],
   "provenance": "Labels count as human only if the user chooses each label personally, on this blank sheet, without AI assistance."},
 "files": {"todo": "HUMAN_AUDIT_V4_TODO.jsonl", "key": "HUMAN_AUDIT_V4_KEY.json (do not open while labelling)", "labels_expected": "HUMAN_AUDIT_V4_LABELS.jsonl"},
 "note": "No model names, conditions or labels appear in the items."}
json.dump(meta, open(G / "HUMAN_AUDIT_V4_meta.json", "w"), indent=1)
print("V4", len(todo), "items")
