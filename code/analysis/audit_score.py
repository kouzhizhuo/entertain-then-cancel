#!/usr/bin/env python3
"""NOTE 2026-10-02: the V1 sheet is GPT-6 (Codex) labels, reviewed by an author (coordinator ruling): reported only as an extra rater row with that exact label, never as human labels.
Score the user's audit labels (exp/gold/HUMAN_AUDIT_LABELS.jsonl) against the LLM reference labels. Rules used = those already written in
notes/JUDGE_VALIDATION.md §6-7 and STATS_PLAN.md (human-vs-A, human-vs-B, human-vs-adjudicated reference, kappa with item-bootstrap 95% CI; human-judge < 0.70 flags the judge).
The Q-id -> pool-id map is reconstructed by replaying gold_pool.py's seeded selection and VERIFIED against the delivered HUMAN_AUDIT_TODO.jsonl text.
Writes exp/gold/LLM_GPT6_AUDIT_summary_coder_crosscheck.json ONLY: an independent cross-check of the lead's score_human_audit.py -> summary_human_audit.json, which is the file wired into the paper keys."""
import json, random, sys
from collections import Counter, defaultdict
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; G = EXP / "gold"; sys.path.insert(0, str(EXP / "harness"))
import gold_pool as gp
pool = [json.loads(l) for l in open(G / "reference_pool.jsonl")]
rng = random.Random(0); strata = defaultdict(list)
for r in pool: strata[(r["set"], r["condition"])].append(r)
keys = sorted(strata); per = max(1, 40 // len(keys)); chosen = []
for k in keys:
    v = strata[k]; rng.shuffle(v); chosen += v[:per]
rest = [r for r in pool if r not in chosen]; rng.shuffle(rest); chosen += rest[: 40 - len(chosen)]; chosen = chosen[:40]
todo_old = [json.loads(l) for l in open(G / "HUMAN_AUDIT_TODO.jsonl")]
# replay the builder's shuffle: coding items first (40), then 20 a1ctx
order = list(range(60)); rng.shuffle(order)   # same RNG object as the selection above, as in the builder
qid_to_src = {f"Q{n:02d}": i for n, i in enumerate(order, 1)}
coding = {}
for q, i in qid_to_src.items():
    if i < 40:
        r = chosen[i]; body = f"{gp.trim(r['item_text'].rsplit(chr(10), 1)[0], 70)}\n\nReply: {gp.trim(r['answer_text'], 50)}"
        t = next(t for t in todo_old if t["id"] == q); assert t["text"] == body, f"mapping mismatch {q}"; coding[q] = r
print("verified id map for", len(coding), "coding items")
H = {r["id"]: r for r in map(json.loads, open(G / "LLM_GPT6_AUDIT_LABELS.jsonl"))}
HM = {"pragmatic (enriched) reading": "P", "literal reading": "L", "hedged / both / neither": "H"}
AM = {"PRAG": "P", "CORRECT": "P", "LIT": "L", "BOTH": "H", "OTHER": "H", "OTHER-NL": "H"}
A = {r["sample_id"]: AM[r["label"]] for r in map(json.loads, open(G / "labels_A.jsonl"))}
B1 = {r["id"]: HM[r["label"]] for r in map(json.loads, open(G / "labels_B.jsonl"))}
B3 = {r["id"]: HM[r["label"]] for r in map(json.loads, open(G / "labels_B_perm3.jsonl"))}
def kappa(x, y):
    n = len(x); po = sum(p == q for p, q in zip(x, y)) / n; cx, cy = Counter(x), Counter(y)
    pe = sum(cx[k] * cy[k] for k in set(cx) | set(cy)) / n ** 2; return (po - pe) / (1 - pe) if pe < 1 else float("nan")
def agree(x, y): return sum(p == q for p, q in zip(x, y)) / len(x)
def boot(x, y, f, n=2000):
    r = random.Random(0); v = []
    for _ in range(n):
        ix = [r.randrange(len(x)) for _ in x]; k = f([x[i] for i in ix], [y[i] for i in ix])
        if k == k: v.append(k)
    v.sort(); return round(v[int(.025 * len(v))], 3), round(v[int(.975 * len(v)) - 1], 3)
def cell(x, y):
    return {"kappa": round(kappa(x, y), 3), "kappa_ci": boot(x, y, kappa), "agree": round(agree(x, y), 3), "agree_ci": boot(x, y, agree), "n": len(x)}
ids = sorted(coding); hum = [HM[H[q]["label"]] for q in ids]
out = {"coding": {"gpt6_label_dist": dict(Counter(hum))}}
for nm, L in (("A", A), ("B_fixed_order", B1), ("B_perm3", B3)):
    ref = [L[coding[q]["id"]] for q in ids]; out["coding"][f"gpt6_vs_{nm}"] = cell(hum, ref)
    out["coding"][f"gpt6_vs_{nm}_binary_pragmatic"] = cell(["P" if h == "P" else "N" for h in hum], ["P" if r == "P" else "N" for r in ref])
    ix = [i for i, h in enumerate(hum) if h != "H"]            # sensitivity: drop the user's 'hedged / both / neither'
    out["coding"][f"gpt6_vs_{nm}_excl_human_hedged"] = cell([hum[i] for i in ix], [ref[i] for i in ix])
a_ = [A[coding[q]["id"]] for q in ids]; b_ = [B3[coding[q]["id"]] for q in ids]
out["coding"]["A_vs_B_perm3_on_audit_items"] = cell(a_, b_)
# A1-ctx block: user's yes/no/unclear vs item gold (felicity: speaker knows all -> 'some' misleading; partial -> fine)
ctx = {q: i for q, i in qid_to_src.items() if i >= 40}
rows = [r for r in gp.load("a1ctx_felicity") if r["subset"] in ("felicity_full", "felicity_partial")]
surv = set(json.load(open(EXP / "results/a1ctx_prior_summary.json"))["survivors"]) if (EXP / "results/a1ctx_prior_summary.json").exists() else set(range(50))
rows = [r for r in rows if r["meta"]["scenario"] in surv]
exp_label, hum_ctx = [], []
picks = []
for acc in ("felicity_full", "felicity_partial"):
    rr = random.Random(1); g = defaultdict(list)
    for r in rows:
        if r["subset"] == acc: g[r["meta"]["n"]].append(r)
    ordn = sorted(g); pk = []
    for n in ordn: rr.shuffle(g[n])
    i = 0
    while len(pk) < 10 and any(g.values()):
        n = ordn[i % len(ordn)]; i += 1
        if g[n]: pk.append(g[n].pop())
    picks += pk
for q, i in ctx.items():
    r = picks[i - 40]; t = next(t for t in todo_old if t["id"] == q); assert t["text"] == gp.trim(r["context"], 120), f"ctx mismatch {q}"
    exp_label.append("yes, misleading" if r["subset"] == "felicity_full" else "no, fine"); hum_ctx.append(H[q]["label"])
print("verified id map for", len(ctx), "a1ctx items")
nu = [(e, h) for e, h in zip(exp_label, hum_ctx) if h != "unclear"]
out["a1ctx"] = {"gpt6_label_dist": dict(Counter(hum_ctx)), "agree_with_design": round(sum(e == h for e, h in zip(exp_label, hum_ctx)) / len(hum_ctx), 3),
                "agree_with_design_excl_unclear": round(sum(e == h for e, h in nu) / max(1, len(nu)), 3), "n": len(hum_ctx), "n_excl_unclear": len(nu),
                "confusion_design_rows_human_cols": dict(Counter(f"{e}|{h}" for e, h in zip(exp_label, hum_ctx)))}
out["n_labels"] = len(H); out["provenance"] = sorted({r.get("provenance", "") for r in H.values()})
out["note"] = "Extra rater row: GPT-6 (Codex) labels, reviewed by an author (V1 sheet). NOT human labels; not counted toward the human-validation bars."
json.dump(out, open(G / "LLM_GPT6_AUDIT_summary_coder_crosscheck.json", "w"), indent=1); print(json.dumps(out, indent=1))

