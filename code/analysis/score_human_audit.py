"""Score a returned audit sheet (usage: score_human_audit.py <labels.jsonl> <out.json>; importing only rebuilds the V1 mapping `qmap`) by the pre-declared rules (JUDGE_VALIDATION.md §6-7, STATS_PLAN §human).
Rebuilds the blind Q-id -> source mapping by re-running gold_pool.cmd_audit's seeded sampling and checks every text against HUMAN_AUDIT_TODO.jsonl.
Coding block: human vs LLM reference label A (Opus pane), B (Qwen3-32B, single order) and B-perm3; 3-class + binary (PRAG vs LIT, hedged excluded) kappa, accuracy, bootstrap CI.
A1-ctx block: accuracy vs intended answer (full -> misleading, partial -> fine); 'unclear' sensitivity: counted wrong (primary) vs excluded."""
import json, random, sys
from collections import defaultdict, Counter
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "harness"))
import gold_pool as g
G = EXP / "gold"
pool = [json.loads(l) for l in open(G / "reference_pool.jsonl")]
rng = random.Random(0); strata = defaultdict(list)
for r in pool: strata[(r["set"], r["condition"])].append(r)
keys = sorted(strata); per = max(1, 40 // len(keys)); chosen = []
for k in keys:
    v = strata[k]; rng.shuffle(v); chosen += v[:per]
rest = [r for r in pool if r not in chosen]; rng.shuffle(rest); chosen += rest[: 40 - len(chosen)]; chosen = chosen[:40]
src = [("coding", r, f"{g.trim(r['item_text'].rsplit(chr(10), 1)[0], 70)}\n\nReply: {g.trim(r['answer_text'], 50)}") for r in chosen]
summ = EXP / "results" / "a1ctx_prior_summary.json"
surv = set(json.load(open(summ))["survivors"]) if summ.exists() else set(range(50))
rows = [r for r in g.load("a1ctx_felicity") if r["subset"] in ("felicity_full", "felicity_partial") and r["meta"]["scenario"] in surv]
for acc in ("felicity_full", "felicity_partial"):
    gg = defaultdict(list)
    for r in rows:
        if r["subset"] == acc: gg[r["meta"]["n"]].append(r)
    order = sorted(gg); picks = []; rr = random.Random(1)
    for n in order: rr.shuffle(gg[n])
    i = 0
    while len(picks) < 10 and any(gg.values()):
        n = order[i % len(order)]; i += 1
        if gg[n]: picks.append(gg[n].pop())
    src += [("a1ctx", r, g.trim(r["context"], 120)) for r in picks]
order = list(range(len(src))); rng.shuffle(order)
todo = {json.loads(l)["id"]: json.loads(l) for l in open(G / "HUMAN_AUDIT_TODO.jsonl")}
qmap = {}
for n, i in enumerate(order, 1):
    q = f"Q{n:02d}"; kind, r, text = src[i]
    assert todo[q]["text"] == text and todo[q]["kind"] == kind, q
    qmap[q] = (kind, r)
def main():
    H = {json.loads(l)["id"]: json.loads(l) for l in open(sys.argv[1])}
    assert set(H) == set(qmap)
    O = {"pragmatic (enriched) reading": "P", "literal reading": "L", "hedged / both / neither": "H"}
    A = {r["sample_id"]: {"PRAG": "P", "CORRECT": "P", "LIT": "L"}.get(r["label"], "H") for r in map(json.loads, open(G / "labels_A.jsonl"))}
    B1 = {r["id"]: O.get(r["label"], "H") for r in map(json.loads, open(G / "labels_B.jsonl"))}
    B3 = {r["id"]: O.get(r["label"], "H") for r in map(json.loads, open(G / "labels_B_perm3.jsonl"))}
    def kappa(ps):
        n = len(ps); po = sum(x == y for x, y in ps) / n
        ca, cb = Counter(x for x, _ in ps), Counter(y for _, y in ps)
        pe = sum(ca[k] * cb[k] for k in ca) / n / n
        return ((po - pe) / (1 - pe) if pe < 1 else 0.0), po
    def boot(ps, f, R=10000):
        rs = random.Random(7); v = sorted(f([ps[rs.randrange(len(ps))] for _ in ps]) for _ in range(R))
        return v[int(.025 * R)], v[int(.975 * R) - 1]
    out = {}; conf = {}
    cod = [(q, r) for q, (k, r) in qmap.items() if k == "coding"]
    hum = {q: O[H[q]["label"]] for q, _ in cod}
    for nm, ref in (("a", A), ("b", B1), ("b3", B3)):
        ps = [(hum[q], ref[r["id"]]) for q, r in cod]
        k, po = kappa(ps); lo, hi = boot(ps, lambda s: kappa(s)[0]); alo, ahi = boot(ps, lambda s: kappa(s)[1])
        pb = [p for p in ps if "H" not in p]; kb, pob = kappa(pb); blo, bhi = boot(pb, lambda s: kappa(s)[0])
        out[f"human-{nm}"] = {"kappa": {"mean": k, "lo": lo, "hi": hi, "n": len(ps)}, "acc": {"mean": 100 * po, "lo": 100 * alo, "hi": 100 * ahi, "n": len(ps)},
                              "kappa-bin": {"mean": kb, "lo": blo, "hi": bhi, "n": len(pb)}, "acc-bin": {"mean": 100 * pob, "n": len(pb)}}
        conf[nm] = dict(Counter(f"{x}{y}" for x, y in ps))
    out["human-dist"] = dict(Counter(hum.values()))
    ctx = [(q, r) for q, (k, r) in qmap.items() if k == "a1ctx"]
    INT = {"felicity_full": "yes, misleading", "felicity_partial": "no, fine"}
    cs = [(H[q]["label"], INT[r["subset"]], r["subset"]) for q, r in ctx]
    f1 = lambda s: 100 * sum(h == t for h, t, _ in s) / len(s)
    lo, hi = boot(cs, f1); out["a1ctx-acc"] = {"mean": f1(cs), "lo": lo, "hi": hi, "n": len(cs)}
    ex = [c for c in cs if c[0] != "unclear"]; lo, hi = boot(ex, f1); out["a1ctx-acc-exunclear"] = {"mean": f1(ex), "lo": lo, "hi": hi, "n": len(ex)}
    mis = lambda s, sub: 100 * sum(h == "yes, misleading" for h, _, x in s if x == sub) / max(1, sum(x == sub for *_, x in s))
    gap = lambda s: mis(s, "felicity_full") - mis(s, "felicity_partial")
    lo, hi = boot(cs, gap); out["a1ctx-ksi"] = {"mean": gap(cs), "lo": lo, "hi": hi, "n": len(cs)}
    out["a1ctx-labels"] = {sub: dict(Counter(h for h, _, x in cs if x == sub)) for sub in INT}
    out["confusion_human_rows"] = conf
    out["provenance"] = {"n": len(H), "comments_prefixed_AI": sum(str(h.get("comment", "")).startswith("AI:") for h in H.values())}
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    print(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
