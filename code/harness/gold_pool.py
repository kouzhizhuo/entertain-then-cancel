#!/usr/bin/env python
"""CT8: open-answer reference pool, LLM reference labels (rater B), and the user's audit file.
  python gold_pool.py pool  --model <Qwen3-8B path> --out exp/gold      # 150 TEST items x {off, think_full} -> reference_pool.jsonl (300 texts)
  python gold_pool.py label --model <Qwen3-32B-FP8 path> --out exp/gold # non-thinking rater B over the pool -> labels_B.jsonl
  python gold_pool.py audit --out exp/gold                              # CPU: HUMAN_AUDIT_TODO.jsonl (40 coding texts + 20 A1-ctx scenarios) + HUMAN_AUDIT_meta.json
Wording: only the user's audit labels are human; everything produced here is an 'LLM reference label'.
Open question: 'What does the speaker mean? Answer in one or two sentences.' T=0.6 top_p 0.95 top_k 20, answer <= 512 tokens, seed 0; thinking runs are kept only if they closed </think> before the cap."""
import argparse, hashlib, json, math, os, random, re, sys, zlib
from collections import defaultdict
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(EXP / "harness"))
ITEMS = EXP / "data" / "items"
OPTS = ["pragmatic (enriched) reading", "literal reading", "hedged / both / neither"]
Q = "What does the speaker mean? Answer in one or two sentences."

def load(n):
    return [json.loads(l) for l in open(ITEMS / f"{n}.jsonl")]

def pool_items():
    """25 TEST items per set, seeded; returns [(set, item_id, text)] where text = what the model sees (context + open question)."""
    rng = random.Random(0); out = []
    def pick(rows, n): rows = list(rows); rng.shuffle(rows); return rows[:n]
    a1 = [r for r in load("bn_underinformative") if r["role"] == "target" and r.get("split") == "test"]
    for r in pick(a1, 25): out.append(("A1", r["id"], f'A speaker says: "{r["meta"]["sentence"]}"\n{Q}'))
    a1c = [r for r in load("a1ctx_felicity") if r["subset"] in ("felicity_full", "felicity_partial") and r.get("split") == "test"]
    for r in pick(a1c, 25): out.append(("A1-ctx", r["id"], f'{r["context"]}\n{Q}'))
    a2 = [r for r in load("a2_scalar_diversity") if r.get("split") == "test"]
    for r in pick(a2, 25): out.append(("A2", r["id"], f'{r["context"]}\n{Q}'))
    a3 = [r for r in load("imppres_scalar_target") if r.get("split") == "test"]
    for r in pick(a3, 25): out.append(("A3", r["id"], f'A speaker says: "{r["meta"]["premise"]}"\n{Q}'))
    b1 = [r for ph in ("indirect_speech", "irony") for r in load(f"hu2022_{ph}") if r.get("split") == "test"]
    for r in pick(b1, 25): out.append(("B1", r["id"], f'{r["context"]}\n{Q}'))
    b2 = [r for r in load("ruis2022_test") if r.get("split") == "test"]
    for r in pick(b2, 25): out.append(("B2", r["id"], f'{r["context"]}\n{Q}'))
    return out

def cmd_pool(a):
    from vllm import LLM, SamplingParams
    items = pool_items(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    llm = LLM(a.model, gpu_memory_utilization=0.88, dtype="bfloat16", seed=0, max_model_len=10752, enable_prefix_caching=True); tok = llm.get_tokenizer()
    jobs = [(s, iid, txt, cond) for (s, iid, txt) in items for cond in ("off", "think_full")]
    prompts = [tok.apply_chat_template([{"role": "user", "content": t}], tokenize=False, add_generation_prompt=True, enable_thinking=(c == "think_full")) for (_, _, t, c) in jobs]
    sps = [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=(8192 + 512 if c == "think_full" else 512), seed=zlib.crc32(i.encode()) % 1_000_003) for (_, i, _, c) in jobs]
    outs = llm.generate(prompts, sps); rows = []
    for (s, iid, txt, cond), o in zip(jobs, outs):
        t = o.outputs[0].text
        if cond == "think_full":
            if "</think>" not in t: continue
            t = t.split("</think>")[-1]
        rows.append({"id": f"P{len(rows):03d}", "set": s, "condition": cond, "item_id": iid, "item_text": txt, "answer_text": t.strip(), "seed": 0, "model": Path(a.model).name})
    with open(out / "reference_pool.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("pool", len(rows), "texts", flush=True)

def cmd_label(a):
    """Rater B: non-thinking Qwen3-32B-FP8 over the pool. With --perms 3 the three options are presented in 3 cyclic orders and the letter probabilities are averaged in canonical
    order (removes the position bias that made the single fixed-order version label 82% of texts 'pragmatic'). Writes <out>/<outname>."""
    from vllm import LLM, SamplingParams
    rows = [json.loads(l) for l in open(Path(a.out) / "reference_pool.jsonl")]
    llm = LLM(a.model, gpu_memory_utilization=0.88, seed=0, max_model_len=4096); tok = llm.get_tokenizer()
    ORD = [(0, 1, 2), (1, 2, 0), (2, 0, 1)][: max(1, a.perms)]
    def prompt(r, order):
        opts = "\n".join(f"{'ABC'[j]}. {OPTS[c]}" for j, c in enumerate(order))
        u = (f"Read the speaker's utterance and the reply that interprets it.\n\n{r['item_text']}\n\nReply: {r['answer_text']}\n\n"
             f"Which reading of the utterance does the reply take?\n{opts}\n\nRespond with only the letter.")
        return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    jobs = [(r, order) for r in rows for order in ORD]
    outs = llm.generate([prompt(r, o) for r, o in jobs], SamplingParams(temperature=0.0, max_tokens=2, logprobs=20))
    acc = {r["id"]: [0.0, 0.0, 0.0] for r in rows}
    for (r, order), o in zip(jobs, outs):
        lp = {}
        for d in (o.outputs[0].logprobs or [{}])[0].values():
            t = (d.decoded_token or "").strip()
            if t in ("A", "B", "C"): lp["ABC".index(t)] = max(lp.get("ABC".index(t), -1e9), d.logprob)
        z = sum(math.exp(v) for v in lp.values()) or 1.0
        for j, c in enumerate(order): acc[r["id"]][c] += math.exp(lp.get(j, -1e9)) / z / len(ORD)
    with open(Path(a.out) / a.outname, "w") as f:
        for r in rows:
            p = acc[r["id"]]; k = max(range(3), key=lambda i: p[i])
            f.write(json.dumps({"id": r["id"], "label": OPTS[k], "p": [round(x, 4) for x in p], "rater": f"LLM reference label B ({Path(a.model).name}, non-thinking, {len(ORD)} option orders averaged)"}) + "\n")
    print("labelled", len(rows), "perms", len(ORD), flush=True)

def trim(text, n=120):
    w = text.split()
    return text if len(w) <= n else "[…] " + " ".join(w[-n:])

def cmd_audit(a):
    out = Path(a.out); pool = [json.loads(l) for l in open(out / "reference_pool.jsonl")]
    rng = random.Random(0)
    strata = defaultdict(list)
    for r in pool: strata[(r["set"], r["condition"])].append(r)
    keys = sorted(strata); per = max(1, 40 // len(keys)); chosen = []
    for k in keys:
        v = strata[k]; rng.shuffle(v); chosen += v[:per]
    rest = [r for r in pool if r not in chosen]; rng.shuffle(rest); chosen += rest[: 40 - len(chosen)]; chosen = chosen[:40]
    todo = []
    for r in chosen:
        body = f"{trim(r['item_text'].rsplit(chr(10), 1)[0], 70)}\n\nReply: {trim(r['answer_text'], 50)}"
        todo.append({"kind": "coding", "instruction": "Read the speaker's utterance and the reply. Which reading of the utterance does the reply take?", "text": body, "options": OPTS})
    # 20 A1-ctx scenarios: 10 full-access, 10 partial-access, survivors of the prior probe, stratified by N
    summ = EXP / "results" / "a1ctx_prior_summary.json"
    surv = set(json.load(open(summ))["survivors"]) if summ.exists() else set(range(50))
    rows = [r for r in load("a1ctx_felicity") if r["subset"] in ("felicity_full", "felicity_partial") and r["meta"]["scenario"] in surv]
    ctxt = []
    for acc in ("felicity_full", "felicity_partial"):
        g = defaultdict(list)
        for r in rows:
            if r["subset"] == acc: g[r["meta"]["n"]].append(r)
        order = sorted(g); picks = []; rr = random.Random(1)
        for n in order: rr.shuffle(g[n])
        i = 0
        while len(picks) < 10 and any(g.values()):       # round-robin over N strata
            n = order[i % len(order)]; i += 1
            if g[n]: picks.append(g[n].pop())
        ctxt += picks
    for r in ctxt:
        u = r["meta"]["utterance"]
        todo.append({"kind": "a1ctx", "instruction": "Is the speaker's use of 'some' misleading here?", "text": trim(r["context"], 120), "options": ["yes, misleading", "no, fine", "unclear"]})
    order = list(range(len(todo))); rng.shuffle(order)
    with open(out / "HUMAN_AUDIT_TODO.jsonl", "w") as f:
        for n, i in enumerate(order, 1):
            t = dict(todo[i]); t["id"] = f"Q{n:02d}"; f.write(json.dumps(t, ensure_ascii=False) + "\n")
    meta = {"seed": 0, "a1ctx_seed": 1, "n_total": len(todo), "blocks": {"coding": {"n": 40, "minutes": 25}, "a1ctx": {"n": 20, "minutes": 12}}, "total_minutes_estimate": 37,
            "strata_coding": {f"{k[0]}|{k[1]}": min(per, len(strata[k])) for k in keys}, "a1ctx_full_partial": [10, 10], "a1ctx_survivors_used": len(surv),
            "general_instructions": ("Please answer each item on its own, without looking anything up. For the 40 coding items choose the option that best describes which reading of the speaker's utterance the reply takes "
                                     "(the enriched, pragmatic one; the literal one; or hedged/both/neither). For the 20 scenario items decide whether the speaker's use of 'some' is misleading given what the speaker knows. "
                                     "Answer by gut reading; there are no trick items. Budget about 40 minutes in total."),
            "note": "No model names, conditions or labels appear in the items."}
    json.dump(meta, open(out / "HUMAN_AUDIT_meta.json", "w"), indent=1)
    print("audit file", len(todo), "items")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["pool", "label", "audit"]); ap.add_argument("--model"); ap.add_argument("--out", required=True); ap.add_argument("--perms", type=int, default=1); ap.add_argument("--outname", default="labels_B.jsonl")
    a = ap.parse_args(); {"pool": cmd_pool, "label": cmd_label, "audit": cmd_audit}[a.cmd](a)
