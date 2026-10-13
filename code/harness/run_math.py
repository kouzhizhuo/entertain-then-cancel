#!/usr/bin/env python
"""Capability covariate: MATH-500 (200-item seeded subset) under NT (enable_thinking=False) and NAT (natural thinking, cap) for one model.
  python run_math.py --model <path> --out <dir> [--n 200] [--cap 8192]
Grading: math_verify (parse gold and the last \\boxed{} of the post-thinking text). A NAT trace that hits the cap without </think> counts as wrong.
Writes <out>/math500.json (acc, mean tokens, truncation share per condition) and samples.jsonl."""
import argparse, json, random, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--cap", type=int, default=8192); ap.add_argument("--nt_cap", type=int, default=4096); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--data", default="<path>"); ap.add_argument("--gpu_util", type=float, default=0.88)
ap.add_argument("--tp", type=int, default=1); ap.add_argument("--conds", nargs="+", default=["nt", "nat"])
a = ap.parse_args()
from math_verify import parse, verify
from vllm import LLM, SamplingParams
rows = [json.loads(l) for l in open(a.data)]
random.Random(0).shuffle(rows); rows = rows[: a.n]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, dtype="bfloat16", seed=0, max_model_len=a.cap + 2560, enable_prefix_caching=True)
tok = llm.get_tokenizer()
SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
def chat(q, think): return tok.apply_chat_template([{"role": "user", "content": q + SUF}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
res, samples = {}, []
t0 = time.time()
for name, think, cap in [x for x in (("nt", False, a.nt_cap), ("nat", True, a.cap + 2048)) if x[0] in a.conds]:
    sps = SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=cap, seed=a.seed)
    outs = llm.generate([chat(r["problem"], think) for r in rows], sps)
    ok, toks, trunc = [], [], 0
    for r, o in zip(rows, outs):
        c = o.outputs[0]; txt = c.text
        final = txt.split("</think>")[-1] if "</think>" in txt else ("" if think else txt)
        gold = parse(f"${r['answer']}$"); pred = parse(final)
        good = bool(final) and verify(gold, pred)
        unfinished = think and "</think>" not in txt
        ok.append(1 if good else 0); toks.append(len(c.token_ids)); trunc += 1 if unfinished else 0
        samples.append({"id": r.get("unique_id"), "cond": name, "correct": good, "tokens": len(c.token_ids), "unfinished": bool(unfinished), "level": r.get("level")})
    res[name] = {"n": len(rows), "acc": 100 * sum(ok) / len(ok), "tokens_mean": sum(toks) / len(toks), "unfinished_share": 100 * trunc / len(rows)}
    print(name, res[name], flush=True)
json.dump({"model": a.model, "n": a.n, "seed": a.seed, "cap_nat": a.cap, "cap_nt": a.nt_cap, "sampling": "T0.6 top_p0.95 top_k20", "results": res, "wall_seconds": time.time() - t0},
          open(out / "math500.json", "w"), indent=1)
with open(out / "samples.jsonl", "w") as f:
    for s in samples: f.write(json.dumps(s) + "\n")
