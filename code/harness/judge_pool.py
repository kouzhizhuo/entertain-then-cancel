#!/usr/bin/env python
"""Open-weight JUDGE codes (gpt-oss, harmony template) for the 294-text reference pool: the same rater prompt and three option orders as labeler B (gold_pool.py label), reasoning effort low, greedy,
answer = letter in the final channel. Judge family (gpt-oss) differs from the answering model (Qwen3-8B) and from rater B (Qwen3-32B). Writes <out>/<outname> with one row per pool text
({id, label, votes, rater}); an unparsable order counts as no vote. usage: python judge_pool.py --model <gpt-oss path> --out exp/gold --outname labels_judge_gptoss20b.jsonl [--tp 1] [--effort low]"""
import argparse, json, os, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gold_pool import OPTS
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--outname", required=True)
ap.add_argument("--tp", type=int, default=1); ap.add_argument("--effort", default="low"); ap.add_argument("--max_tokens", type=int, default=1024)
a = ap.parse_args(); os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
rows = [json.loads(l) for l in open(Path(a.out) / "reference_pool.jsonl")]
llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=0.88, seed=0, max_model_len=a.max_tokens + 2048); tok = llm.get_tokenizer()
ORD = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]; FINAL = "<|channel|>final<|message|>"
def prompt(r, order):
    opts = "\n".join(f"{'ABC'[j]}. {OPTS[c]}" for j, c in enumerate(order))
    u = (f"Read the speaker's utterance and the reply that interprets it.\n\n{r['item_text']}\n\nReply: {r['answer_text']}\n\n"
         f"Which reading of the utterance does the reply take?\n{opts}\n\nRespond with only the letter.")
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, reasoning_effort=a.effort)
jobs = [(r, o) for r in rows for o in ORD]
outs = llm.generate([prompt(r, o) for r, o in jobs], SamplingParams(temperature=0.0, max_tokens=a.max_tokens, skip_special_tokens=False))
votes = {r["id"]: [] for r in rows}; bad = 0
for (r, order), o in zip(jobs, outs):
    t = o.outputs[0].text
    fin = t.split(FINAL, 1)[1] if FINAL in t else ""
    L = next((ch for ch in fin.strip()[:6] if ch in "ABC"), None)
    if L is None: bad += 1; continue
    votes[r["id"]].append(OPTS[order["ABC".index(L)]])
with open(Path(a.out) / a.outname, "w") as f:
    for r in rows:
        v = Counter(votes[r["id"]]); lab = v.most_common(1)[0][0] if v else "hedged / both / neither"
        f.write(json.dumps({"id": r["id"], "label": lab, "votes": dict(v), "n_votes": sum(v.values()), "rater": f"open-weight judge {Path(a.model).name}, reasoning {a.effort}, 3 option orders, majority"}) + "\n")
print("judged", len(rows), "texts; unparsable orders:", bad, "of", len(jobs), flush=True)
