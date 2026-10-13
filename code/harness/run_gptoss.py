#!/usr/bin/env python
"""gpt-oss (harmony template) under reasoning-effort levels low/medium/high (E3; vLLM, same items/prompts/permutations as run_vllm.py).
  python run_gptoss.py --config ../configs/E3_check.json --model <path> --out <dir> [--efforts low medium high]
Generation: T=1.0, top_p=1.0 (gpt-oss recommended), cap = cfg.cap tokens; text decoded with special tokens kept so the analysis / final channels can be split.
Answer: first option label after '<|channel|>final<|message|>'. If the cap is hit before the final channel, the analysis is closed
('<|end|><|start|>assistant<|channel|>final<|message|>') and the label is read from the restricted next-token distribution (forced = True).
Logged per sample: realised reasoning tokens (tokens of the analysis channel), total tokens, forced, finish reason.
Also writes effort_check.json: median / IQR of realised reasoning tokens per effort level and whether low < medium < high (E3 requires it)."""
import argparse, json, math, os, sys, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import is_pragmatic, labels_of, letter_from_text, load_items_from_config, perm_for, render_user, sha16
from aggregate import aggregate
ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--efforts", nargs="+", default=["low", "medium", "high"]); ap.add_argument("--tp", type=int, default=1)
ap.add_argument("--gpu_util", type=float, default=0.88); ap.add_argument("--limit", type=int, default=None)
a = ap.parse_args()
cfg = json.load(open(a.config)); items = load_items_from_config(cfg)[: a.limit]; seeds = cfg["seeds"]; cap = cfg.get("cap", 8192)
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, seed=0, max_model_len=cap + 2048, enable_prefix_caching=True)
tok = llm.get_tokenizer()
FINAL = "<|channel|>final<|message|>"
jobs = []
for ii, it in enumerate(items):
    for s in seeds:
        perm = perm_for(it["id"], s, len(it["options"]), it.get("shuffle", True))
        user = render_user(it, perm, "default")
        for e in a.efforts:
            pr = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, reasoning_effort=e)
            jobs.append((ii, s, e, perm, pr))
sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=cap, skip_special_tokens=False, seed=s * 1_000_003 + zlib.crc32(items[ii]["id"].encode()) % 1_000_003)
       for (ii, s, e, perm, pr) in jobs]
t0 = time.time(); outs = llm.generate([j[4] for j in jobs], sps); print(f"[gen] {len(jobs)} in {time.time()-t0:.0f}s", flush=True)
rows, forced_jobs = [], []
for (ii, s, e, perm, pr), o in zip(jobs, outs):
    it = items[ii]; co = o.outputs[0]; txt = co.text; k = len(it["options"]); lab = labels_of(it)
    if FINAL in txt:
        ana, fin = txt.split(FINAL, 1); pos = letter_from_text(fin.replace("<|return|>", "").replace("<|end|>", ""), k, lab); forced = False
    else:
        ana, pos, forced = txt, None, True
    rows.append({"ii": ii, "s": s, "e": e, "perm": perm, "pos": pos, "forced": forced, "ana_tokens": len(tok.encode(ana, add_special_tokens=False)),
                 "tot": len(co.token_ids), "finish": co.finish_reason, "prompt": pr, "ana": ana, "p_pos": None})
fj = [r for r in rows if r["forced"]]
if fj:
    sp2 = SamplingParams(temperature=0.0, max_tokens=4, logprobs=20)
    o2 = llm.generate([r["prompt"] + r["ana"] + "<|end|><|start|>assistant" + FINAL for r in fj], sp2)
    for r, o in zip(fj, o2):
        it = items[r["ii"]]; lab = labels_of(it); lp = {}
        for d in (o.outputs[0].logprobs or [{}])[0].values():
            t = (d.decoded_token or "").strip()
            if len(t) == 1 and t in lab: lp[lab.index(t)] = max(lp.get(lab.index(t), -1e9), d.logprob)
        if lp: r["pos"] = max(lp, key=lp.get)
samples = []
for r in rows:
    it = items[r["ii"]]; k = len(it["options"]); valid = r["pos"] is not None; choice = r["perm"][r["pos"]] if valid else None
    samples.append({"item_id": it["id"], "bucket": it["bucket"], "phenomenon": it["phenomenon"], "subset": it["subset"], "role": it["role"], "split": it.get("split"),
                    "variant": it.get("variant_applied"), "condition": f"effort_{r['e']}", "seed": r["s"], "perm": r["perm"], "choice": choice, "valid": valid,
                    "is_gold": is_pragmatic(it, choice) if valid else None,
                    "is_literal": ((choice == it["literal"]) if valid else None) if it.get("literal") is not None else None,
                    "choice_kind": it["option_kinds"][choice] if valid else None, "think_mode": True, "think_tokens": r["ana_tokens"], "natural": not r["forced"],
                    "stage1_finish": r["finish"], "answer_tokens": r["tot"] - r["ana_tokens"], "total_tokens": r["tot"]})
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "samples.jsonl", "w") as f:
    for r in samples: f.write(json.dumps(r, ensure_ascii=False) + "\n")
import numpy as np
chk = {}
for e in a.efforts:
    t = [r["think_tokens"] for r in samples if r["condition"] == f"effort_{e}"]
    chk[e] = {"n": len(t), "median": float(np.median(t)), "q25": float(np.percentile(t, 25)), "q75": float(np.percentile(t, 75)), "forced_share": float(np.mean([not r["natural"] for r in samples if r["condition"] == f"effort_{e}"]))}
meds = [chk[e]["median"] for e in a.efforts]
chk["monotone_low_lt_medium_lt_high"] = all(x < y for x, y in zip(meds, meds[1:]))
json.dump(chk, open(out / "effort_check.json", "w"), indent=1)
json.dump({"meta": {"model": a.model, "config_sha16": sha16(a.config), "efforts": a.efforts, "seeds": seeds, "n_items": len(items), "sampling": "T1.0 top_p1.0", "cap": cap,
                    "wall_seconds": time.time() - t0, "conditions": [f"effort_{e}" for e in a.efforts], "effort_check": chk}, "aggregate": aggregate(samples)}, open(out / "results.json", "w"), indent=1)
print("[done]", json.dumps(chk), flush=True)
