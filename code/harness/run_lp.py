#!/usr/bin/env python
"""P-lp (estimand i of plan E2): fixed 4-shot plain-text prompt, next-token letter probabilities, for base / stage checkpoints (no chat template).
  python run_lp.py --config ../configs/E2_lp.json --model <path> --out <dir>
Same items, permutations and scoring as run_vllm.py (condition name 'lp4'); the 4 shots are neutral (non-pragmatic) questions in the same format, letters balanced."""
import argparse, json, math, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import is_pragmatic, labels_of, letter_from_text, load_items_from_config, perm_for, render_user, sha16
from aggregate import aggregate
SHOTS = [("Which of these is a mammal?", ["Salmon", "Sparrow", "Elephant"], 2), ("What is 7 + 5?", ["11", "12", "13"], 1),
         ("Which word is spelled correctly?", ["recieve", "receive", "receeve"], 1), ("Which of these is the largest planet?", ["Jupiter", "Mars", "Venus"], 0)]
def shots():
    out = []
    for q, opts, g in SHOTS:
        out.append(f"{q}\n\nOptions:\n" + "\n".join(f"{'ABC'[i]}. {o}" for i, o in enumerate(opts)) + f"\n\nRespond with only the letter of the best answer.\nAnswer: {'ABC'[g]}")
    return "\n\n".join(out)
ap = argparse.ArgumentParser(); ap.add_argument("--config", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu_util", type=float, default=0.88); a = ap.parse_args()
cfg = json.load(open(a.config)); items = load_items_from_config(cfg); seeds = cfg["seeds"]
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, dtype="bfloat16", seed=0, max_model_len=4096, enable_prefix_caching=True)
H = shots(); jobs = []
for ii, it in enumerate(items):
    for s in seeds:
        perm = perm_for(it["id"], s, len(it["options"]), it.get("shuffle", True)); jobs.append((ii, s, perm, H + "\n\n" + render_user(it, perm, "default") + "\nAnswer:"))
t0 = time.time(); outs = llm.generate([j[3] for j in jobs], SamplingParams(temperature=0.0, max_tokens=2, logprobs=20)); rows = []
for (ii, s, perm, pr), o in zip(jobs, outs):
    it = items[ii]; k = len(it["options"]); lab = labels_of(it); lp = {}
    for d in (o.outputs[0].logprobs or [{}])[0].values():
        t = (d.decoded_token or "").strip()
        if len(t) == 1 and t in lab: lp[lab.index(t)] = max(lp.get(lab.index(t), -1e9), d.logprob)
    valid = bool(lp); z = sum(math.exp(v) for v in lp.values()) or 1.0; p_pos = [math.exp(lp[j]) / z if j in lp else 0.0 for j in range(k)]
    pos = max(range(k), key=lambda j: p_pos[j]) if valid else None; choice = perm[pos] if valid else None
    pc = [0.0] * k
    for j, c in enumerate(perm): pc[c] = p_pos[j]
    rows.append({"item_id": it["id"], "bucket": it["bucket"], "phenomenon": it["phenomenon"], "subset": it["subset"], "role": it["role"], "split": it.get("split"),
                 "variant": it.get("variant_applied"), "condition": "lp4", "seed": s, "perm": perm, "choice": choice, "valid": valid, "is_gold": is_pragmatic(it, choice) if valid else None,
                 "is_literal": ((choice == it["literal"]) if valid else None) if it.get("literal") is not None else None, "p_canon": pc, "think_mode": False,
                 "think_tokens": 0, "natural": None, "stage1_finish": None, "answer_tokens": 1, "total_tokens": 1, "gold": it["gold"]})
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
with open(out / "samples.jsonl", "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
json.dump({"meta": {"model": a.model, "config_sha16": sha16(a.config), "conditions": ["lp4"], "seeds": seeds, "n_items": len(items), "wall_seconds": time.time() - t0,
                    "sampling": "4-shot plain text, greedy letter readout", "config": cfg}, "aggregate": aggregate(rows)}, open(out / "results.json", "w"), indent=1)
print("[done] lp4", len(rows), flush=True)
