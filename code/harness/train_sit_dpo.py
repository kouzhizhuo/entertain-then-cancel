#!/usr/bin/env python3
"""ATTEMPT 4 SIT-DPO (notes/METHOD_DESIGN.md "ATTEMPT 4"). FIXED a priori, no tuning, final checkpoint only:
  DPO with LoRA r=16, alpha=32 (dropout 0.05) on q,k,v,o,gate,up,down; bf16 base; beta=0.1; AdamW lr 2e-5, cosine to 0 (3% linear warm-up, fixed), weight decay 0, grad-clip 1.0; 1 epoch; max length 4096 (a pair with either side longer is DROPPED and counted);
  seed 0; effective batch 8 pairs (micro-batch 1 pair x 8 accumulation; the spec leaves this open, fixed here before any run). Start from the BASE; the reference is the frozen base (adapter disabled, no grad).
  loss = -logsigmoid(beta * ((logp_pol(chosen) - logp_pol(rejected)) - (logp_ref(chosen) - logp_ref(rejected)))), log-probs SUMMED over the supervised tokens (thought + close tag + answer + end-of-turn); the prompt and the generation prompt / <think> opener are masked.
usage: python train_sit_dpo.py --model <path> --data <pairs.jsonl|pairs_shuffled.jsonl> --out <dir> [--limit 8 --max_steps 2] [--merge]    writes <out>/adapter, train_log.json, merged/ with --merge"""
import argparse, json, math, random, time
from pathlib import Path
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
HP = dict(r=16, alpha=32, dropout=0.05, lr=2e-5, beta=0.1, epochs=1, max_len=4096, seed=0, eff_batch=8, warmup=0.03, wd=0.0, clip=1.0)
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--max_steps", type=int, default=None); ap.add_argument("--think_open", default="<think>\n"); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
random.seed(HP["seed"]); torch.manual_seed(HP["seed"]); torch.cuda.manual_seed_all(HP["seed"])
tok = AutoTokenizer.from_pretrained(a.model); eos = tok.eos_token
rows = [json.loads(l) for l in open(a.data)][: a.limit]
def enc(r, side):
    p = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    if not p.rstrip().endswith(a.think_open.strip()): p += a.think_open
    pi = tok(p, add_special_tokens=False)["input_ids"]; fi = tok(p + r[side] + eos, add_special_tokens=False)["input_ids"]
    assert fi[: len(pi)] == pi, "prompt/target tokenisation boundary mismatch"
    return fi, len(pi)
data, dropped = [], 0
for r in rows:
    c, pc = enc(r, "chosen"); w, pw = enc(r, "rejected")
    if max(len(c), len(w)) > HP["max_len"]: dropped += 1; continue
    data.append((c, pc, w, pw))
print(f"[data] {len(rows)} pairs, kept {len(data)}, dropped(>{HP['max_len']}) {dropped}", flush=True)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=HP["r"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
params = [p for p in model.parameters() if p.requires_grad]; opt = torch.optim.AdamW(params, lr=HP["lr"], weight_decay=HP["wd"])
def seq_logp(ids, plen):
    x = torch.tensor([ids], device="cuda"); lg = model(input_ids=x).logits[:, :-1].float()
    lp = torch.log_softmax(lg, -1).gather(-1, x[:, 1:].unsqueeze(-1)).squeeze(-1)[0]; del lg
    return lp[plen - 1:].sum()                                      # supervised = every token after the prompt
total = math.ceil(len(data) / HP["eff_batch"]) * HP["epochs"]; warm = max(1, round(HP["warmup"] * total))
lr_at = lambda s: HP["lr"] * (s + 1) / warm if s < warm else 0.5 * HP["lr"] * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
log, step, t0 = {"hp": HP, "n_pairs": len(data), "dropped_over_max_len": dropped, "total_steps": total, "steps": []}, 0, time.time()
model.train(); torch.cuda.reset_peak_memory_stats()
order = list(range(len(data))); random.Random(HP["seed"]).shuffle(order)
for b in range(0, len(order), HP["eff_batch"]):
    batch = [data[i] for i in order[b: b + HP["eff_batch"]]]; ts = time.time(); ls = acc = marg = 0.0
    for g in opt.param_groups: g["lr"] = lr_at(step)
    for c, pc, w, pw in batch:
        with torch.no_grad(), model.disable_adapter(): rc, rw = seq_logp(c, pc), seq_logp(w, pw)
        pcl = seq_logp(c, pc); pwl = seq_logp(w, pw)
        z = HP["beta"] * ((pcl - pwl) - (rc - rw)); loss = -F.logsigmoid(z) / len(batch); loss.backward()
        ls += loss.item(); acc += float(z.item() > 0) / len(batch); marg += z.item() / HP["beta"] / len(batch)
    gn = torch.nn.utils.clip_grad_norm_(params, HP["clip"]).item(); opt.step(); opt.zero_grad(set_to_none=True); step += 1; dt = time.time() - ts
    log["steps"].append({"step": step, "loss": ls, "reward_acc": acc, "margin": marg, "lr": lr_at(step - 1), "grad_norm": gn, "sec": dt, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30})
    print(f"[train] step {step}/{total} loss {ls:.4f} acc {acc:.2f} margin {marg:.2f} lr {lr_at(step-1):.2e} |g| {gn:.2f} {dt:.1f}s peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
    if a.max_steps and step >= a.max_steps: break
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); model.save_pretrained(out / "adapter"); tok.save_pretrained(out / "adapter")
log["wall_seconds"] = time.time() - t0; log["sec_per_step_mean"] = sum(s["sec"] for s in log["steps"]) / len(log["steps"]); log["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
json.dump(log, open(out / "train_log.json", "w"), indent=1)
if a.merge: m = model.merge_and_unload(); m.save_pretrained(out / "merged", safe_serialization=True); tok.save_pretrained(out / "merged")
print(f"[done] steps {step}, {log['sec_per_step_mean']:.1f}s/step, peak {log['peak_mem_gb']:.1f} GB", flush=True)
