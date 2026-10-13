#!/usr/bin/env python3
"""SIT-train LoRA SFT (D-CONTINGENCY, notes/METHOD_DESIGN.md). FIXED hyper-parameters, no tuning, final checkpoint only:
  LoRA r=16, alpha=32, dropout=0.05 on q,k,v,o,gate,up,down; bf16 base; AdamW lr 1e-4, cosine decay to 0 (3% linear warm-up, fixed), weight decay 0, grad-clip 1.0;
  2 epochs; max length 4096 tokens (examples longer than that are DROPPED, not truncated, and counted); seed 0; effective batch 16 sequences (micro-batch 1 x 16 accumulation),
  loss = token-mean over the supervised tokens of the 16 sequences; loss ONLY on thought + close tag + answer + end-of-turn token (the prompt incl. the generation prompt and the "<think>\\n" opener is masked).
usage: python train_sit_lora.py --model <path> --data <selected.jsonl|control.jsonl> --out <dir> [--limit 50 --max_steps 2]   (smoke: 50 examples, 2 optimizer steps)
data line: {"messages":[{"role":"user","content":...}], "target": "<thought></think>\\n\\n<answer>"}  (GSM8K: target = thought + </think> + answer text)
Writes <out>/adapter (final LoRA only), <out>/train_log.json; --merge also writes <out>/merged (bf16 merged weights, for serving without LoRA support)."""
import argparse, json, math, os, random, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

HP = dict(r=16, alpha=32, dropout=0.05, lr=1e-4, epochs=2, max_len=4096, seed=0, eff_batch=16, warmup=0.03, wd=0.0, clip=1.0)
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--max_steps", type=int, default=None, help="smoke only: stop after this many optimizer steps (the cosine schedule is still computed for the full run)")
ap.add_argument("--think_open", default="<think>\n"); ap.add_argument("--merge", action="store_true"); a = ap.parse_args()

random.seed(HP["seed"]); torch.manual_seed(HP["seed"]); torch.cuda.manual_seed_all(HP["seed"])
tok = AutoTokenizer.from_pretrained(a.model); eos = tok.eos_token
rows = [json.loads(l) for l in open(a.data)][: a.limit]

def encode(r):
    p = tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True)   # as the harness chat(): thinking on
    if not p.rstrip().endswith(a.think_open.strip()): p += a.think_open
    pi = tok(p, add_special_tokens=False)["input_ids"]; fi = tok(p + r["target"] + eos, add_special_tokens=False)["input_ids"]
    assert fi[: len(pi)] == pi, "prompt/target tokenisation boundary mismatch"          # the boundary must not merge tokens, or the mask would be wrong
    return fi, [-100] * len(pi) + fi[len(pi):]
data, dropped = [], 0
for r in rows:
    ids, lab = encode(r)
    if len(ids) > HP["max_len"]: dropped += 1; continue
    data.append((ids, lab))
n_sup = sum(sum(x != -100 for x in l) for _, l in data)
print(f"[data] {len(rows)} rows, kept {len(data)}, dropped(>{HP['max_len']}) {dropped}, tokens {sum(len(i) for i, _ in data)}, supervised tokens {n_sup} (mean {n_sup/max(1,len(data)):.0f} per example)", flush=True)
ex_ids, ex_lab = data[0]; first = next(i for i, x in enumerate(ex_lab) if x != -100)
print("[mask] example 0: prompt tokens", first, "| last masked piece:", repr(tok.decode(ex_ids[max(0, first - 6):first])), "| first supervised piece:", repr(tok.decode(ex_ids[first:first + 6])), "| last supervised piece:", repr(tok.decode(ex_ids[-4:])), flush=True)

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.config.use_cache = False
model = get_peft_model(model, LoraConfig(r=HP["r"], lora_alpha=HP["alpha"], lora_dropout=HP["dropout"], target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
model.print_trainable_parameters()
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=HP["lr"], weight_decay=HP["wd"])
steps_per_epoch = math.ceil(len(data) / HP["eff_batch"]); total = steps_per_epoch * HP["epochs"]; warm = max(1, round(HP["warmup"] * total))
lr_at = lambda s: HP["lr"] * (s + 1) / warm if s < warm else 0.5 * HP["lr"] * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm)))
log, step, t0 = {"hp": HP, "n_examples": len(data), "dropped_over_max_len": dropped, "supervised_tokens": n_sup, "total_steps": total, "steps": []}, 0, time.time()
model.train(); torch.cuda.reset_peak_memory_stats()
for ep in range(HP["epochs"]):
    order = list(range(len(data))); random.Random(HP["seed"] + ep).shuffle(order)
    for b in range(0, len(order), HP["eff_batch"]):
        batch = [data[i] for i in order[b: b + HP["eff_batch"]]]; ntok = sum(sum(x != -100 for x in l[1:]) for _, l in batch); loss_sum = 0.0; ts = time.time(); toks = 0
        for g in opt.param_groups: g["lr"] = lr_at(step)
        for ids, lab in batch:
            x = torch.tensor([ids], device="cuda"); y = torch.tensor([lab], device="cuda"); toks += len(ids)
            logits = model(input_ids=x).logits[:, :-1].float(); loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / ntok
            loss.backward(); loss_sum += loss.item(); del logits, loss
        gn = torch.nn.utils.clip_grad_norm_(params, HP["clip"]).item(); opt.step(); opt.zero_grad(set_to_none=True); step += 1
        dt = time.time() - ts; log["steps"].append({"step": step, "epoch": ep, "loss": loss_sum, "lr": lr_at(step - 1), "grad_norm": gn, "sec": dt, "tokens": toks, "supervised": ntok, "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30})
        print(f"[train] step {step}/{total} ep {ep} loss {loss_sum:.4f} lr {lr_at(step-1):.2e} |g| {gn:.2f} {dt:.1f}s {toks} tok peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB", flush=True)
        if a.max_steps and step >= a.max_steps: break
    if a.max_steps and step >= a.max_steps: break
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); model.save_pretrained(out / "adapter"); tok.save_pretrained(out / "adapter")
log["wall_seconds"] = time.time() - t0; log["sec_per_step_mean"] = sum(s["sec"] for s in log["steps"]) / len(log["steps"]); log["sec_per_example_mean"] = log["sec_per_step_mean"] / HP["eff_batch"]
log["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30; json.dump(log, open(out / "train_log.json", "w"), indent=1)
if a.merge:
    m = model.merge_and_unload(); m.save_pretrained(out / "merged", safe_serialization=True); tok.save_pretrained(out / "merged")
print(f"[done] steps {step}, {log['sec_per_step_mean']:.1f}s/step, peak {log['peak_mem_gb']:.1f} GB, adapter -> {out/'adapter'}", flush=True)
