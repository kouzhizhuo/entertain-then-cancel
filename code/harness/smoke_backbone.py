#!/usr/bin/env python
"""Backbone thinking-switch smoke test (10 items x every plausible switch). usage: smoke_backbone.py --model <path> --out <json>
Variants: chat-template enable_thinking True/False, system prompt /think vs /no_think, Granite-style thinking=True/False, plain template.
For each: mean generated tokens (cap 1536), share of outputs containing <think>/</think>, share that stopped before the cap, first 160 chars, prompt tail.
A switch 'works' if think-on vs think-off differ in realised tokens (off ~ 0 / short, on > 0)."""
import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_set, perm_for, render_user
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=10)
ap.add_argument("--max_tokens", type=int, default=1536); ap.add_argument("--tp", type=int, default=1); a = ap.parse_args()
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
a3 = [r for r in load_set("imppres_scalar_target") if r.get("split") == "test"][: a.n // 2]
a2 = [r for r in load_set("a2_scalar_diversity") if r.get("split") == "test"][: a.n - a.n // 2]
items = []
for r in a3:
    v = r["variants"]["felicity"]; items.append({**r, **v})
items += a2
users = [render_user(it, perm_for(it["id"], 0, len(it["options"])), "default") for it in items]
llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=0.85, dtype="bfloat16", seed=0, max_model_len=a.max_tokens + 2048, trust_remote_code=True)
tok = llm.get_tokenizer()
V = {"tmpl_think": ({}, {"enable_thinking": True}), "tmpl_nothink": ({}, {"enable_thinking": False}),
     "sys_think": ({"system": "/think"}, {}), "sys_nothink": ({"system": "/no_think"}, {}),
     "granite_think": ({}, {"thinking": True}), "granite_nothink": ({}, {"thinking": False}), "plain": ({}, {})}
res = {"model": a.model, "n": len(users), "variants": {}}
for name, (extra, kw) in V.items():
    try:
        prompts = []
        for u in users:
            msgs = ([{"role": "system", "content": extra["system"]}] if "system" in extra else []) + [{"role": "user", "content": u}]
            prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))
        outs = llm.generate(prompts, SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=0))
        txt = [o.outputs[0].text for o in outs]; nt = [len(o.outputs[0].token_ids) for o in outs]
        res["variants"][name] = {"mean_tokens": sum(nt) / len(nt), "max_tokens": max(nt), "share_think_tag": sum(("<think>" in t or "</think>" in t) for t in txt) / len(txt),
                                 "share_stopped": sum(o.outputs[0].finish_reason == "stop" for o in outs) / len(outs), "first160": txt[0][:160], "prompt_tail": prompts[0][-90:]}
    except Exception as e:
        res["variants"][name] = {"error": f"{type(e).__name__}: {str(e)[:160]}"}
    print(name, {k: v for k, v in res["variants"][name].items() if k in ("mean_tokens", "share_think_tag", "error")}, flush=True)
json.dump(res, open(a.out, "w"), indent=1)
