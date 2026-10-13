#!/usr/bin/env python
"""P2 inference harness (vLLM, Qwen3 family): thinking off / thinking budgets (budget forcing) / forced-choice extraction.

Env `gen` on <host>, ONE GPU (P2 owns GPU 4):
  export CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0
  python run_vllm.py --config ../configs/pilot.json --model <path> --out ../results/pilot/Qwen3-8B

Design (see notes in exp/harness/README.md)
  * Prompt = item context + question + lettered options (order permuted per (item, seed)) + "Respond with only the letter".
  * Conditions: `off` (chat template enable_thinking=False) | `think_b<N>` (thinking on, capped at N tokens, then forced
    close: "...I have to give the solution based on the thinking directly now.\n</think>") | `think_full` (natural end, cap
    = cfg.cap) | optional `+brief` / `+careful` soft-prompt suffixes.  think_b0 = thinking template but closed immediately.
  * COUPLED budgets: one thinking trace is sampled per (item, seed, hint) up to the largest budget and cut at each smaller
    budget, so budget curves compare *the same reasoning*, cut at different lengths (paired design; also ~N x cheaper).
  * Answer extraction: after "</think>\n\n" the next-token distribution is restricted to the option letters (argmax = choice;
    the letter probabilities are stored).  Greedy, so answers are deterministic given the (possibly cut) trace; seeds vary the
    trace sampling and the option order.
  * Logged per sample: seed, permutation, choice, letter probabilities, thinking tokens, natural/forced, stage-1 finish reason,
    answer tokens, total tokens.  Also traces.jsonl (thinking text) and prompt_examples.txt.
"""
import argparse, json, os, platform, sys, time, zlib
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IST, LATE, LETTERS, is_pragmatic, labels_of, letter_from_text, load_items_from_config, parse_condition, perm_for, render_user, sha16
from aggregate import aggregate

FORCE_MSG = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n"
CLOSE = "</think>\n\n"   # default; overridden from --think_close / --think_open for models with other markers (e.g. Magistral [THINK]...[/THINK])


def _grp(rows):
    d = {}
    for r in rows:
        d.setdefault((r["bucket"], r["condition"]), []).append(r)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="local path (or HF id) of the model")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--conditions", nargs="*")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N items (smoke test)")
    ap.add_argument("--gpu_util", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--think_open", default="<think>\n", help="string that opens the thinking block (prefilled unless the template already ends with it)")
    ap.add_argument("--think_close", default="</think>", help="string that closes the thinking block (stop string in stage 1)")
    ap.add_argument("--think_kwarg", default="enable_thinking", help="chat-template kwarg that switches thinking (Granite: thinking)")
    ap.add_argument("--sysflag", action="store_true", help="switch thinking with a system prompt '/think' | '/no_think' (Nemotron-Nano-v2) instead of a template kwarg")
    ap.add_argument("--think_post", default="\n\n", help="text after the closing tag before the forced-choice readout (Granite: '\n<response>')")
    ap.add_argument("--system_file", default=None, help="text file used as the system prompt for every request (Magistral: its SYSTEM_PROMPT.txt switches thinking on)")
    ap.add_argument("--mistral_raw", action="store_true", help="build '<s>[SYSTEM_PROMPT]..[/SYSTEM_PROMPT][INST]..[/INST]' by hand (repos with tekken.json and no HF chat template)")
    ap.add_argument("--reasoning_effort", default=None, help="Qwen3.8: passed to the chat template as reasoning_effort (xhigh|medium|low) when thinking is on; the template turns it into a system instruction")
    ap.add_argument("--max_num_seqs", type=int, default=None, help="hybrid (Mamba/linear-attention) models such as Qwen3.5/3.8: vLLM needs max_num_seqs <= the number of Mamba cache blocks (272 for 27B at 0.88 utilisation)")
    ap.add_argument("--sys_on", default=None, help="system prompt text that switches thinking ON (Llama-Nemotron-Nano-8B-v1: 'detailed thinking on')")
    ap.add_argument("--sys_off", default=None, help="system prompt text that switches thinking OFF ('detailed thinking off')")
    ap.add_argument("--user_suffix_off", default="", help="text appended to the user message when thinking is OFF (openPangu: ' /no_think')")
    ap.add_argument("--user_suffix_on", default="", help="text appended to the user message when thinking is ON")
    ap.add_argument("--off_prefix", default="", help="text appended to the generation prompt in OFF mode (openPangu fast mode opens with an empty thought: '[unused16][unused17]')")
    ap.add_argument("--model_impl", default=None, help="vLLM model_impl (e.g. 'transformers' for custom-code models)")
    ap.add_argument("--system_always", action="store_true", help="use --system_file in BOTH thinking modes (openPangu: the vendor's mandatory system prompt)")
    ap.add_argument("--keep_special", action="store_true", help="skip_special_tokens=False in every generation (openPangu thought markers are special tokens)")
    ap.add_argument("--stop_token_ids", type=int, nargs="*", default=None, help="extra stop token ids (openPangu: 45892)")
    ap.add_argument("--single_close", action="store_true", help="natural traces: the cached token ids include the stop token (the close tag); strip it so stage 2 / the late insertion see ONE close tag (default off = original behaviour with a doubled close tag after natural traces)")
    ap.add_argument("--late_max", type=int, default=256, help="max extra thinking tokens after a late sentence (conditions ist_late / late_<key>)")
    ap.add_argument("--tokenizer_mode", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true")
    ap.add_argument("--dry", action="store_true", help="render prompts only (no model)")
    a = ap.parse_args()

    cfg = json.load(open(a.config))
    OPEN_TAG, CLOSE_TAG = a.think_open, a.think_close; CLOSE = CLOSE_TAG + a.think_post
    def _sc(t): return t.rstrip()[: -len(CLOSE_TAG.strip())] if t.rstrip().endswith(CLOSE_TAG.strip()) else t
    seeds = a.seeds if a.seeds is not None and len(a.seeds) else cfg["seeds"]
    conds = [parse_condition(c) for c in (a.conditions or cfg["conditions"])]
    variant = cfg.get("variant", "default")
    cap = cfg.get("cap", 8192)
    temp, top_p, top_k = cfg.get("temperature", 0.7), cfg.get("top_p", 0.95), cfg.get("top_k", 20)
    items = load_items_from_config(cfg)
    if a.limit:
        items = items[: a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = a.model_name or Path(a.model).name
    print(f"[cfg] {name}: {len(items)} items x {len(seeds)} seeds x {len(conds)} conditions; cap={cap} T={temp}", flush=True)

    # ---- build requests -------------------------------------------------------------------------------------------
    reqs = {}  # (item_idx, seed, hint) -> dict(perm, user)
    for ii, it in enumerate(items):
        for s in seeds:
            perm = perm_for(it["id"], s, len(it["options"]), it.get("shuffle", True))
            for c in conds:
                key = (ii, s, c["hint"])
                if key not in reqs:
                    reqs[key] = {"perm": perm, "user": render_user(it, perm, "alt" if variant == "alt" else "default", c["hint"])}
    if a.dry:
        for k in list(reqs)[:3]:
            print("-" * 60, k, "\n" + reqs[k]["user"])
        return

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    import vllm, torch
    t0 = time.time()
    llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, dtype="bfloat16", seed=0,
              max_model_len=a.max_model_len or (cap + 2048), **({"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}), **({"model_impl": a.model_impl} if a.model_impl else {}), enable_prefix_caching=True, tokenizer_mode=a.tokenizer_mode, trust_remote_code=a.trust_remote_code)
    tok = llm.get_tokenizer()
    _extra = {}
    if a.keep_special: _extra["skip_special_tokens"] = False
    if a.stop_token_ids: _extra["stop_token_ids"] = list(a.stop_token_ids)
    _SP = SamplingParams; SamplingParams = lambda **kw: _SP(**kw, **_extra)   # applies the extra decoding options to every generation

    SYS = open(a.system_file).read().strip() if a.system_file else None
    def chat(user, think):
        if a.mistral_raw:   # UNVERIFIED until a GPU smoke test: thinking on = system prompt present, off = no system prompt
            return "<s>" + (f"[SYSTEM_PROMPT]{SYS}[/SYSTEM_PROMPT]" if (SYS and think) else "") + f"[INST]{user}[/INST]"
        user = user + (a.user_suffix_on if think else a.user_suffix_off)
        msgs = [{"role": "user", "content": user}]; kw = {}
        if SYS and (think or a.system_always): msgs = [{"role": "system", "content": SYS}] + msgs
        if a.sys_on: msgs = [{"role": "system", "content": a.sys_on if think else a.sys_off}] + msgs
        elif a.sysflag: msgs = [{"role": "system", "content": "/think" if think else "/no_think"}] + msgs
        elif not SYS: kw = {a.think_kwarg: think}
        if think and a.reasoning_effort: kw["reasoning_effort"] = a.reasoning_effort
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        return out + (a.off_prefix if not think else "")

    # ---- stage 1: one thinking trace per (item, seed, hint, seed-sentence), cut later at every budget -----------------
    think_conds = [c for c in conds if c["think"]]
    cot_conds = [c for c in conds if c["cot"]]
    traces, cot_tr, late_ext = {}, {}, {}
    def _open_think(p):  # templates that already end in <think> (e.g. OLMo-3-Think) must not get a second one
        return p if p.rstrip().endswith(OPEN_TAG.strip()) else p + OPEN_TAG
    def _seed(ii, s, extra=0):
        return s * 1_000_003 + extra + zlib.crc32(items[ii]["id"].encode()) % 1_000_003
    if think_conds:
        max_b = max((c["budget"] if c["budget"] is not None else cap) for c in think_conds)
        keys1 = sorted({(ii, s, c["hint"], c["seedtype"]) for (ii, s, h) in reqs for c in think_conds if c["hint"] == h},
                       key=lambda k: (k[0], k[1], k[2] or "", k[3] or ""))
        prompts1 = [_open_think(chat(reqs[(k[0], k[1], k[2])]["user"], True)) + ((IST[k[3]] + " ") if k[3] else "") for k in keys1]
        sps1 = [SamplingParams(n=1, temperature=temp, top_p=top_p, top_k=top_k, max_tokens=max(max_b, 1), stop=[CLOSE_TAG], seed=_seed(k[0], k[1]))
                for k in keys1]
        t1 = time.time()
        if max_b < 1:   # only think_b0-type conditions: the thinking trace is empty by definition, nothing to generate (vLLM rejects max_tokens=0)
            for k, p in zip(keys1, prompts1):
                traces[k] = {"prompt": p, "ids": [], "natural_full": False, "finish": "length", "stop_reason": None}
            print(f"[stage1] skipped: max budget 0 ({len(keys1)} empty traces)", flush=True)
        else:
            outs1 = llm.generate(prompts1, sps1)
            print(f"[stage1] {len(keys1)} traces, {sum(len(o.outputs[0].token_ids) for o in outs1)} tokens, {time.time()-t1:.0f}s", flush=True)
            for k, p, o in zip(keys1, prompts1, outs1):
                c = o.outputs[0]
                natural_full = c.finish_reason == "stop"  # stopped on "</think>" (or EOS) before the cap
                traces[k] = {"prompt": p, "ids": list(c.token_ids), "natural_full": natural_full,
                             "finish": c.finish_reason, "stop_reason": c.stop_reason}
        # IST-late: append the fixed sentence after a NATURAL trace and let the model think on for <= 256 tokens
        if any(c["late"] for c in think_conds):
            lk = [(k, c["latekey"]) for k in traces if k[3] is None and traces[k]["natural_full"] for c in think_conds if c["late"] and c["hint"] == k[2]]
            lk = sorted(set(lk), key=lambda t: (t[0][0], t[0][1], str(t[0][2]), t[1]))
            lp_ = [traces[k]["prompt"] + (_sc(tok.decode(traces[k]["ids"])).rstrip() if a.single_close else tok.decode(traces[k]["ids"])) + "\n\n" + LATE[lkey] + " " for k, lkey in lk]
            outl = llm.generate(lp_, [SamplingParams(n=1, temperature=temp, top_p=top_p, top_k=top_k, max_tokens=a.late_max, stop=[CLOSE_TAG], seed=_seed(k[0], k[1], 7)) for k, lkey in lk])
            for (k, lkey), o in zip(lk, outl):
                late_ext[(k, lkey)] = (o.outputs[0].text, len(o.outputs[0].token_ids))
            print(f"[stage1b] late extensions {len(lk)}", flush=True)
        with open(out / "traces.jsonl", "w") as f:
            for k, tr in traces.items():
                f.write(json.dumps({"item_id": items[k[0]]["id"], "seed": k[1], "hint": k[2], "seed_type": k[3], "n_tokens": len(tr["ids"]),
                                    "finish": tr["finish"], "text": tok.decode(tr["ids"])}, ensure_ascii=False) + "\n")
    if cot_conds:   # visible prompted CoT (step: <=512 tokens) / short rationale (rat: <=80 tokens) without native thinking
        SUFX = {"step": "\nLet's think step by step.", "rat": "\nBriefly explain your reasoning in one or two sentences, then answer."}; MAXT = {"step": 512, "rat": 80}
        ck = sorted({(ii, s, c["hint"], c["cot"]) for (ii, s, h) in reqs for c in cot_conds if c["hint"] == h}, key=lambda t: (t[0], t[1], t[2] or "", str(t[3])))   # hints mix None and str when off_cot and off_cot+maxims coexist
        cp = [chat(reqs[(k[0], k[1], k[2])]["user"] + SUFX[k[3]], False) for k in ck]
        oc = llm.generate(cp, [SamplingParams(n=1, temperature=temp, top_p=top_p, top_k=top_k, max_tokens=MAXT[k[3]], seed=_seed(k[0], k[1], 13)) for k in ck])
        for k, p, o in zip(ck, cp, oc):
            cot_tr[k] = {"prompt": p, "text": o.outputs[0].text, "n": len(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason}
        print(f"[stage1c] visible-rationale traces {len(ck)}", flush=True)

    # ---- stage 2: forced-choice answers ------------------------------------------------------------------------------
    jobs = []  # (item_idx, seed, cond, prompt2, meta)
    for ii, it in enumerate(items):
        for s in seeds:
            for c in conds:
                r = reqs[(ii, s, c["hint"])]
                if c["cot"]:
                    ct = cot_tr[(ii, s, c["hint"], c["cot"])]
                    jobs.append((ii, s, c, ct["prompt"] + ct["text"].rstrip() + "\n\nAnswer: ", {"think_tokens": ct["n"], "natural": ct["finish"] == "stop", "stage1_finish": ct["finish"]}))
                    continue
                if not c["think"]:
                    p0 = (chat(r["user"] + " /no_think", True) + "<think>\n\n</think>\n\n") if c["soft"] else chat(r["user"], False)
                    jobs.append((ii, s, c, p0, {"think_tokens": 0, "natural": None, "stage1_finish": None}))
                    continue
                tr = traces[(ii, s, c["hint"], c["seedtype"])]
                B = c["budget"] if c["budget"] is not None else cap
                natural = tr["natural_full"] and len(tr["ids"]) <= B
                ids = tr["ids"][:B] if not natural else tr["ids"]
                text = tok.decode(ids) if ids else ""
                if a.single_close and natural: text = _sc(text)
                extra_tok = 0
                if c["late"] and natural and ((ii, s, c["hint"], None), c["latekey"]) in late_ext:
                    ext, extra_tok = late_ext[((ii, s, c["hint"], None), c["latekey"])]
                    extra_tok += len(tok.encode(LATE[c["latekey"]], add_special_tokens=False))      # the sentence itself counts as thinking tokens
                    text = (text.rstrip() if a.single_close else text) + "\n\n" + LATE[c["latekey"]] + " " + ext
                p2 = tr["prompt"] + text + (CLOSE if (natural or c["plain"]) else FORCE_MSG + CLOSE)
                jobs.append((ii, s, c, p2, {"think_tokens": len(ids) + extra_tok, "inj_tokens": extra_tok, "natural": natural,
                                            "stage1_finish": "stop" if natural else "length"}))
    sp2 = SamplingParams(n=1, temperature=0.0, max_tokens=8, logprobs=20)
    t2 = time.time()
    outs2 = llm.generate([j[3] for j in jobs], sp2)
    print(f"[stage2] {len(jobs)} answers, {time.time()-t2:.0f}s", flush=True)

    rows, finish_counts = [], Counter()
    with open(out / "prompt_examples.txt", "w") as pf:
        seen = set()
        for (ii, s, c, p2, meta), o in zip(jobs, outs2):
            it = items[ii]; k = len(it["options"]); perm = reqs[(ii, s, c["hint"])]["perm"]
            co = o.outputs[0]
            lp = {}
            if co.logprobs:
                for tid, d in co.logprobs[0].items():
                    tokstr = (d.decoded_token or "").strip()
                    lab = labels_of(it)
                    if len(tokstr) == 1 and tokstr in lab:
                        j = lab.index(tokstr)
                        lp[j] = max(lp.get(j, -1e9), d.logprob)
            if lp:
                import math
                z = sum(math.exp(v) for v in lp.values())
                p_pos = [math.exp(lp[j]) / z if j in lp else 0.0 for j in range(k)]
                pos = max(range(k), key=lambda j: p_pos[j]); valid = True
            else:
                pos = letter_from_text(co.text, k, labels_of(it)); valid = pos is not None; p_pos = [0.0] * k
            choice = perm[pos] if valid else None
            p_canon = [0.0] * k
            for j, cidx in enumerate(perm):
                p_canon[cidx] = p_pos[j]
            n_ans = len(co.token_ids)
            row = {"item_id": it["id"], "bucket": it["bucket"], "phenomenon": it["phenomenon"], "subset": it["subset"],
                   "role": it["role"], "split": it.get("split"), "variant": it.get("variant_applied"), "condition": c["name"], "seed": s, "perm": perm, "choice": choice, "valid": valid,
                   "is_gold": is_pragmatic(it, choice) if valid else None,
                   "rating": (choice + 1) if (valid and it.get("scale")) else None,
                   "rating_exp": (sum(p_canon[i] * (i + 1) for i in range(k)) if (valid and it.get("scale")) else None),
                   "is_literal": ((choice == it["literal"]) if valid else None) if it.get("literal") is not None else None,
                   "choice_kind": it["option_kinds"][choice] if valid else None, "p_canon": p_canon, "answer_text": co.text,
                   "think_mode": c["think"], "think_tokens": meta["think_tokens"], "natural": meta["natural"],
                   "stage1_finish": meta["stage1_finish"], "inj_tokens": meta.get("inj_tokens", 0), "answer_tokens": n_ans, "total_tokens": meta["think_tokens"] + n_ans}
            rows.append(row); finish_counts[(c["name"], meta["stage1_finish"])] += 1
            if c["name"] not in seen:
                seen.add(c["name"]); pf.write(f"===== {c['name']} | {it['id']} | seed {s}\n{p2}\n--> {co.text!r}\n\n")


    # ---- free-answer pass: unconstrained final line "Answer: X" on a subset (hedge rate) ---------------------------------
    free_rows = []
    fa = cfg.get("free_answer")
    if fa:
        import re as _re
        FREE_TAIL = "Give a brief justification if needed, then end with a final line of the form 'Answer: X', where X is the letter of your chosen option."
        ANS = _re.compile(r"Answer\s*[:\uff1a]\s*\**\s*\(?([A-H])\b")
        HEDGE = _re.compile(r"\b(depends|ambiguous|either|cannot (be )?determin|can't (be )?determin|not (enough|sufficient)|unclear|impossible to (say|tell))\b", _re.I)
        cand = [i for i, it in enumerate(items) if not it.get("scale")]   # rating items have no letter answer
        step = max(1, len(cand) // fa["n_items"])
        sub = cand[::step][: fa["n_items"]]
        fjobs = []
        for ii in sub:
            it = items[ii]
            for s in seeds:
                perm = perm_for(it["id"], s, len(it["options"]), it.get("shuffle", True))
                user = render_user(it, perm, "alt" if variant == "alt" else "default", None).replace("Respond with only the letter of the best answer.", FREE_TAIL)
                for cn in fa["conditions"]:
                    c = parse_condition(cn)
                    fjobs.append((ii, s, c, perm, chat(user, c["think"])))
        spf = [SamplingParams(n=1, temperature=temp, top_p=top_p, top_k=top_k,
                              max_tokens=(cap + fa.get("answer_tokens", 512)) if c["think"] else fa.get("answer_tokens", 512),
                              seed=(s * 1_000_003 + 17 + zlib.crc32(items[ii]["id"].encode()) % 1_000_003))
               for (ii, s, c, perm, pr) in fjobs]
        tf = time.time()
        outf = llm.generate([j[4] for j in fjobs], spf)
        print(f"[free] {len(fjobs)} free answers, {time.time()-tf:.0f}s", flush=True)
        for (ii, s, c, perm, pr), o in zip(fjobs, outf):
            it = items[ii]; co = o.outputs[0]; k = len(it["options"])
            txt = co.text
            final = txt.split(CLOSE_TAG)[-1] if CLOSE_TAG in txt else (txt if not c["think"] else "")
            ms = ANS.findall(final)
            pos = LETTERS.index(ms[-1]) if ms and LETTERS.index(ms[-1]) < k else None
            choice = perm[pos] if pos is not None else None
            free_rows.append({"item_id": it["id"], "bucket": it["bucket"], "phenomenon": it["phenomenon"], "subset": it["subset"],
                              "role": it["role"], "split": it.get("split"), "variant": it.get("variant_applied"), "condition": c["name"], "seed": s, "perm": perm,
                              "committed": choice is not None, "choice": choice,
                              "is_gold": is_pragmatic(it, choice),
                              "is_literal": ((choice == it["literal"]) if choice is not None else None) if it.get("literal") is not None else None,
                              "hedge_words": bool(HEDGE.search(final)), "unfinished_think": c["think"] and CLOSE_TAG not in txt,
                              "finish": co.finish_reason, "n_tokens": len(co.token_ids), "text": txt})
        with open(out / "free_samples.jsonl", "w") as f:
            for r in free_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(out / "samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    agg = aggregate(rows)
    wall = time.time() - t0
    meta = {"model": a.model, "model_name": name, "config_file": a.config, "config_sha16": sha16(a.config), "config": cfg,
            "seeds": seeds, "conditions": [c["name"] for c in conds], "n_items": len(items), "n_samples": len(rows),
            "sampling": {"temperature": temp, "top_p": top_p, "top_k": top_k, "stage2": "greedy, letter-restricted argmax", "cap": cap},
            "finish_counts": {f"{k[0]}|{k[1]}": v for k, v in finish_counts.items()},
            "free_answer": ({f"{b}|{cn}": {"n": len(v), "committed": sum(r["committed"] for r in v) / len(v),
                                            "hedge_words": sum(r["hedge_words"] for r in v) / len(v),
                                            "unfinished_think": sum(r["unfinished_think"] for r in v) / len(v)}
                             for (b, cn), v in _grp(free_rows).items()} if free_rows else None),
            "total_generated_tokens": sum(r["total_tokens"] for r in rows), "wall_seconds": wall,
            "vllm": vllm.__version__, "torch": torch.__version__, "python": platform.python_version(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "code_sha16": {p: sha16(Path(__file__).parent / p) for p in ("run_vllm.py", "common.py", "aggregate.py")},
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump({"meta": meta, "aggregate": agg}, open(out / "results.json", "w"), indent=1)
    print(f"[done] {len(rows)} samples, {wall:.0f}s -> {out}/results.json", flush=True)


if __name__ == "__main__":
    main()
