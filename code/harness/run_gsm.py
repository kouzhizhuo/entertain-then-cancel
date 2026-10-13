#!/usr/bin/env python
"""METHOD safety set GSM8K (free numeric answers) with the same arms as the forced-choice runs (METHOD_DESIGN v1): off, think_full (NAT), think_bN_plain (cap N then forced close), seed_* (SIT seeds / Grice prefix opened inside the thought), '+hint' user-turn instructions.
One deterministic decode per item (T=0.6, top_p 0.95, top_k 20, fixed per-item seed); tokens = generated tokens INCLUDING the seed. Grading: math_verify of the last \\boxed{} in the text after the thought; an unfinished thought counts wrong.
usage: python run_gsm.py --model <path> --out <dir> --split dev|test --conds off think_full seed_route ... [model-interface flags as in run_vllm.py]
Writes <out>/results.json (per condition: n, acc, tokens_mean, tokens_p95, unfinished_share) and samples.jsonl."""
import argparse, json, os, sys, time, zlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import HINTS, IST, LATE, parse_condition
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--split", default="test"); ap.add_argument("--conds", nargs="+", required=True)
ap.add_argument("--cap", type=int, default=8192); ap.add_argument("--tp", type=int, default=1); ap.add_argument("--gpu_util", type=float, default=0.88); ap.add_argument("--max_num_seqs", type=int, default=None)
ap.add_argument("--think_open", default="<think>\n"); ap.add_argument("--think_close", default="</think>"); ap.add_argument("--think_post", default="\n\n")
ap.add_argument("--think_kwarg", default="enable_thinking"); ap.add_argument("--sysflag", action="store_true"); ap.add_argument("--sys_on", default=None); ap.add_argument("--sys_off", default=None)
ap.add_argument("--system_file", default=None); ap.add_argument("--system_always", action="store_true"); ap.add_argument("--user_suffix_off", default=""); ap.add_argument("--off_prefix", default="")
ap.add_argument("--reasoning_effort", default=None); ap.add_argument("--trust_remote_code", action="store_true"); ap.add_argument("--keep_special", action="store_true"); ap.add_argument("--stop_token_ids", type=int, nargs="*", default=None)
ap.add_argument("--limit", type=int, default=None); ap.add_argument("--late_max", type=int, default=256)
ap.add_argument("--items_file", default=None, help="override the item file (SIT-train sampling uses data/items/sit_train_gsm.jsonl); --reps K repeats every item K times with distinct seeds (ids get #rK); --save_text stores thought+answer text in samples.jsonl")
ap.add_argument("--reps", type=int, default=1); ap.add_argument("--save_text", action="store_true")
a = ap.parse_args(); os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from math_verify import parse, verify
from vllm import LLM, SamplingParams
rows = [json.loads(l) for l in open(a.items_file or Path(__file__).resolve().parents[1] / "data" / "items" / f"method_gsm_{a.split}.jsonl")][: a.limit]
if a.reps > 1: rows = [{**r, "id": f"{r['id']}#r{k}", "orig_id": r["id"]} for k in range(a.reps) for r in rows]
kw = {"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}
def main():
    llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=a.gpu_util, dtype="bfloat16", seed=0, max_model_len=a.cap + 3072, enable_prefix_caching=True, trust_remote_code=a.trust_remote_code, **kw)
    tok = llm.get_tokenizer(); extra = {}
    if a.keep_special: extra["skip_special_tokens"] = False
    if a.stop_token_ids: extra["stop_token_ids"] = list(a.stop_token_ids)
    SYS = open(a.system_file).read().strip() if a.system_file else None
    OPEN, CLOSE = a.think_open, a.think_close
    SUF = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    def chat(user, think):
        user = user + (a.user_suffix_off if not think else "")
        msgs = [{"role": "user", "content": user}]; k = {}
        if SYS and (think or a.system_always): msgs = [{"role": "system", "content": SYS}] + msgs
        if a.sys_on: msgs = [{"role": "system", "content": a.sys_on if think else a.sys_off}] + msgs
        elif a.sysflag: msgs = [{"role": "system", "content": "/think" if think else "/no_think"}] + msgs
        elif not SYS: k = {a.think_kwarg: think}
        if think and a.reasoning_effort: k["reasoning_effort"] = a.reasoning_effort
        out = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **k)
        return out + (a.off_prefix if not think else "")
    def open_think(p): return p if p.rstrip().endswith(OPEN.strip()) else p + OPEN
    def sp(n, seedkey, stop=None): return SamplingParams(n=1, temperature=0.6, top_p=0.95, top_k=20, max_tokens=n, stop=stop, seed=zlib.crc32(seedkey.encode()) % 1_000_003, **extra)
    res, samples, t0 = {}, [], time.time()
    HAS_LATE = any(parse_condition(n)["late"] for n in a.conds); S1 = {}
    def stage1(hint, user, seed):
        """natural thought stopped at the close, shared by NAT / late_<key> arms of the same hint (branch from the SAME saved thought)"""
        if (hint, seed) not in S1:
            o1 = llm.generate([open_think(chat(u, True)) + seed for u in user], [sp(a.cap, r["id"] + "|s1" + str(hint), stop=[CLOSE]) for r in rows])
            S1[(hint, seed)] = (o1, [o.outputs[0].text for o in o1], [o.outputs[0].stop_reason == CLOSE for o in o1])
        return S1[(hint, seed)]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for name in a.conds:
        c = parse_condition(name); user = [r["problem"] + SUF + (("\n\n" + HINTS[c["hint"]]) if c["hint"] else "") for r in rows]
        seed = IST[c["seedtype"]] + " " if c["seedtype"] else ""; nseed = len(tok.encode(seed, add_special_tokens=False)) if seed else 0
        injtok = [0] * len(rows)
        if not c["think"]:
            outs = llm.generate([chat(u, False) for u in user], [sp(4096, r["id"] + name) for r in rows]); texts = [o.outputs[0].text for o in outs]; ntok = [len(o.outputs[0].token_ids) for o in outs]; unfinished = [False] * len(rows)
        elif c["late"]:                                # late check: NAT thought (stop at the close), then IST[key] inside the same thought, <= late_max more thinking tokens, then the answer
            o1, t1, done = stage1(c["hint"], user, seed); late = LATE[c["latekey"]]; nlate = len(tok.encode(late, add_special_tokens=False))
            pr_ = [open_think(chat(u, True)) + seed + t + "\n\n" + late + " " for u, t, d_ in zip(user, t1, done)]
            idx = [i for i, d_ in enumerate(done) if d_]; ext = {}
            if idx:
                o2 = llm.generate([pr_[i] for i in idx], [sp(a.late_max, rows[i]["id"] + "|late" + str(c["hint"]), stop=[CLOSE]) for i in idx])
                for i, o in zip(idx, o2): ext[i] = (o.outputs[0].text, len(o.outputs[0].token_ids))
            full = [(open_think(chat(u, True)) + seed + t + ("\n\n" + late + " " + ext[i][0] if i in ext else "") + "\n" + CLOSE + a.think_post) for i, (u, t) in enumerate(zip(user, t1))]
            o3 = llm.generate(full, [sp(1024, r["id"] + "|ans" + str(c["hint"])) for r in rows]); texts = [CLOSE + o.outputs[0].text for o in o3]
            ntok = [len(x.outputs[0].token_ids) + len(y.outputs[0].token_ids) + nseed + ((nlate + ext[i][1]) if i in ext else 0) for i, (x, y) in enumerate(zip(o1, o3))]; unfinished = [not d_ for d_ in done]; injtok = [(nlate + ext[i][1]) if i in ext else 0 for i in range(len(rows))]
        elif c["budget"] is None and HAS_LATE and not c["seedtype"]:   # NAT branch of the paired design: same saved stage-1 thought, readout without continuation
            o1, t1, done = stage1(c["hint"], user, seed)
            full = [open_think(chat(u, True)) + t + ("" if d_ else "\n") + CLOSE + a.think_post for u, t, d_ in zip(user, t1, done)]
            o3 = llm.generate(full, [sp(1024, r["id"] + "|ans" + str(c["hint"])) for r in rows]); texts = [CLOSE + o.outputs[0].text for o in o3]
            ntok = [len(x.outputs[0].token_ids) + len(y.outputs[0].token_ids) for x, y in zip(o1, o3)]; unfinished = [not d_ for d_ in done]
        elif c["budget"] is None:                      # NAT (+ optional seed)
            outs = llm.generate([open_think(chat(u, True)) + seed for u in user], [sp(a.cap + 2048, r["id"] + name) for r in rows])
            texts = [o.outputs[0].text for o in outs]; ntok = [len(o.outputs[0].token_ids) + nseed for o in outs]; unfinished = [CLOSE not in t for t in texts]
        else:                                          # plain cap N then forced close and a short answer
            pr = [open_think(chat(u, True)) + seed for u in user]
            if c["budget"] == 0:                       # forced-empty thought: no generation before the close
                class _O: text = ""; stop_reason = None; token_ids = []
                class _R: outputs = [_O()]
                o1 = [_R() for _ in rows]; t1 = [""] * len(rows)
            else:
                o1 = llm.generate(pr, [sp(c["budget"], r["id"] + name, stop=[CLOSE]) for r in rows]); t1 = [o.outputs[0].text for o in o1]
            pr2 = [p + t + ("" if o.outputs[0].stop_reason == CLOSE else "\n") + CLOSE + a.think_post for p, t, o in zip(pr, t1, o1)]
            o2 = llm.generate(pr2, [sp(1024, r["id"] + name + "|ans") for r in rows]); texts = [t + CLOSE + o.outputs[0].text for t, o in zip(t1, o2)]
            ntok = [len(x.outputs[0].token_ids) + len(y.outputs[0].token_ids) + nseed for x, y in zip(o1, o2)]; unfinished = [False] * len(rows)
        ok = []
        for r, t, n, u, ij in zip(rows, texts, ntok, unfinished, injtok):
            final = t.split(CLOSE)[-1] if (c["think"] and CLOSE in t) else ("" if c["think"] else t)
            good = bool(final) and not u and verify(parse(f"${r['answer']}$"), parse(final)); ok.append(good)
            samples.append({"id": r["id"], "cond": name, "correct": bool(good), "tokens": n, "inj_tokens": ij, "unfinished": bool(u), **({"text": t} if a.save_text else {})})
        tk = sorted(ntok)
        res[name] = {"n": len(rows), "acc": 100 * sum(ok) / len(ok), "tokens_mean": sum(ntok) / len(ntok), "inj_tokens_mean": sum(injtok) / len(injtok), "pre_tokens_mean": (sum(ntok) - sum(injtok)) / len(ntok), "tokens_p95": tk[int(0.95 * len(tk)) - 1], "unfinished_share": 100 * sum(unfinished) / len(rows), "seed_tokens": nseed}
        print(name, res[name], flush=True)
    json.dump({"model": a.model, "split": a.split, "n": len(rows), "results": res, "wall_seconds": time.time() - t0}, open(out / "results.json", "w"), indent=1)
    with open(out / "samples.jsonl", "w") as f:
        for s in samples: f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
