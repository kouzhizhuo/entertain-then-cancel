#!/usr/bin/env python3
"""Frontier panel: closed models as experimental SUBJECTS (never judges) through the local LiteLLM proxy (user approval 2026-10-02; notes/FRONTIER_PANEL.md).
  python run_api.py selftest --models gpt-5 claude-3.7 ...         # one tiny call per (model, dose): verifies the ID and the thinking-control parameter
  python run_api.py run --config ../configs/E1f_items.json --model openai/gpt-5 --doses eff-min eff-low eff-med eff-high [--limit N] [--cap_usd 15]
Same prompts, permutations and letter scoring as the open models (common.py). One sample per (item, seed). Every call is appended to results/api/<model>/<dose>/samples.jsonl
(resumable; reasoning tokens from the usage fields; $ from the proxy's x-litellm-response-cost header) and each finished batch adds a row to logs/api_spend.md.
Doses: openai eff-min|eff-low|eff-med|eff-high (reasoning_effort); anthropic off|bud1k|bud4k|bud16k (thinking budget_tokens); google bud0|bud128|bud1k|bud8k (thinking budget)."""
import argparse, concurrent.futures as cf, json, os, sys, time, urllib.request, urllib.error
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import is_pragmatic, labels_of, letter_from_text, load_items_from_config, perm_for, render_user
EXP = Path(__file__).resolve().parents[1]; OUT = EXP / "results" / "api"; LEDGER = EXP.parent / "logs" / "api_spend.md"
BASE = "http://localhost:8000/v1/chat/completions"
EFF = {"eff-min": "minimal", "eff-low": "low", "eff-med": "medium", "eff-high": "high"}
BUD = {"off512": 0, "off": 0, "bud0": 0, "bud128": 128, "bud1k": 1024, "bud4k": 4096, "bud8k": 8192, "bud16k": 16384}

def body_for(model, dose, user):
    b = {"model": model, "messages": [{"role": "user", "content": user}]}
    fam = model.split("/")[0]
    if model.startswith("selfhosted/deepseek"):        # DeepSeek-V4 self-hosted endpoints: thinking toggled through chat_template_kwargs (verified 2026-10-05: flash-0731 default ON, pro default OFF)
        b["chat_template_kwargs"] = {"thinking": dose != "off"}; b["max_tokens"] = 16000 if dose != "off" else 512; return b
    if fam == "openai":
        b["max_completion_tokens"] = 16000; b["reasoning_effort"] = EFF[dose]
    elif fam == "anthropic" and dose in ("eff-low", "eff-high"):      # adaptive-thinking models (Opus 4.8): effort instead of a token budget
        b["reasoning_effort"] = EFF[dose]; b["max_tokens"] = 4000
    elif fam == "anthropic":
        n = BUD[dose]
        if n: b["thinking"] = {"type": "enabled", "budget_tokens": n}; b["max_tokens"] = n + 512
        else: b["max_tokens"] = 512 if dose == "off512" else 64   # off512 = same extraction, larger answer cap (artefact check: the 64-token cap truncated Claude answers that start with an explanation)
    elif fam == "google":
        n = BUD[dose]; b["max_completion_tokens"] = n + 512 if n else 512
        b["thinking"] = {"type": "enabled", "budget_tokens": n} if n else {"type": "disabled", "budget_tokens": 0}
    return b

def post(b, timeout=300, tries=4):
    """POST with backoff on rate limits / transient server errors (429, 5xx, timeouts)."""
    for k in range(tries):
        r = _post(b, timeout)
        if r[2] is None or not any(x in r[2] for x in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "timed out", "URLError")): return r
        time.sleep(4 * (k + 1) ** 2)
    return r


def _post(b, timeout=300):
    req = urllib.request.Request(BASE, data=json.dumps(b).encode(), headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"], "Content-Type": "application/json"})
    try:
        r = urllib.request.urlopen(req, timeout=timeout); hdr = {k.lower(): v for k, v in r.headers.items()}
        return json.load(r), hdr, None
    except urllib.error.HTTPError as e:
        return None, {}, f"HTTP {e.code}: {e.read().decode()[:300]}"
    except Exception as e:
        return None, {}, f"{type(e).__name__}: {str(e)[:200]}"

def parse(resp, hdr):
    ch = resp["choices"][0]; msg = ch["message"]; u = resp.get("usage", {}) or {}
    d = u.get("completion_tokens_details") or {}
    return {"text": msg.get("content") or "", "finish": ch.get("finish_reason"), "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "reasoning_tokens": d.get("reasoning_tokens"), "has_reasoning_content": bool(msg.get("reasoning_content") or msg.get("thinking_blocks")),
            "usd": float(hdr.get("x-litellm-response-cost", "nan")) if hdr.get("x-litellm-response-cost") else None, "model_returned": resp.get("model")}

PRICE_EST = {"openai/o4-mini": (1.10e-6, 4.40e-6)}   # USD per token (in, out): ESTIMATE, the proxy returns no cost for this model; flagged 'est.' in the ledger

def selftest(a):
    q = "Which option is a mammal?\nOptions:\nA. Salmon\nB. Elephant\nC. Sparrow\n\nRespond with only the letter of the best answer."
    plan = {"openai/gpt-5": list(EFF), "openai/gpt-5.4": ["eff-min", "eff-high"], "openai/o4-mini": ["eff-low", "eff-high"],
            "anthropic/claude-3.7-sonnet": ["off", "bud1k"], "anthropic/claude-sonnet-4.6": ["off", "off512", "bud1k"], "anthropic/claude-opus-4.8": ["off", "bud4k"],
            "google/gemini-2.5-flash": ["bud0", "bud1k"], "google/gemini-2.5-pro": ["bud128", "bud1k"]}
    for m, doses in plan.items():
        if a.models and not any(x in m for x in a.models): continue
        for d in doses:
            resp, hdr, err = post(body_for(m, d, q), 120)
            if err: print(f"{m:34s} {d:8s} ERROR {err[:220]}"); continue
            p = parse(resp, hdr); print(f"{m:34s} {d:8s} ok text={p['text'][:12]!r} finish={p['finish']} in={p['prompt_tokens']} out={p['completion_tokens']} reasoning={p['reasoning_tokens']} reasoning_content={p['has_reasoning_content']} usd={p['usd']} returned={p['model_returned']}")

def total_spend():
    t = 0.0
    for f in [f for d in OUT.parent.glob("api*") for f in d.rglob("samples.jsonl")]:
        for l in open(f):
            try: t += (json.loads(l).get("usd") or 0.0)
            except Exception: pass
    return t

def run(a):
    items = load_items_from_config(json.load(open(a.config)))[: a.limit]; seeds = json.load(open(a.config)).get("seeds", [0])
    cfg = json.load(open(a.config)); mdir = a.model.replace("/", "_"); root = (OUT.parent / a.out_root) if a.out_root else OUT
    for dose in a.doses:
        d = root / mdir / dose; d.mkdir(parents=True, exist_ok=True); f = d / "samples.jsonl"
        done = {(r["item_id"], r["seed"]) for r in map(json.loads, open(f))} if f.exists() else set()
        jobs = [(it, s) for it in items for s in seeds if (it["id"], s) not in done]
        print(f"[{a.model} {dose}] {len(jobs)} calls to do ({len(done)} done)", flush=True)
        stop = {"v": False}; lock = __import__("threading").Lock(); batch = {"n": 0, "usd": 0.0, "pt": 0, "ct": 0, "rt": 0, "err": 0}
        def call(job):
            it, s = job
            if stop["v"]: return None
            perm = perm_for(it["id"], s, len(it["options"]), it.get("shuffle", True)); user = render_user(it, perm, "default")
            resp, hdr, err = post(body_for(a.model, dose, user))
            if err:
                with lock: batch["err"] += 1
                return {"item_id": it["id"], "seed": s, "error": err}
            p = parse(resp, hdr)
            if p["usd"] is None and a.model in PRICE_EST: p["usd"] = (p["prompt_tokens"] or 0) * PRICE_EST[a.model][0] + (p["completion_tokens"] or 0) * PRICE_EST[a.model][1]; p["usd_est"] = True
            pos = letter_from_text(p["text"], len(perm), labels_of(it)); valid = pos is not None; choice = perm[pos] if valid else None
            row = {"item_id": it["id"], "bucket": it["bucket"], "phenomenon": it["phenomenon"], "subset": it["subset"], "role": it["role"], "split": it.get("split"), "variant": it.get("variant_applied"),
                   "model": a.model, "condition": dose, "seed": s, "perm": perm, "choice": choice, "valid": valid, "is_gold": is_pragmatic(it, choice) if valid else None,
                   "is_literal": ((choice == it["literal"]) if valid else None) if it.get("literal") is not None else None, "think_mode": True,
                   "think_tokens": p["reasoning_tokens"] if p["reasoning_tokens"] is not None else max((p["completion_tokens"] or 0) - 3, 0), "natural": p["finish"] in ("stop", "end_turn"),
                   "answer_text": p["text"][:200], "date": time.strftime("%Y-%m-%d"), "usd_est": bool(p.get("usd_est")), **{k: p[k] for k in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "usd", "finish", "model_returned")}}
            with lock:
                batch["n"] += 1; batch["usd"] += p["usd"] or 0; batch["pt"] += p["prompt_tokens"] or 0; batch["ct"] += p["completion_tokens"] or 0; batch["rt"] += p["reasoning_tokens"] or 0
                if batch["usd"] >= a.cap_usd: stop["v"] = True
            return row
        with open(f, "a") as fo, cf.ThreadPoolExecutor(a.workers) as ex:
            for r in ex.map(call, jobs):
                if r and "error" not in r: fo.write(json.dumps(r, ensure_ascii=False) + "\n"); fo.flush()
                elif r: print("  error:", r["error"][:160], flush=True)
        tot = total_spend()
        row = f"| {time.strftime('%Y-%m-%d')} | {a.model} | {dose} | {batch['n']} | {batch['pt']} | {batch['ct']} | {batch['rt']} | {batch['usd']:.3f} | results/{a.out_root or 'api'}/{mdir}/{dose} (errors {batch['err']}; cumulative ${tot:.2f}) |\n"
        with open(LEDGER, "a") as lf: lf.write(row)
        print(f"[{a.model} {dose}] done: {batch['n']} calls, ${batch['usd']:.3f} (cumulative ${tot:.2f})", flush=True)
        if tot >= a.global_cap: print("GLOBAL CAP reached; stopping"); break

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["selftest", "run"]); ap.add_argument("--models", nargs="*"); ap.add_argument("--config"); ap.add_argument("--model")
    ap.add_argument("--doses", nargs="*", default=[]); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out_root", default=None, help="results subdir instead of api (second-format checks: api_mc)"); ap.add_argument("--cap_usd", type=float, default=15.0, help="per (model, dose) batch"); ap.add_argument("--global_cap", type=float, default=150.0, help="stop when the ledger total exceeds this (first pass; $300 is the paper cap)")
    a = ap.parse_args(); selftest(a) if a.cmd == "selftest" else run(a)
