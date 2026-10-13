#!/usr/bin/env python3
"""OPTIONAL API runner (reasoning-effort levels). UNTESTED: as of 2026-09-29 no working key exists (see logs/api_spend.md).

  python run_openai.py --config ../configs/api_effort.json --model gpt-5-mini --efforts minimal low medium high \
                       --out ../results/api/gpt-5-mini --cap_usd 25

Same items / permutations / prompt text as run_vllm.py (common.py), one request per (item, seed, effort) via the Responses API.
Answer = first option letter of the reply (forced choice by instruction; no logprobs are available for reasoning models).
Logged per sample: usage (input, output, reasoning tokens), finish/incomplete reason, seed, permutation.
Guards: refuses to run without a price entry (openai_prices.json), stops when the running spend reaches --cap_usd, and appends a line
to logs/api_spend.md. Only the environment key OPENAI_API_KEY (and OPENAI_BASE_URL if set by the user) is used.
"""
import argparse, concurrent.futures as cf, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import letter_from_text, load_items_from_config, perm_for, render_user
from aggregate import aggregate

LEDGER = Path(__file__).resolve().parents[2] / "logs" / "api_spend.md"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--efforts", nargs="+", default=["minimal", "low", "medium", "high"])
    ap.add_argument("--out", required=True); ap.add_argument("--cap_usd", type=float, default=25.0)
    ap.add_argument("--max_output_tokens", type=int, default=16000); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    prices = json.load(open(Path(__file__).with_name("openai_prices.json"))).get(a.model) or {}
    if not prices.get("input") or not prices.get("output"):
        sys.exit(f"no verified price for {a.model} in openai_prices.json - fill it in first (spend accounting is mandatory)")
    cfg = json.load(open(a.config)); items = load_items_from_config(cfg)[: a.limit]
    variant = cfg.get("variant", "default"); seeds = cfg["seeds"]
    jobs = [(it, s, e) for it in items for s in seeds for e in a.efforts]
    print(f"[api] {len(jobs)} requests, model={a.model}, efforts={a.efforts}")
    if a.dry:
        return
    from openai import OpenAI
    client = OpenAI()
    spent = {"usd": 0.0}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    def call(job):
        it, s, e = job
        if spent["usd"] >= a.cap_usd:
            return None
        perm = perm_for(it["id"], s, len(it["options"]))
        r = client.responses.create(model=a.model, input=render_user(it, perm, variant),
                                    reasoning={"effort": e}, max_output_tokens=a.max_output_tokens)
        u = r.usage
        rt = getattr(getattr(u, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        usd = (u.input_tokens * prices["input"] + u.output_tokens * prices["output"]) / 1e6
        spent["usd"] += usd
        pos = letter_from_text(r.output_text or "", len(perm)); valid = pos is not None
        choice = perm[pos] if valid else None
        return {"item_id": it["id"], "bucket": it["bucket"], "phenomenon": it["phenomenon"], "subset": it["subset"], "role": it["role"],
                "condition": f"effort_{e}", "seed": s, "perm": perm, "choice": choice, "valid": valid,
                "is_gold": (choice == it["gold"]) if valid else None,
                "is_literal": ((choice == it["literal"]) if valid else None) if it.get("literal") is not None else None,
                "choice_kind": it["option_kinds"][choice] if valid else None, "answer_text": r.output_text,
                "think_mode": True, "think_tokens": rt, "natural": r.status == "completed", "stage1_finish": r.status,
                "answer_tokens": u.output_tokens - rt, "total_tokens": u.output_tokens, "input_tokens": u.input_tokens, "usd": usd}

    rows = []
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for r in ex.map(call, jobs):
            if r:
                rows.append(r)
    with open(out / "samples.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"meta": {"model": a.model, "efforts": a.efforts, "seeds": seeds, "n_samples": len(rows), "usd": spent["usd"],
                        "cap_usd": a.cap_usd, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, "aggregate": aggregate(rows)},
              open(out / "results.json", "w"), indent=1)
    with open(LEDGER, "a") as f:
        f.write(f"| {time.strftime('%Y-%m-%d')} | {a.model} | {','.join(a.efforts)} | {len(rows)} | "
                f"{sum(r['input_tokens'] for r in rows)} | {sum(r['total_tokens'] for r in rows)} | "
                f"{sum(r['think_tokens'] for r in rows)} | {spent['usd']:.2f} | {out} |\n")
    print(f"[api] done, spent ${spent['usd']:.2f}")


if __name__ == "__main__":
    main()
