#!/usr/bin/env python
"""Machine-translate the 160 IMPPRES quantifier/connective test sentence pairs (E1f_items buckets f_a3q, f_a3c) into German and Chinese with Qwen3-32B (non-thinking, greedy) for the non-English slice.
No ready non-English implicature set exists on the mirror (MultiPragEval absent, aisingapore set gated), so this is a MACHINE-TRANSLATED slice: disclose it, and note that the translator shares the Qwen3 family with the subject models.
usage: python translate_items.py --model <Qwen3-32B path> --out exp/data/translations   -> imppres_{de,zh}.jsonl ({item_id, lang, premise_t, hypothesis_t, ok})"""
import argparse, json, os, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_items_from_config
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tp", type=int, default=1)
a = ap.parse_args(); os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
from vllm import LLM, SamplingParams
items = load_items_from_config(json.load(open(Path(__file__).resolve().parents[1] / "configs" / "E1f_items.json")))
items = [i for i in items if i["bucket"] in ("f_a3q", "f_a3c")]
LANG = {"de": "German", "zh": "Chinese (Simplified)"}
llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=0.88, seed=0, max_model_len=2048); tok = llm.get_tokenizer()
def prompt(it, lang):
    p, h = it["meta"]["premise"], it["meta"]["hypothesis"]
    u = (f"Translate the two English sentences below into {LANG[lang]}. Keep the meaning exactly: translate quantifiers, connectives and modals literally (some, all, not all, no, or, and, can, have to, ...), keep each sentence a single natural sentence, "
         f"and do not add or drop information.\nReturn only JSON: {{\"s1\": \"...\", \"s2\": \"...\"}}.\n\ns1: {p}\ns2: {h}")
    return tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for lang in LANG:
    outs = llm.generate([prompt(i, lang) for i in items], SamplingParams(temperature=0.0, max_tokens=300))
    ok = 0
    with open(out / f"imppres_{lang}.jsonl", "w") as f:
        for it, o in zip(items, outs):
            m = re.search(r"\{.*\}", o.outputs[0].text, re.S); s1 = s2 = None
            try: d = json.loads(m.group(0)); s1, s2 = d["s1"].strip(), d["s2"].strip()
            except Exception: pass
            ok += bool(s1 and s2)
            f.write(json.dumps({"item_id": it["id"], "lang": lang, "premise_t": s1, "hypothesis_t": s2, "ok": bool(s1 and s2), "model": Path(a.model).name}, ensure_ascii=False) + "\n")
    print(lang, "translated", ok, "of", len(items), flush=True)
