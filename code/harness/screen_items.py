#!/usr/bin/env python
"""Open-weight co-rater screening (TEAM_RULES 1.7): Qwen3-32B screens every A1 triple (80 incl. the 35 new) and every A1-ctx scenario (50) against the written criteria.
  python screen_items.py --model <Qwen3-32B-FP8 path> --out exp/gold
Output: item_screen_qwen32b.jsonl (id, kind, verdict OK|FIX|DROP, reason, fix, raw) and item_screen_meta.json (model, date, sampling, the exact prompt templates).
Criteria text = notes/ITEM_VALIDATION.md (A1 criteria 1-4) + criterion 3' and A1_CTX_ITEMS.md section 6 (scenario checklist). Thinking mode, T=0.6, one pass (seed 0)."""
import argparse, json, re, time, zlib
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
TRIPLE_PROMPT = """You are screening one stimulus item for a psycholinguistics experiment on scalar implicature ("some" implicating "not all").

The item is a category triple. X = "{x}", Y = "{y}", P = "{p}", Z = "{z}". The generated sentences are:
 T1 (critical, underinformative): Some {x} are {y}.
 T2: Some {y} are {x}.   (must be true and informative)
 T3: Some {x} are {z}.   (must be false)
 T4: All {x} are {y}.    (must be true)
 T5: All {y} are {x}.    (must be false)
 T6: All {x} are {z}.    (must be false)
 T0: Some {x} {p}.       (must be true and informative)

Criteria:
1. "All {x} are {y}" is true without common exceptions (so "Some {x} are {y}" is underinformative).
2. "Some {x} {p}" is true and informative: clearly partial, not near-universal.
3. "Some {x} are {z}" is false under every common sense of the words. 3'. It must also be false under cladistic taxonomy (birds are reptiles; tetrapods are lobe-finned fish), so Z is never "reptiles" or "fish" for a bird, mammal or other tetrapod X.
4. No lexical ambiguity changes any of these (e.g. a second common sense of a word that makes a sentence true or false).
Also check T2, T5 and T6 against the same standard.

Answer with ONLY a JSON object: {{"verdict": "OK" | "FIX" | "DROP", "reason": "<one sentence naming the failed criterion, or why it passes>", "fix": "<replacement for the failing element, or empty>"}}"""
SCEN_PROMPT = """You are screening one stimulus item for a psycholinguistics experiment on speaker knowledge and scalar implicature (a Goodman & Stuhlmueller style design).

A speaker says "Some of the {pl} {prop}". The scenario has two versions.
FULL access: {full}
PARTIAL access: {part}
Questions: (felicity) "{fq}"  (inference) "{iq}"
Intended answers: under FULL access the pragmatic answer is No (the speaker could have said "all", so "some" is misleading / implies not all); under PARTIAL access the pragmatic answer is Yes (two checked, "some" is the strongest thing the speaker can say).

Checklist:
1. The property is contingent and observable by "checking": it is not a category fact and not unknowable.
2. The full-access context makes "all" the strongest true description; the partial context gives no hint about the unchecked objects.
3. The utterance is grammatical and natural; the number words and "the other k" are correct.
4. There is no world-knowledge prior that makes "all" implausible independently of the utterance (e.g. "all eight seats are taken" is plausible; check this case).
5. Keep the item only if it passes all four.

Answer with ONLY a JSON object: {{"verdict": "OK" | "FIX" | "DROP", "reason": "<one sentence naming the failed checklist item, or why it passes>", "fix": "<suggested change, or empty>"}}"""
def load(n): return [json.loads(l) for l in open(EXP / "data" / "items" / f"{n}.jsonl")]
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tp", type=int, default=1); a = ap.parse_args()
jobs = []
bn = {}
for r in load("bn_underinformative"):
    if r["subset"] == "underinformative": bn[r["meta"]["pair_index"]] = r
for k, r in sorted(bn.items()):
    m = r["meta"]; jobs.append((f"a1.triple.{k}", "triple", TRIPLE_PROMPT.format(x=m["category"], y=m["y"], p=m["partial"], z=m["z"]), {"pair_index": k, "cohort": m["cohort"], "triple": [m["category"], m["y"], m["partial"], m["z"]]}))
fel = {(r["meta"]["scenario"], r["meta"]["access"]): r for r in load("a1ctx_felicity") if r["subset"] in ("felicity_full", "felicity_partial")}
inf_q = {r["meta"]["scenario"]: r["question"] for r in load("a1ctx_inference") if r["subset"] == "inference_full"}
for sc in range(50):
    f, p = fel[(sc, "full")], fel[(sc, "partial")]; m = f["meta"]
    jobs.append((f"a1ctx.scenario.{sc}", "scenario", SCEN_PROMPT.format(pl=m["noun"], prop=m["property"], full=f["context"], part=p["context"], fq=f["question"], iq=inf_q[sc]), {"scenario": sc}))
from vllm import LLM, SamplingParams
llm = LLM(a.model, tensor_parallel_size=a.tp, gpu_memory_utilization=0.88, seed=0, max_model_len=8192); tok = llm.get_tokenizer()
prompts = [tok.apply_chat_template([{"role": "user", "content": j[2]}], tokenize=False, add_generation_prompt=True, enable_thinking=True) for j in jobs]
outs = llm.generate(prompts, [SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=4096, seed=zlib.crc32(j[0].encode()) % 1_000_003) for j in jobs])
out = Path(a.out); out.mkdir(parents=True, exist_ok=True); n_ok = 0
with open(out / "item_screen_qwen32b.jsonl", "w") as f:
    for j, o in zip(jobs, outs):
        t = o.outputs[0].text; final = t.split("</think>")[-1]
        m = re.findall(r"\{.*?\}", final, re.S); v = {}
        for c in reversed(m):
            try: v = json.loads(c); break
            except Exception: continue
        verdict = str(v.get("verdict", "")).upper(); verdict = verdict if verdict in ("OK", "FIX", "DROP") else "UNPARSED"
        f.write(json.dumps({"id": j[0], "kind": j[1], "verdict": verdict, "reason": v.get("reason", ""), "fix": v.get("fix", ""), "meta": j[3], "raw": final.strip()[:600]}, ensure_ascii=False) + "\n")
json.dump({"rater": "LLM screening co-rater (open-weight)", "model": Path(a.model).name, "date": time.strftime("%Y-%m-%d"), "sampling": "thinking mode, T=0.6 top_p 0.95 top_k 20, max 4096 tokens, one pass",
           "n_triples": sum(j[1] == "triple" for j in jobs), "n_scenarios": sum(j[1] == "scenario" for j in jobs), "criteria_source": "notes/ITEM_VALIDATION.md (A1 criteria 1-4, 3'), notes/A1_CTX_ITEMS.md section 6",
           "triple_prompt": TRIPLE_PROMPT, "scenario_prompt": SCEN_PROMPT}, open(out / "item_screen_meta.json", "w"), indent=1)
print("screened", len(jobs))
