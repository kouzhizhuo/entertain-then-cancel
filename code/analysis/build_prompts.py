#!/usr/bin/env python3
"""VERBATIM PROMPTS (lead request): exp/results/PROMPTS/verbatim.json, one fully rendered prompt per format and arm, Qwen3 ChatML as the harness renders it (no system turn for Qwen3; the assistant prefix is part of the string).
User turns are rendered by the harness itself (common.render_user, identity option order); arm tails (thinking prefix, stop phrase, forced close, late checks) are copied from prompt_examples.txt of the actual runs. '[... thought ...]' marks where the sampled thought sits."""
import json, re, sys
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]; R = EXP / "results"; sys.path.insert(0, str(EXP / "harness"))
from common import load_items_from_config, load_set, render_user, apply_variant, HINTS, IST, LATE
def chatml(user, prefix=""): return f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{prefix}"
def examples(p):
    out = {}
    if not Path(p).exists(): return out
    for b in re.split(r"^===== ", open(p).read(), flags=re.M)[1:]:
        h, _, body = b.partition("\n"); out.setdefault(h.split(" | ")[0], (h, body))
    return out
def ex(group, model, cond, srcs=("runs", "<host>")):
    for s in srcs:
        e = examples(R / s / group / model / "prompt_examples.txt")
        if cond in e: return {"source": f"results/{s}/{group}/{model}/prompt_examples.txt", "condition": cond, "header": e[cond][0], "text": e[cond][1].rstrip()}
    return None
def first(cfgname, bucket):
    cfg = json.load(open(EXP / "configs" / f"{cfgname}.json")); its = [i for i in load_items_from_config(cfg) if i["bucket"] == bucket]; return its[0]
V = {}
yn = first("E1_sfx_think", "a3f_fel"); tw = first("E1_sfx_think", "a3f_nli"); fl = first("E1_probe", "x_a3q")
V["yes_no_inference"] = {"note": "Yes/No inference format (felicity framing), forced-choice letters; the arm tail is appended after the user turn.", "item": yn["id"], "user": render_user(yn, [0, 1], "default", None), "chatml_NAT_prefix": chatml(render_user(yn, [0, 1], "default", None), "<think>\n")}
V["three_way_choice"] = {"note": "polarity-free three-way format (NLI): the second statement follows / may be true or false / must be false", "item": tw["id"], "user": render_user(tw, list(range(len(tw["options"]))), "default", None), "chatml_NAT_prefix": chatml(render_user(tw, list(range(len(tw["options"]))), "default", None), "<think>\n")}
V["flipped_question"] = {"note": "polarity-flipped felicity question (yes-bias probe): literal answer Yes, pragmatic answer No", "item": fl["id"], "user": render_user(fl, [0, 1], "default", None), "chatml_NAT_prefix": chatml(render_user(fl, [0, 1], "default", None), "<think>\n")}
dev = load_set("method_dev", None)
for key, st in (("circa_multiple_choice", "circa"), ("pub_multiple_choice", "pub_task2")):
    it = next(r for r in dev if r["meta"]["set"] == st); u = render_user(it, list(range(len(it["options"]))), "default", None)
    V[key] = {"note": "multiple-choice format with the dataset's own options", "item": it["id"], "user": u, "chatml_NAT_prefix": chatml(u, "<think>\n")}
V["budget_stop_phrase"] = ex("E1_main", "Qwen3-8B", "think_b0") or {}
V["budget_stop_phrase"]["note"] = "think_b<N> (non-plain): the thought is cut at N tokens and this stop phrase is appended before the close tag; shown for N=0 (empty thought)"
V["trace_prefix_forced_close"] = ex("E1_main", "Qwen3-8B", "think_b256_plain") or {}
V["trace_prefix_forced_close"]["note"] = "think_b<N>_plain: the first N thinking tokens of the SAME trace, then the native close tag (no stop phrase)"
V["NT_off"] = ex("E1_main", "Qwen3-8B", "off") or {}; V["NT_off"]["note"] = "thinking switched off: empty thought block in the chat template"
V["forced_empty_NT"] = ex("E1_sfx", "Qwen3.5-4B", "think_b0_plain") or {}; V["forced_empty_NT"]["note"] = "forced-empty thinking (think_b0_plain): the native close tag directly after the opener"
V["NAT"] = ex("M1_dev_SC", "q3", "think_full") or ex("M1_dev_v2", "Qwen3-8B", "think_full") or {}
seed = ex("M1_dev", "Qwen3-8B", "seed_intent", ("<host>", "runs")) or {}; seed["seed_sentence"] = IST["m_intent"]; seed["note"] = "start-of-thinking seed (v1, 'intent'): the sentence opens the thinking block and the model continues from it"; V["remedy_seed"] = seed
V["remedy_late_check"] = ex("M1_dev_SC", "q3", "late_sitcheck", ("<host>", "runs")) or {}; V["remedy_late_check"]["late_text"] = LATE["sitcheck"]; V["remedy_late_check"]["note"] = "v2 SIT-check: appended after the natural thought INSIDE the thinking block (single close tag; run_vllm --single_close), joined with a blank line, at most 256 further thinking tokens"
V["remedy_neutral_check"] = ex("M1_dev_SC", "q3", "late_neutral", ("<host>", "runs")) or {}; V["remedy_neutral_check"]["late_text"] = LATE["neutral"]; V["remedy_neutral_check"]["note"] = "matched neutral control, same position and mechanics"
ta = render_user(yn, [0, 1], "default", "taskaware"); V["remedy_task_aware"] = {"note": "user-turn task-aware instruction (frozen strongest baseline)", "hint_text": HINTS["taskaware"], "user": ta, "chatml_NAT_prefix": chatml(ta, "<think>\n")}
V["_harness_note"] = "Caveat found 2026-10-07: in every run before the --single_close fix, a NATURAL trace was followed by a doubled close tag at stage 2 ('...\\n</think></think>\\n\\n'); capped traces had one. Examples marked source results/*/M1_dev_v2 or E1_main show this form."
(R / "PROMPTS").mkdir(exist_ok=True); json.dump(V, open(R / "PROMPTS/verbatim.json", "w"), indent=1, ensure_ascii=False); print({k: bool(v) for k, v in V.items() if not k.startswith("_")})
