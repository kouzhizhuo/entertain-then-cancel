"""Shared helpers for the P2 harness: item loading/sampling, prompt rendering, option permutation, letter parsing.

Everything that decides *what the model sees* lives here so that vLLM, Slurm and (optional) API runners share it.
"""
import hashlib, json, random, re
from collections import defaultdict
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
ITEMS_DIR = HARNESS.parent / "data" / "items"
LETTERS = "ABCDEFGH"

# soft-prompt modifiers appended to the user message (used as "+brief" / "+careful" suffixes of a condition name)
HINTS = {
    "maxims": "Interpret what the speaker says the way a cooperative speaker would intend it, following Grice's maxims of quantity, quality, relation and manner.",
    "brief": "Answer quickly, without much deliberation.",
    "careful": "Think carefully about what the speaker means before you answer.",
    "speaker": "Consider what the speaker meant.",                                  # method baseline: instruction in the user turn (METHOD_DESIGN prompt baseline 3)
    "sitroute": "First identify what the question asks: strict truth or entailment, or what the speaker intends. For intended meaning, use context and the speaker's goals, and infer unstated meaning only when the context supports it and nothing cancels it. For entailment, keep only what follows necessarily.",   # the exact SIT-route text as a user-turn instruction (placement control)
    "taskaware": 'Use context to infer what the speaker intends; when the question asks what strictly follows, answer that instead.',   # task-aware pragmatic instruction in the user turn (baseline)
    "pragcot": "Think step by step about what the speaker intends to convey and what the utterance implies before you answer.",   # pragmatic-CoT prompt baseline (wording is OUR paraphrase of Sato et al. 2025: UNVERIFIED)
}
INSTRUCTION = "Respond with only the letter of the best answer."
INSTRUCTION_NUM = "Respond with only the number of your answer."


def _get(d, path):
    for p in path.split("."):
        d = d[p]
    return d


def load_set(name, items_dir=None):
    p = Path(items_dir or ITEMS_DIR) / f"{name}.jsonl"
    return [json.loads(l) for l in open(p)]


def sample_set(rows, spec, rng):
    """spec: {"n": int} random sample | {"stratify": [field,...], "per_cell": k} | {} (all rows)."""
    if "stratify" in spec:
        cells = defaultdict(list)
        for r in rows:
            cells[tuple(_get(r, f) for f in spec["stratify"])].append(r)
        out = []
        for key in sorted(cells):
            c = cells[key]
            rng.shuffle(c)
            out += c[: spec["per_cell"]]
        return out
    if "n" in spec and spec["n"] < len(rows):
        rows = list(rows)
        rng.shuffle(rows)
        return rows[: spec["n"]]
    return list(rows)


def apply_prior(item):
    """No-utterance prior probe (E6): drop the premise / utterance lines from the context, keep the question and options. Items without
    a removable premise (e.g. category sentences) are returned unchanged and flagged prior_applicable=False."""
    it = dict(item); ctx = item.get("context", "")
    keep = [l for l in ctx.split("\n") if not l.startswith(("A speaker says:", "First sentence:", "Sentence:"))]
    it["context"] = "\n".join(keep); it["prior_applicable"] = len(keep) < len(ctx.split("\n"))
    it["id"] = item["id"]
    return it


# token-matched seeds for the bias-controlled matched test (Qwen3 tokenizer: 26 / 26 / 26 tokens); the E8 seeds below are NOT matched (neutral = 16 tokens vs 26) and are left unchanged
IST = {"m_intent": "Let me first work out what the speaker intends to communicate here, and what a cooperative listener would take away, before I decide.",
       "m_neutral": "Let me first work through the wording of this question and each answer option, one careful step at a time, before I decide.",
       "m_logic": "Let me first work out what strictly follows from what is said, and what a careful logician would conclude, before I decide.",
       "m_route": "First identify what the question asks: strict truth or entailment, or what the speaker intends. For intended meaning, use context and the speaker's goals, and infer unstated meaning only when the context supports it and nothing cancels it. For entailment, keep only what follows necessarily.",   # SIT-route (METHOD_DESIGN v1, text fixed)
       "m_task": 'First identify what the question asks: strict truth or entailment, or what the speaker intends.',   # task-check-only seed (ablation)
       "m_grice": "Let me interpret what the speaker says the way a cooperative speaker would intend it, following Grice's maxims of quantity, quality, relation and manner, and reason step by step about what is implied before I decide.",   # Grice/pragmatic CoT as a THINKING-PREFIX seed (baseline)
       "m_dual": "First, what does the speaker most likely intend to convey here? Then, does the question ask about that meaning or about what the words strictly entail?",   # SIT-dual (METHOD_DESIGN)
       "early": "Let me first work out what the speaker intends to communicate here, and what a cooperative listener would take away, before I decide.",
       "neutral": "Let me carefully and systematically work through this question step by step before I decide.",
       "late": "Before I answer: what did the speaker mean to convey, and what would a cooperative listener conclude?"}

# v2 late checks (notes/METHOD_DESIGN.md "AMENDMENT v2"): EXACT texts, appended inside the same thought after "\n\n"; separate from the v1 start-of-thinking IST seeds
LATE = {"late": IST["late"],      # v1 exploratory sentence (ist_late)
        "sitcheck": "Before I answer: does the question ask what the words strictly entail, or what the speaker means? If it asks what the speaker means, have I dropped a meaning the context supports, or added one it does not support? If it asks about strict entailment, keep only what follows necessarily.",
        "neutral": "Before I answer, let me look back over my reasoning once more. Have I read the question and every option carefully, kept track of each step, and avoided any slip in what I wrote above? If something looks wrong, I will fix it now; if not, I will keep my answer as it is."}


def apply_variant(item, variant):
    """Replace the item's framing by variants[variant] if it has one (else keep the default framing)."""
    v = (item.get("variants") or {}).get(variant) if variant not in (None, "default", "alt") else None
    it = dict(item)
    if variant == "felicityflip" and (item.get("variants") or {}).get("felicity"):
        # polarity-flipped felicity question (E1F yes-bias discriminator): same utterance and second statement, but the question asks whether the second statement is still LEFT OPEN,
        # so the literal answer is Yes and the pragmatic answer is No (the original question has pragmatic = Yes). A pure Yes-bias would now LOWER the pragmatic rate.
        v = dict(item["variants"]["felicity"]); v.update(question="After the speaker's statement, is it still left open whether the second statement is true? Answer Yes or No.",
                                                         options=["Yes", "No"], option_kinds=["literal", "pragmatic"], gold=1, literal=0)
    if v:
        it.update(v)
    it["variant_applied"] = variant if v else ("default" if variant != "alt" else "alt")
    return it


def _where_ok(row, where):
    for path, allowed in where.items():
        try:
            val = _get(row, path)
        except KeyError:
            return False
        if val not in allowed:
            return False
    return True


def load_items_from_config(cfg):
    rng = random.Random(cfg.get("item_seed", 1234))
    items = []
    for spec in cfg["sets"]:
        rows = load_set(spec["file"], cfg.get("items_dir"))
        if "role_filter" in spec:
            rows = [r for r in rows if r["role"] == spec["role_filter"]]
        if "where" in spec:  # {"meta.spec_relation": ["no_impl"]}
            rows = [r for r in rows if _where_ok(r, spec["where"])]
        rows = sample_set(rows, spec, rng)
        fs = fewshot_prefix(spec["fewshot"]["file"], spec["fewshot"]["k"], spec["fewshot"].get("seed", 0), cfg.get("items_dir")) if spec.get("fewshot") else None
        for r in rows:
            r["bucket"] = spec.get("bucket", spec["file"])
            if fs is not None: r["fewshot_prefix"] = fs
            if spec.get("load"): r["load_prefix"] = load_prefix(spec["load"]["genre"], spec["load"]["words"], r["id"])
        items += [(apply_prior(apply_variant(r, spec.get("variant", cfg.get("variant", "default")))) if spec.get("prior") else apply_variant(r, spec.get("variant", cfg.get("variant", "default")))) for r in rows]
        if spec.get("prior"):
            items = [i for i in items if i.get("prior_applicable", True) or i["bucket"] != spec.get("bucket", spec["file"])]
    return items


def labels_of(item):
    """Answer labels shown to the model: letters A.. by default, digits for 1-7 scales (item['labels'])."""
    return item.get("labels") or LETTERS[: len(item["options"])]


def is_pragmatic(item, choice):
    """Gold test: choice in item['gold_set'] if the item defines one (ternary: 'a bit wrong' or 'completely wrong'), else choice == gold."""
    if choice is None:
        return None
    gs = item.get("gold_set")
    return (choice in gs) if gs else (choice == item["gold"])


def perm_for(item_id, seed, k, shuffle=True):
    """Deterministic option permutation for (item, seed): perm[j] = canonical index shown at position j. Ordinal scales are never shuffled."""
    if not shuffle:
        return list(range(k))
    h = int(hashlib.sha256(f"{item_id}|{seed}".encode()).hexdigest()[:12], 16)
    p = list(range(k))
    random.Random(h).shuffle(p)
    return p


def fewshot_prefix(file, k, seed=0, items_dir=None):
    """k worked examples drawn (seeded) from a DEV file, formatted like the test prompt with the gold answer: yes/no implicature items (Ruis et al. style)."""
    rows = load_set(file, items_dir); random.Random(seed).shuffle(rows)
    ex = []
    for r in rows[:k]:
        ex.append(f"{r['context']}\n{r['question']}\nAnswer: {r['options'][r['gold']]}")
    return "Here are some solved examples.\n\n" + "\n\n".join(ex) + "\n\nNow the new case.\n" if k else ""


def load_prefix(genre, words, item_id, fill_dir=None):
    """E4 context load: `words` words of filtered public-domain filler (sentences in file order, window start fixed per item id), then a one-line reset."""
    sents = [l.strip() for l in open(Path(fill_dir or ITEMS_DIR.parent / "fillers") / f"{genre}.txt") if l.strip()]
    n = len(sents); h = int(hashlib.sha256(f"load|{item_id}".encode()).hexdigest()[:8], 16)
    out, w, i = [], 0, h % n
    while w < words:
        out.append(sents[i % n]); w += len(sents[i % n].split()); i += 1
    return " ".join(out) + "\n\n---\nThe text above is unrelated background material. Please ignore it and answer the following.\n"


def render_user(item, perm, variant="default", hint=None):
    q = item["question"]
    if variant == "alt" and item.get("question_alt"):
        q = item["question_alt"]
    lab = labels_of(item)
    parts = []
    if item.get("load_prefix"):
        parts.append(item["load_prefix"].rstrip())
    if item.get("fewshot_prefix"):
        parts.append(item["fewshot_prefix"].rstrip())
    if item.get("context"):
        parts.append(item["context"])
    parts.append(q)
    if item.get("scale"):      # 1-7 rating: the scale is in the question; answer is a digit
        instr = INSTRUCTION_NUM
    else:
        parts.append("Options:\n" + "\n".join(f"{lab[j]}. {item['options'][c]}" for j, c in enumerate(perm)))
        instr = INSTRUCTION
    parts.append(instr if not hint else f"{HINTS[hint]} {instr}")
    return "\n\n".join(parts)


def parse_condition(name):
    """'off' | 'think_b512' | 'think_full' | optional '+brief'/'+careful' suffix -> dict."""
    base, _, hint = name.partition("+")
    if hint and hint not in HINTS:
        raise ValueError(f"unknown hint {hint} in condition {name}")
    d = {"name": name, "hint": hint or None, "plain": False, "soft": False, "seedtype": None, "late": False, "latekey": None, "cot": False}
    if base == "off_cot":                  # visible prompted CoT without native thinking (<= 512 tokens), then "Answer:" readout
        return {**d, "think": False, "budget": 0, "cot": "step"}
    if base == "off_rat":                  # NT with a SHORT visible rationale (<= 80 tokens), then "Answer:" readout
        return {**d, "think": False, "budget": 0, "cot": "rat"}
    if base in ("seed_intent", "seed_neutral", "seed_logic", "seed_dual", "seed_route", "seed_task", "seed_grice"):   # token-matched thinking seeds (matched test)
        return {**d, "think": True, "budget": None, "seedtype": "m_" + base[5:]}
    if base in ("ist_early", "neutral_seed"):   # Intent-Seeded Thinking / length-matched neutral seed: thinking starts with a fixed sentence, then NAT
        return {**d, "think": True, "budget": None, "seedtype": "early" if base == "ist_early" else "neutral"}
    if base == "ist_late":                 # NAT, then a fixed sentence before </think> and <= 256 more thinking tokens
        return {**d, "think": True, "budget": None, "late": True, "latekey": "late"}
    if base.startswith("late_"):           # generic late check: NAT, then IST[<key>] appended inside the SAME thought (before the close) and <= late_max more thinking tokens
        if base[5:] not in LATE: raise ValueError(f"unknown late key {base!r}")
        return {**d, "think": True, "budget": None, "late": True, "latekey": base[5:]}
    if base == "off":                      # NT-empty: enable_thinking=False (empty think block from the template)
        return {**d, "think": False, "budget": 0}
    if base == "off_soft":                 # NT-soft: '/no_think' in the user turn, thinking template, empty think block
        return {**d, "think": False, "budget": 0, "soft": True}
    m = re.fullmatch(r"think_b(\d+)(_plain)?", base)
    if m:                                  # CAP-B (stop phrase) / CAP-plain-B (bare </think>)
        return {**d, "think": True, "budget": int(m.group(1)), "plain": bool(m.group(2))}
    if base == "think_full":               # NAT (cap = cfg.cap)
        return {**d, "think": True, "budget": None}
    raise ValueError(f"unknown condition {name}")


def letter_from_text(text, k, labels=None):
    """First standalone option label in a free-text answer (fallback parse), or None."""
    lab = labels or LETTERS[:k]
    m = re.search(rf"\b([{lab}])\b", text.strip()[:40])
    return lab.index(m.group(1)) if m else None


def sha16(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
