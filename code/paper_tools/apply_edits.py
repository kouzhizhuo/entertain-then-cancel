"""Apply LiteLLM-proposed edits safely. Input JSON: [{"file": "sections/x.tex", "old": "...", "new": "...", "why": "..."}].
Rules: `old` must occur exactly once; \\res/\\cite/\\Cref/\\ref/\\label arguments in `new` must be a subset of those in `old`
plus (mode=review) any existing \\res key; mode=polish requires identical \\res, \\cite, \\Cref/\\ref, \\label multisets and identical digit tokens."""
import json, re, sys, collections
mode, src = sys.argv[1], sys.argv[2]
KEYS = set(re.findall(r'\\resdef\{([^}]+)\}', open("build/results_full.tex").read()))
def toks(s, pat): return collections.Counter(re.findall(pat, s))
PATS = {"res": r'\\res\{([^}]+)\}', "cite": r'\\cite[pt]?\{([^}]+)\}', "ref": r'\\[Cc]?ref\{([^}]+)\}', "label": r'\\label\{([^}]+)\}'}
raw = open(src).read(); dec = json.JSONDecoder(); items = []; i = raw.find('{')
while i != -1:
    try: obj, j = dec.raw_decode(raw, i); items.append(obj) if isinstance(obj, dict) and "old" in obj else None; i = raw.find('{', j)
    except json.JSONDecodeError: i = raw.find('{', i + 1)
ok = rej = 0
for e in items:
    f, old, new = e["file"], e["old"], e["new"]
    try: s = open(f).read()
    except OSError: print("REJ nofile", f); rej += 1; continue
    why = None
    if s.count(old) != 1: why = f"old occurs {s.count(old)}x"
    for n, p in PATS.items():
        a, b = toks(old, p), toks(new, p)
        if mode == "polish" and a != b: why = why or f"{n} changed"
        if mode == "review":
            extra = set(b) - set(a)
            if n == "res" and extra - KEYS: why = why or f"unknown res {extra - KEYS}"
            if n in ("cite", "label") and extra: why = why or f"new {n} {extra}"
    if mode == "polish" and toks(re.sub(PATS['res'], '', old), r'\d+(?:\.\d+)?') != toks(re.sub(PATS['res'], '', new), r'\d+(?:\.\d+)?'): why = why or "digits changed"
    if mode == "review" and set(re.findall(r'(?<![\w{.])\d+(?:\.\d+)?', re.sub(PATS['res'], '', new))) - set(re.findall(r'(?<![\w{.])\d+(?:\.\d+)?', re.sub(PATS['res'], '', old))) - set("0123456789"):
        why = why or "new bare number"
    if why: print("REJ", f, why, "|", old[:70].replace("\n", " ")); rej += 1; continue
    open(f, "w").write(s.replace(old, new, 1)); ok += 1
print(f"applied {ok}, rejected {rej}")
