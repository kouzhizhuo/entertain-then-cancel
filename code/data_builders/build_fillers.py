#!/usr/bin/env python3
"""C2 load fillers for E4: two public-domain genres (Project Gutenberg #1342 Pride and Prejudice = fiction, #1228 On the Origin of Species = expository), body text only,
split into sentences, and every sentence containing a quantifier, numeral, negation, disjunction/conjunction scale term or modal is DROPPED so the filler cannot
prime scalar readings. Output data/fillers/{fiction,expository}.txt (one sentence per line) + manifest."""
import json, re
from pathlib import Path
H = Path(__file__).resolve().parent
STOP = set("""some all most many few several none no not never nobody nothing nowhere none neither either both each every any anyone anything one two three four five six seven eight nine ten
eleven twelve twenty thirty forty fifty hundred thousand million half quarter or must may might could can should would cannot some somewhat more less much
possible possibly probably certainly good great excellent warm hot cold cool large big huge small tiny likes like love hate""".split())
def sents(path, start_pat, end_pat):
    t = Path(path).read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    a = re.search(start_pat, t); b = re.search(end_pat, t)
    t = t[a.end(): b.start()] if a and b else t
    t = re.sub(r"\n\s*\n", " ¶ ", t); t = re.sub(r"\s+", " ", t)
    out = []
    t = re.sub(r"\b(Mr|Mrs|Ms|Dr|St|Messrs|Col|Capt|Lt|Sir|No|vs|etc|i\.e|e\.g|Fig|cf)\.", lambda m: m.group(0).replace(".", "§"), t)   # no sentence break after abbreviations
    for s in re.split(r"(?<=[.!?])\s+", t):
        s = s.replace("§", ".")
        s = s.replace("¶", "").strip(" _")
        w = re.findall(r"[A-Za-z']+", s.lower())
        if 6 <= len(w) <= 40 and not any(x in STOP or x.endswith("n't") for x in w) and not re.search(r"[0-9\[\]{}<>]", s) and s[:1].isupper():
            out.append(s)
    return out
man = {}
for name, f, sp, ep in (("fiction", "pg1342.txt", r"\*\*\* START OF.*?\*\*\*", r"\*\*\* END OF"), ("expository", "pg1228.txt", r"\*\*\* START OF.*?\*\*\*", r"\*\*\* END OF")):
    ss = sents(H / "raw" / "fillers" / f, sp, ep)
    (H / "fillers" / f"{name}.txt").write_text("\n".join(ss) + "\n")
    man[name] = {"source": f"Project Gutenberg {f[2:-4]}", "sentences": len(ss), "words": sum(len(s.split()) for s in ss), "licence": "public domain (US)"}
    print(name, man[name])
json.dump(man, open(H / "fillers" / "manifest.json", "w"), indent=1)
