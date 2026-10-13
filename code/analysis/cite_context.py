"""Context triage: citing sentences vs cited abstracts (Semantic Scholar / arXiv), keyword overlap; flags low-overlap pairs."""
import re, json, csv, os, time, glob, urllib.request, urllib.parse
C = "../notes/abstract_cache.json"; cache = json.load(open(C)) if os.path.exists(C) else {}
bib = open("custom.bib").read()
def title(k):
    m = re.search(r'@\w+\{' + re.escape(k) + r',(.*?)\n\}', bib, re.S); t = re.search(r'title\s*=\s*[{"](.+?)[}"]\s*,?\s*\n', m.group(1), re.S) if m else None
    return re.sub(r'[{}\\]|\s+', ' ', t.group(1)).strip() if t else ""
def abstract(k):
    if k in cache: return cache[k]
    t = title(k); a = ""
    for _ in range(3):
        try:
            r = urllib.request.urlopen(urllib.request.Request("https://api.semanticscholar.org/graph/v1/paper/search/match?query=" + urllib.parse.quote(t) + "&fields=title,abstract,tldr", headers={"User-Agent": "x"}), timeout=30)
            d = json.loads(r.read())["data"][0]; a = (d.get("abstract") or "") + " " + ((d.get("tldr") or {}).get("text") or ""); break
        except Exception: time.sleep(4)
    cache[k] = t + " " + a; json.dump(cache, open(C, "w")); time.sleep(1.1); return cache[k]
STOP = set("the a an of and or in on to for with is are be by as that this we our from at it its their more than which can not into thinking model models language".split())
toks = lambda s: {w for w in re.findall(r'[a-z][a-z\-]{3,}', s.lower()) if w not in STOP}
pairs = []
for f in sorted(glob.glob("sections/*.tex")):
    if os.path.basename(f) in ("A_remedies.tex", "5_sit.tex", "6_sit_results.tex", "5_origin.tex", "5_remedies.tex"): continue
    s = re.sub(r'(?m)^%.*$', '', open(f).read())
    for sent in re.split(r'(?<=[.;:])\s+', s):
        for m in re.finditer(r'\\cite[pt]?\{([^}]+)\}', sent):
            for k in m.group(1).split(","):
                pairs.append((k.strip(), f, re.sub(r'\\[a-zA-Z]+\{[^}]*\}|\s+', ' ', sent)[:300]))
out = []
for k, f, sent in pairs:
    a = abstract(k); ov = toks(sent) & toks(a); score = len(ov) / max(1, len(toks(sent)))
    out.append((k, f, round(score, 2), sorted(ov)[:8], sent))
out.sort(key=lambda x: x[2])
json.dump(out, open("../notes/cite_context_pairs.json", "w"), indent=1)
for r in out[:25]: print(r[2], r[0], "|", r[4][:150])
print(len(out), "pairs")
