"""Verify every cited bib entry against arXiv / Crossref / ACL Anthology / Semantic Scholar. Writes notes/CITATION_AUDIT_20261013.csv."""
import re, json, csv, sys, time, difflib, urllib.request, urllib.parse, xml.etree.ElementTree as ET
keys = open(sys.argv[1]).read().split()
bib = open("custom.bib").read()
def field(b, f):
    m = re.search(r'\b' + f + r'\s*=\s*[{"](.+?)[}"]\s*,?\s*\n', b, re.S | re.I)
    return re.sub(r'[{}\\]|\s+', ' ', m.group(1)).strip() if m else ""
def get(url, hdr=None):
    for t in range(3):
        try:
            r = urllib.request.Request(url, headers=hdr or {"User-Agent": "cite-audit/1.0 (mailto:<email>)"})
            return urllib.request.urlopen(r, timeout=40).read().decode("utf-8", "ignore")
        except Exception as e:
            time.sleep(3 * (t + 1)); err = str(e)
    return None
norm = lambda s: re.sub(r'[^a-z0-9 ]', '', s.lower().replace('-', ' ')).split()
sim = lambda a, b: difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio()
rows = []
for k in keys:
    m = re.search(r'@\w+\{' + re.escape(k) + r',(.*?)\n\}', bib, re.S)
    if not m: rows.append([k, "MISSING-IN-BIB", "", "", ""]); continue
    b = m.group(1); title, author, year = field(b, "title"), field(b, "author"), field(b, "year")
    first = re.split(r',| and ', author)[0].split()[-1] if "," not in author.split(" and ")[0] else author.split(" and ")[0].split(",")[0].strip()
    ax = re.search(r'(?:eprint|arxiv|journal|url|note)\s*=\s*[{"][^}"]*?(\d{4}\.\d{4,5})', b, re.I)
    doi = re.search(r'doi\s*=\s*[{"]([^}"]+)', b, re.I); url = field(b, "url")
    got, src = None, ""
    if "aclanthology.org/" in url:
        a = url.rstrip("/").split("aclanthology.org/")[-1].replace(".pdf", "")
        t = get(f"https://aclanthology.org/{a}.bib")
        if t:
            gt = re.search(r'title\s*=\s*"(.+?)"', t, re.S); gy = re.search(r'year\s*=\s*"(\d+)"', t); ga = re.search(r'author\s*=\s*"(.+?)"', t, re.S)
            got = (re.sub(r'[{}]', '', gt.group(1)) if gt else "", gy.group(1) if gy else "", ga.group(1) if ga else ""); src = f"https://aclanthology.org/{a}"
    if not got and doi:
        t = get("https://api.crossref.org/works/" + urllib.parse.quote(doi.group(1)))
        if t:
            j = json.loads(t)["message"]; got = ((j.get("title") or [""])[0], str((j.get("issued", {}).get("date-parts") or [[""]])[0][0]), " ".join(a.get("family", "") for a in j.get("author", [])[:1])); src = "https://doi.org/" + doi.group(1)
    if not got and ax:
        t = get("http://export.arxiv.org/api/query?id_list=" + ax.group(1))
        if t:
            e = ET.fromstring(t).find("{http://www.w3.org/2005/Atom}entry")
            if e is not None and e.find("{http://www.w3.org/2005/Atom}title") is not None:
                got = (" ".join(e.find("{http://www.w3.org/2005/Atom}title").text.split()), e.find("{http://www.w3.org/2005/Atom}published").text[:4], e.find("{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name").text); src = "https://arxiv.org/abs/" + ax.group(1)
    if not got:
        t = get("https://api.semanticscholar.org/graph/v1/paper/search/match?query=" + urllib.parse.quote(title) + "&fields=title,year,authors,externalIds,venue")
        if t:
            try:
                d = json.loads(t)["data"][0]; got = (d["title"], str(d.get("year")), (d.get("authors") or [{"name": ""}])[0]["name"]); src = "https://www.semanticscholar.org/paper/" + d["paperId"]
            except Exception: pass
        time.sleep(1.2)
    if not got: rows.append([k, "UNVERIFIED", title, "", ""]); continue
    s_t = sim(title, got[0]); fa_ok = first.lower()[:5] in got[2].lower() if first else True
    status = "OK" if s_t > 0.9 and fa_ok and (not year or not got[1] or abs(int(year) - int(got[1])) <= 1) else "CHECK"
    rows.append([k, status, src, f"title_sim={s_t:.2f} first={first}/{got[2][:25]} year={year}/{got[1]}", got[0][:90]])
    print(k, status, f"{s_t:.2f}", year, got[1], first, "|", got[2][:20], flush=True)
with open("../notes/CITATION_AUDIT_20261013.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["key", "status", "evidence_url", "checks", "matched_title"]); w.writerows(rows)
