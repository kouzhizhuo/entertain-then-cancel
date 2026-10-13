"""P1-style table colouring (idempotent). Delta cells: vermilion (cInterv) if negative, blue (cOurs) if positive, log-scale tint.
Level cells (\\cellpm, accuracy/rate \\res): light orange (cCmpA) tint by value. Tables without numeric cells: light grey header row.
usage: python3 tools/style_tables.py FILE...  (run from paper/, after numbers/results.tex exists)"""
import re, math, sys
V = {}
for k, v in re.findall(r'\\resdef\{([^}]+)\}\{([^}]*)\}', open("numbers/results.tex").read()):
    try: V[k] = float(v.replace("$", "").replace("\\text{-}", "-").replace("−", "-").replace("{,}", ""))
    except ValueError: pass
def key_of(cell):
    m = re.search(r'\\res\{([^}]+\.mean)\}', cell)
    if m: return m.group(1)
    m = re.search(r'\\cell(dpm|d|pm|m)\{([^}]+)\}\{([^}]+)\}', cell)
    if m: return f"{m.group(2)}.{m.group(3)}.mean"
def tint(cell):
    if "\\cellcolor" in cell or "\\multicolumn" in cell: return cell
    k = key_of(cell)
    if not k or k not in V: return cell
    v = V[k]; metric = k.split(".")[-2]
    if metric.startswith("d"):
        a = int(round(min(25, 3 + 22 * math.log1p(abs(v)) / math.log1p(80))))
        col = f"{'cInterv' if v < 0 else 'cOurs'}!{a}"
    else:
        if not (0 <= v <= 100): return cell
        col = f"cCmpA!{int(round(2 + 13 * v / 100))}"
    lead = re.match(r'\s*', cell).group(0)
    return f"{lead}\\cellcolor{{{col}}}{cell[len(lead):]}"
def style(s):
    out, pos = [], 0
    for m in re.finditer(r'\\begin\{(tabular\*?|tabularx)\}', s):
        if m.start() < pos: continue
        env = m.group(1); end = s.find(f"\\end{{{env}}}", m.end())
        if end < 0: continue
        body = s[m.end():end]; L = body.split("\n"); first_mid = next((i for i, l in enumerate(L) if "\\midrule" in l), None)
        numeric = False
        for i, l in enumerate(L):
            if first_mid is None or i <= first_mid: continue
            if l.rstrip().endswith("\\\\") and "&" in l:
                c = l.rstrip()[:-2].split("&"); new = [c[0]] + [tint(x) for x in c[1:]]
                numeric |= new != c; L[i] = "&".join(new) + "\\\\"
        if not numeric and not any("\\cellcolor" in l for l in L) and first_mid:
            h = max(i for i in range(first_mid) if L[i].rstrip().endswith("\\\\")) if any(L[i].rstrip().endswith("\\\\") for i in range(first_mid)) else None
            if h is not None and "\\rowcolor" not in L[h]: L[h] = re.sub(r'^(\s*)', r'\1\\rowcolor{gray!8}', L[h], count=1)
        out.append(s[pos:m.end()] + "\n".join(L)); pos = end
    out.append(s[pos:]); return "".join(out)
n = 0
for f in sys.argv[1:]:
    s = open(f).read(); t = style(s)
    if t != s: open(f, "w").write(t); n += 1; print("styled", f)
print(n, "files")
