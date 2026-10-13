"""Move every \\caption{...} (+ following \\label) to the end of its table/figure float (caption below). Idempotent. usage: python3 tools/captions_below.py FILE..."""
import re, sys
def grab(s, i):  # s[i] == '{' -> index after matching '}'
    d = 0
    for j in range(i, len(s)):
        d += s[j] == '{'; d -= s[j] == '}'
        if d == 0: return j + 1
def fix(s):
    out, pos = [], 0
    for m in re.finditer(r'\\begin\{(table\*?|figure\*?)\}', s):
        if m.start() < pos: continue
        env = m.group(1); end = s.find('\\end{%s}' % env, m.end())
        if end < 0: continue
        body = s[m.end():end]; caps = []
        while True:
            c = re.search(r'\\caption\*?(\[[^\]]*\])?\{', body)
            if not c: break
            e = grab(body, c.end() - 1); cap = body[c.start():e]
            l = re.match(r'\s*\\label\{[^}]*\}', body[e:]); 
            if l: cap += l.group(0).strip() and '\n  ' + l.group(0).strip(); e += l.end()
            caps.append(cap); body = body[:c.start()].rstrip(' ') + body[e:].lstrip(' ')
        if caps:
            body = re.sub(r'\n\s*\n', '\n', body).rstrip() + '\n  ' + '\n  '.join(caps) + '\n'
        out.append(s[pos:m.end()] + body); pos = end
    out.append(s[pos:]); return ''.join(out)
for f in sys.argv[1:]:
    s = open(f).read(); t = fix(s)
    if t != s: open(f, 'w').write(t); print('fixed', f)
