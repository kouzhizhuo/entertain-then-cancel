#!/usr/bin/env python3
"""Keep only the \\resdef lines whose key the paper references (directly via \\res{}, or via the
\\cell* table macros incl. their pilot fallbacks). Usage: python3 tools/filter_results.py full.tex out.tex"""
import re,sys,glob
used=set()
for f in glob.glob('sections/*.tex')+glob.glob('tables/*.tex'):
    t=open(f).read()
    used.update(re.findall(r'\\res\{([^}]+)\}',t)); used.update(re.findall(r'\\rv\{([^}]+)\}',t))
    for m in re.finditer(r'\\cell(pm|m|d|dinv|dpm|dpminv|sig)\{([^}]+)\}\{([^}]+)\}',t):
        kind,pre,k=m.groups()
        for p in (pre,'pilot'):
            for st in ('mean','std','lo','hi','sig'): used.add(f'{p}.{k}.{st}')
src,out=sys.argv[1],sys.argv[2]
keep=[l for l in open(src) if l.startswith('%') or (re.match(r'\\resdef\{([^}]+)\}',l) and re.match(r'\\resdef\{([^}]+)\}',l).group(1) in used)]
open(out,'w').writelines(keep); print(f'kept {len(keep)} lines of {sum(1 for _ in open(src))}')
