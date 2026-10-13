import json,re,math,collections,sys
def bad(c): raise ValueError('NaN/Inf token '+c)
d=json.load(open(sys.argv[1]),parse_constant=bad)
KEY=re.compile(r'^[A-Za-z0-9._-]+$')
def flat(o,p=''):
    if isinstance(o,dict):
        for k,v in o.items(): yield from flat(v,f'{p}.{k}' if p else k)
    elif isinstance(o,list):
        for i,v in enumerate(o): yield from flat(v,f'{p}.{i}')
    else: yield p,o
F=dict(flat(d))
print('keys',len(F),'bad chars',sum(not KEY.match(k) for k in F),'None',sum(v is None for v in F.values()))
MODELS={'qwen3-4b','qwen3-8b','qwen3-14b','qwen3-4b-t07','qwen3-8b-t07'}
FMT={'tf','fel','tern','grad','inf','nli','mc','yn','open','tint','lp'}
COND_RE=re.compile(r'^(off|offsoft|b0|c\d{4}|p\d{4}|nat)$')
METRIC={'R','lit','hedge','L','acc','les','tokens','dR','dacc','dlit'}
STAT={'mean','std','lo','hi','n'}
ext=collections.Counter()
for k in F:
    if k.startswith('_meta'): continue
    p=(k[:-8] if k.endswith('__digits') else k).split('.')
    m,s,f,c,me,st=p
    if m not in MODELS: ext['model:'+m]+=1
    if not re.match(r'^(a1|a1-ctrl|a3(-\w+)?|a3r-\w+|b1-indirect|b2)$',s): ext['set:'+s]+=1
    elif s in('a1-ctrl','a3-ctrl') or s.startswith('a3r-'): ext['set(ext):'+s.split('-')[0]+'-*']+=1
    if f not in FMT: ext['format:'+f]+=1
    if not COND_RE.match(c): ext['cond:'+c]+=1
    if me not in METRIC: ext['metric:'+me]+=1
    if st not in STAT: ext['stat:'+st]+=1
print('outside/extending RESULTS_KEYS vocabulary:',dict(ext))
rate=[(k,v) for k,v in F.items() if re.search(r'\.(R|lit|acc|les|hedge)\.(mean|lo|hi)$',k)]
print('rates outside 0-100:',[k for k,v in rate if not 0<=v<=100][:5])
print('lo>hi:',[k for k in F if k.endswith('.lo') and (k[:-3]+'.hi') in F and F[k]>F[k[:-3]+'.hi']+1e-9][:5])
out=[k for k,v in F.items() if re.search(r'\.(R|lit|acc)\.mean$',k) and not (F[k[:-5]+'.lo']-1e-6<=v<=F[k[:-5]+'.hi']+1e-6)]
print('mean outside its own CI:',len(out),out[:3])
