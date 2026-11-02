#!/usr/bin/env python3
"""Similaridade A-B de todos os pares do BETA1-R (dHash 8x8 + MSE 64px). Uso: beta1r2_pairsim.py out.json
Saída: [[fonte, arquivo_alvo, dhash, mse64], ...] -> tools/build_beta1r2_pair_weights.py."""
import os, sys, json, random
from multiprocessing import Pool
from PIL import Image
import numpy as np
R='/workspace/data/beta1r'
def dhash(p, n=8):
    im=Image.open(p).convert('L').resize((n+1,n),Image.LANCZOS); a=np.asarray(im,dtype=np.int16)
    return (a[:,1:]>a[:,:-1]).flatten()
def small(p):
    im=Image.open(p).convert('RGB').resize((64,64),Image.BILINEAR); return np.asarray(im,dtype=np.float32)/255
def work(args):
    src,t,r=args
    try:
        d=int((dhash(t)!=dhash(r)).sum()); a,b=small(t),small(r); mse=float(((a-b)**2).mean())
        return src,os.path.basename(t),d,mse
    except Exception as e: return src,os.path.basename(t),-1,-1
jobs=[]
for src in sorted(os.listdir(R)):
    td=f'{R}/{src}/target'; rd=f'{R}/{src}/refs'
    if not os.path.isdir(td): continue
    for f in os.listdir(td):
        if not f.endswith('.jpg'): continue
        r=f'{rd}/{f[:-4]}_1.jpg'
        if os.path.exists(r): jobs.append((src,f'{td}/{f}',r))
with Pool(64) as p: res=p.map(work,jobs,chunksize=64)
json.dump(res,open(sys.argv[1],'w'))
import collections
by=collections.defaultdict(list)
for s,f,d,m in res:
    if d>=0: by[s].append((d,m))
for s,v in by.items():
    d=np.array([x[0] for x in v]); m=np.array([x[1] for x in v])
    print(f'{s:12s} n={len(d):5d} dHash med={np.median(d):4.1f} <=5:{(d<=5).mean():.3f} <=10:{(d<=10).mean():.3f} <=15:{(d<=15).mean():.3f} mse64 med={np.median(m):.4f} mse<0.005:{(m<0.005).mean():.3f}')
d=np.array([x[2] for x in res if x[2]>=0]); print('ALL',len(d),'<=5',(d<=5).mean(),'<=10',(d<=10).mean())
