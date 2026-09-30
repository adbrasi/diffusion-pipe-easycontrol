from pathlib import Path
import sys,csv,json,hashlib,re,os,random,statistics,toml
sys.path.insert(0,'tools')
from nextscene_captions import build_tiers
root=Path('/workspace/nextscene_artifacts/data');manifest=json.load(open('/workspace/heldout/manifest.json'));heldstems={r['stem'] for r in manifest}
groups={r['stem'].split('_image_')[0] for r in manifest if r['subset']=='ds1_recortados'}
heldhash={hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('/workspace/heldout').glob('*/*') if p.suffix in ['.jpg','.png']}
threshold={'ds1_recortados':.30,'ds2_poxima_v2':.25,'ds3_comikontext':.30,'ds4_contexto_curado':.20}
summary={};rng=random.Random(20260930);dirs=[];probe_dirs=[]
for sub,min_dino in threshold.items():
 p=root/(sub+'.csv');rows=list(csv.DictReader(p.open()));counts={};keep=[]
 for r in rows:
  cap=Path(r['target_caption']).read_text().strip() if r['target_caption'] else ''
  reason='keep'
  if r['stem'] in heldstems or r['stem'].split('_image_')[0] in groups:reason='heldout_group'
  elif any(hashlib.sha256(Path(r[k]).read_bytes()).hexdigest() in heldhash for k in ['target','control']):reason='heldout_sha'
  elif float(r['dhash_ham'])<=6 or float(r['pix_sim'])>=.97:reason='near_dup'
  elif float(r['dino_cos'])<min_dino:reason='unrelated'
  elif re.search(r'\b(title card|title screen|opening credits|closing credits|end credits|solid black|black screen|production logo|studio logo)\b',cap,re.I):reason='title_or_blank'
  elif not cap:reason='missing_caption'
  r['bucket']='keep' if reason=='keep' else reason;counts[reason]=counts.get(reason,0)+1
  if reason=='keep':keep.append(r)
 with (root/(sub+'_filtered.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 # Hardlink filtered originals; separate trees let probes cache only ~2K pairs.
 for tree,selected in [('/workspace/ns',keep),('/workspace/ns_probe',rng.sample(keep,min(512,len(keep))))]:
  out=Path(tree)/sub
  for r in selected:
   for field,dest in [('target','target'),('control','control'),('target_caption','target')]:
    src=Path(r[field]);dst=out/dest/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.link(src,dst)
  tiers=build_tiers(out/'target')
  (root/(sub+('_probe' if tree.endswith('_probe') else '')+'_captions.json')).write_text(json.dumps(tiers,indent=2))
 repeats=2 if sub.startswith('ds4') else 1
 dirs.append({'path':'/workspace/ns/'+sub+'/target','control_path':'/workspace/ns/'+sub+'/control','num_repeats':repeats})
 probe_dirs.append({'path':'/workspace/ns_probe/'+sub+'/target','control_path':'/workspace/ns_probe/'+sub+'/control','num_repeats':repeats})
 summary[sub]={'audit_pairs':len(rows),'min_dino':min_dino,'counts':counts,'final_caption_tiers':tiers,'repeats':repeats};summary[sub]['final_caption_tiers'].pop('samples',None)
 print(sub,min_dino,counts,flush=True)
dataset={'resolutions':[512],'enable_ar_bucket':True,'min_ar':.5,'max_ar':2.,'num_ar_buckets':7,'frame_buckets':[1]}
cfg=Path('examples/anima_nextscene/gpu_20260930')
for name,d in [('dataset_full.toml',dirs),('dataset_probe.toml',probe_dirs)]:
 dataset['directory']=d;(cfg/name).write_text(toml.dumps(dataset))
(root/'build_summary.json').write_text(json.dumps(summary,indent=2))
print('Build complete',flush=True)
