import json,textwrap,csv
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
ROOT=Path('/workspace/nextscene_artifacts');DEST=ROOT/'comparacoes';DEST.mkdir(exist_ok=True)
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):
 try:return ImageFont.truetype(fontpath,n)
 except OSError:return ImageFont.load_default()
def render(rows,headers,dest,cell=256,ids=None,title=None):
 header=84;rowbar=26;g=Image.new('RGB',(len(headers)*cell,header+len(rows)*(cell+rowbar)),(244,244,244));d=ImageDraw.Draw(g)
 d.text((12,5),title or '',font=font(15),fill='black')
 for j,h in enumerate(headers):d.multiline_text((j*cell+12,30),h,font=font(14),fill='black',spacing=4)
 for i,row in enumerate(rows):
  y=header+i*(cell+rowbar)
  d.text((10,y+4),(ids or [str(k) for k in range(len(rows))])[i],font=font(12),fill=(45,45,45))
  for j,im in enumerate(row):
   if im is None:continue
   im=im.copy();im.thumbnail((cell,cell));g.paste(im,(j*cell+(cell-im.width)//2,y+rowbar+(cell-im.height)//2))
 g.save(dest)
def open_rgb(p):
 with Image.open(p) as im:return im.convert('RGB')
manifest=json.load(open(ROOT/'data/heldout_short_manifest.json')); bystem={r['heldout_stem']:r for r in manifest}
order=['baseline_A','A250','A500','A750','A1000','baseline_B','B250','B500','B750','B1000']
metrics=[];groups=[]
for group in sorted((ROOT/'E1').iterdir(),key=lambda p:order.index(p.name) if p.name in order else 100):
 if not group.is_dir():continue
 for p in sorted(group.glob('*/metrics.json')):
  m=json.load(p.open());rows=[];ids=[]
  for r in m['pairs']:
   stem=r['stem'];w=r.get('width',512);h=r.get('height',512)
   a=ImageOps.fit(open_rgb('/workspace/heldout/control/'+stem+'.jpg'),(w,h));b=ImageOps.fit(open_rgb('/workspace/heldout/target/'+stem+'.jpg'),(w,h));out=open_rgb(p.parent/(stem+'_true.png'))
   rows.append([a,b,out]);ids.append(stem)
  config=json.load(open(p.parent/'eval_config.json'));label=('Base aligned / LoRA 0' if group.name=='baseline_A' else 'Base disjunto / LoRA 0' if group.name=='baseline_B' else ('Aligned' if group.name.startswith('A') else 'Disjunto')+' / step '+group.name[1:])
  render(rows,['A / referência','B / alvo real',label],DEST/(group.name+'_A_B_resultado.png'),ids=ids,title='512px, 20 steps, seed76, CFG4, ref_cfg1 — '+group.name)
  render(rows,['A / referência','B / alvo real',label],DEST/(group.name+'_A_B_resultado_full.png'),cell=512,ids=ids,title='Outputs individuais a512px — '+group.name)
  groups.append((group.name,rows,label));metrics.append(dict(experimento=group.name,**m['summary']))
if groups:
 first=groups[0][1];rows=[]
 for i in range(len(first)):
  rows.append(first[i][:2]+[r[i][2] for _,r,_ in groups])
 headers=['A / referência','B / alvo real']+[label.replace(' / ','\n') for _,_,label in groups]
 ids=[r['stem'] for r in json.load(open(next((ROOT/'E1'/groups[0][0]).glob('*/metrics.json'))))['pairs']]
 render(rows,headers,DEST/'TODOS_E1_A_B_resultados.png',ids=ids,title='E1 — todos os testes completos; mesma seed76 e mesmo prompt por linha')
 render(rows,headers,DEST/'TODOS_E1_A_B_resultados_full.png',cell=512,ids=ids,title='E1 — todos os testes completos, resolução integral512px')
# The one generation that validated the initial smoke.
sample=sorted((ROOT/'smoke/sample').glob('*.png'))
if sample:
 r=manifest[0];rows=[[ImageOps.fit(open_rgb('/workspace/heldout/control/'+r['heldout_stem']+'.jpg'),(512,512)),ImageOps.fit(open_rgb('/workspace/heldout/target/'+r['heldout_stem']+'.jpg'),(512,512)),open_rgb(sample[0])]]
 render(rows,['A / referência','B / alvo real','Smoke / step10'],DEST/'SMOKE10_A_B_resultado.png',cell=512,ids=[r['heldout_stem']],title='Smoke de integração: seed42, caption completa; não comparável ao ranking E1')
(DEST/'metrics.json').write_text(json.dumps(metrics,indent=2))
if metrics:
 with (DEST/'metrics.csv').open('w') as f:
  wr=csv.DictWriter(f,fieldnames=list(metrics[0]));wr.writeheader();wr.writerows(metrics)
print('Comparison grids:',len(groups),'E1 evaluations + smoke',flush=True)
