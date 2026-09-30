import json, shutil
from pathlib import Path
from PIL import Image, ImageOps
from make_comparisons import ROOT, DEST, render, open_rgb
items=[]
for group in sorted((ROOT/'E1').iterdir()):
    if not group.is_dir(): continue
    for p in group.glob('*/metrics.json'):
        m=json.load(p.open()); pairs=m['pairs']; cfg=json.load((p.parent/'eval_config.json').open())
        for tag, title in [('shuffled','A trocada / controle'), ('null','A zerada / controle')]:
            rows=[]
            for i,r in enumerate(pairs):
                w,h=r.get('width',512),r.get('height',512)
                a=(ImageOps.fit(open_rgb('/workspace/heldout/control/'+pairs[(i+1)%len(pairs)]['stem']+'.jpg'),(w,h)) if tag=='shuffled' else Image.new('RGB',(w,h),(210,210,210)))
                b=ImageOps.fit(open_rgb('/workspace/heldout/target/'+r['stem']+'.jpg'),(w,h))
                rows.append([a,b,open_rgb(p.parent/(r['stem']+'_'+tag+'.png'))])
            render(rows,[title,'B / alvo real',group.name+' / '+tag],DEST/(group.name+'_'+tag+'_A_B_resultado.png'),cell=512,ids=[r['stem'] for r in pairs],title='Mesmo prompt e seed76; null = latent zero, cinza apenas para indicar ausência de imagem')
        shutil.copy2(p.parent/'grid.png',DEST/(group.name+'_A_B_resultados_com_controles.png'))
        items.append((group.name,p.parent,cfg))
lines=['# Comparações — Anima nextscene','','A = imagem de referência. B = próximo frame real do dataset. Resultado = imagem gerada com o prompt de B.','',f'{len(items)} avaliações E1 completas, 12 pares por avaliação, mais o smoke inicial.','', '[Todos os testes lado a lado](TODOS_E1_A_B_resultados.png) · [Resolução completa](TODOS_E1_A_B_resultados_full.png)','','Os testes E1 usam a mesma seed76, os mesmos prompts curtos, 512×512, 20 steps, CFG4, ref_cfg1. Base = LoRA strength0. Aligned/disjunto identificam o layout RoPE. A e B estão recortadas como na avaliação.','','| Teste | A, B, resultado | Alta resolução | Com controles | A trocada | A zerada |','|---|---|---|---|---|---|']
for name,path,cfg in items:
    lines.append(f'| {name} | [grid]({name}_A_B_resultado.png) | [512px]({name}_A_B_resultado_full.png) | [5 colunas]({name}_A_B_resultados_com_controles.png) | [grid]({name}_shuffled_A_B_resultado.png) | [grid]({name}_null_A_B_resultado.png) |')
lines.extend(['','[Smoke de integração — step10](SMOKE10_A_B_resultado.png). Usou seed42 e caption completa; não entra no ranking E1.','','Benchmarks de batch/checkpointing testaram velocidade/memória e não produziram imagens. O smoke micro4 sem checkpointing deu OOM antes do primeiro step.','','Todas as gerações individuais, prompts, métricas e configurações estão em `../E1/<teste>/<checkpoint>/`. Os controles recebem a referência seguinte na lista (shuffled) ou o latent zero (null).','', '[Métricas CSV](metrics.csv) · [Métricas JSON](metrics.json)'])
(DEST/'README.md').write_text('\n'.join(lines)+'\n')
print('Complete comparison grids with controls:',len(items))
