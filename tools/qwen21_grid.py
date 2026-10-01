#!/usr/bin/env python3
"""Compose a labeled scientific contact sheet of existing native outputs."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps, ImageFont
import numpy as np

def main():
    root=Path('/workspace/qwen21')
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',default='smoke10')
    args=parser.parse_args()
    heldout=json.loads((root/'heldout.json').read_text())
    cell_w,cell_h=384,256
    titles=['Cena A - referencia','Cena B - alvo','Base - referencia certa',
            args.stage+' - referencia certa','Base - referencia trocada']
    canvas=Image.new('RGB',(cell_w*5,2*(cell_h+32)),(245,245,245))
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf',16)
    metrics=[]
    for row in range(2):
        source=heldout[row]
        paths=[Path(source['edit_image']),Path(source['image']),
               root/f'samples/base/heldout{row:02}_true.png',
               root/f'samples/{args.stage}/heldout{row:02}_true.png',
               root/'samples/base/heldout00_shuffled.png' if row==0 else None]
        for col,path in enumerate(paths):
            y=row*(cell_h+32)
            draw.text((col*cell_w+8,y+8),titles[col],fill=(15,15,15),font=font)
            if path:
                image=Image.open(path).convert('RGB')
                image=ImageOps.contain(image,(cell_w,cell_h))
                canvas.paste(image,(col*cell_w+(cell_w-image.width)//2,y+32+(cell_h-image.height)//2))
        base=np.asarray(Image.open(paths[2]).convert('RGB'),dtype=np.float32)
        smoke=np.asarray(Image.open(paths[3]).convert('RGB'),dtype=np.float32)
        metrics.append(dict(heldout_index=row,stage=args.stage,base_vs_adapter_rgb_mean_abs=float(abs(base-smoke).mean()),
                            note='Execution/learning smoke, not a visual-quality conclusion'))
    destination=root/'samples'/('grid_smoke.png' if args.stage=='smoke10' else 'grid_'+args.stage+'.png')
    canvas.save(destination)
    (root/'samples'/(('smoke' if args.stage=='smoke10' else args.stage)+'_pixel_comparison.json')).write_text(json.dumps(metrics,indent=2))
    print(destination)

if __name__=='__main__': main()
