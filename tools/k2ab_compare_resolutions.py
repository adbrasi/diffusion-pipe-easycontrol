from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
import json
ROOT=Path('/workspace/k2ab')
ART=ROOT/'artifacts/fp8_512_micro2'
rows=json.loads((ART/'heldout_manifest_quick.json').read_text())
out=ART/'review';out.mkdir(exist_ok=True)
font=ImageFont.truetype('DejaVuSans.ttf',17)
small=ImageFont.truetype('DejaVuSans.ttf',15)
columns=['Referencia (imagem A)','Alvo real (imagem B)','Adapter A/native | 512','Adapter A/native | 1024','Adapter B/beta1 | 512','Adapter B/beta1 | 1024']
def build(steps,filename):
 cell,height,label,header=384,216,38,85
 canvas=Image.new('RGB',(cell*6,header+len(steps)*len(rows)*(height+label)),'white')
 draw=ImageDraw.Draw(canvas)
 draw.text((8,8),'Krea2 | treino512 micro2 | Turbo8 CFG1 | mesma seed76, prompt e referencia | apenas adapters treinados',fill='black',font=font)
 for col,text in enumerate(columns):draw.text((col*cell+7,45),text,fill='black',font=small)
 for idx,(step,row) in enumerate((step,row) for step in steps for row in rows):
  y=header+idx*(height+label)
  draw.text((8,y+5),f'Step {step} | {row["stem"]} | {row["prompt"]}',fill='black',font=small)
  paths=[ROOT/'heldout/control'/row['reference'],ROOT/'heldout/target'/row['reference']]
  for arm in ('A_native','B_beta1_fixed'):
   base=ART/'eval'/f'{arm}_step{step}'
   paths += [base/'Turbo'/f'{row["stem"]}_with_lora.png',base/'resolution_1024/Turbo'/f'{row["stem"]}_with_lora.png']
  for col,path in enumerate(paths):
   tile=ImageOps.contain(Image.open(path).convert('RGB'),(cell,height))
   canvas.paste(tile,(col*cell+(cell-tile.width)//2,y+label+(height-tile.height)//2))
 canvas.save(out/filename,quality=95)
 print(out/filename,flush=True)
if __name__ == '__main__':
 build((1000,),'A_B_step1000_Turbo_512_1024.jpg')
 build((250,500,750,1000),'A_B_steps250_500_750_1000_Turbo_512_1024.jpg')
