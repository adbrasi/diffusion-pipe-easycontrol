import datetime,json,os,re,time
from pathlib import Path
root=Path('/workspace/k2ab')
while True:
 now=datetime.datetime.now(datetime.timezone.utc)
 jobs=[]
 for state in sorted((root/'ops/jobs').glob('m2_*.job.state.json')):
  try:
   data=json.loads(state.read_text())
   if data.get('status')!='running': continue
   pid=data.get('pid')
   if pid:
    try:os.kill(pid,0)
    except ProcessLookupError:continue
   name=state.name.removesuffix('.job.state.json')
   log=root/'artifacts'/f'{name}.log'
   lines=log.read_text(errors='replace').splitlines() if log.exists() else []
   last=next((line for line in reversed(lines) if line.startswith('steps: ')),lines[-1] if lines else 'starting')
   jobs.append({'job':name,'progress':last})
  except (OSError, json.JSONDecodeError):
   continue
 campaign=json.loads((root/'artifacts/fp8_512_micro2/campaign_state.json').read_text())
 checkpoints={}
 for arm in ('A_native','B_beta1_fixed'):
  files=(root/'checkpoints'/f'{arm}_fp8_512_micro2_probe').glob('*/step*/adapter_model.safetensors')
  checkpoints[arm]=max([int(p.parent.name[4:]) for p in files] or [0])
 print(json.dumps({'utc':now.strftime('%H:%M:%S'),'jobs':jobs,'saved_steps':checkpoints,'campaign':campaign['status']},ensure_ascii=False),flush=True)
 if campaign['status'] in ('failed','stopped','awaiting_user_visual_verdict'):
  break
 time.sleep(45)
