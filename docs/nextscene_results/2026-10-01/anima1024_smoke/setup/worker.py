import os,json,time,subprocess,threading,traceback
from pathlib import Path
ROOT=Path('/workspace/diffusion-pipe-easycontrol');QUEUE=Path('/workspace/nextscene_artifacts/anima1024_20261001/jobs');QUEUE.mkdir(exist_ok=True)
def atomic(path,data):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def monitor(proc,path):
 with path.open('w') as f:
  f.write('utc,memory_MiB,util_percent,power_W\n')
  while proc.poll() is None:
   r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True)
   f.write(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+','+r.stdout.strip()+'\n');f.flush();time.sleep(1)
def execute(p,s):
 state=p.with_suffix('.state.json');old=json.loads(state.read_text()) if state.exists() else {}
 if old.get('status') in ['done','failed']:return
 cmd=s['argv'][:]
 # If interrupted by a reboot, resume the newest COMPLETE trainer state.
 if s.get('train_output') and (old.get('status')=='running' or s.get('resume')):
  runs=sorted(Path(s['train_output']).glob('*/latest'))
  if runs and '--resume_from_checkpoint' not in cmd:cmd+=['--resume_from_checkpoint',runs[-1].parent.name]
 log=Path(s['log']);log.parent.mkdir(parents=True,exist_ok=True)
 start=time.time();atomic(state,{'status':'running','start':start,'argv':cmd})
 env=os.environ.copy();env.update(NCCL_P2P_DISABLE='1',OMP_NUM_THREADS='8',PATH='/venv/main/bin:'+env.get('PATH',''),HF_HUB_DISABLE_PROGRESS_BARS='1',TOKENIZERS_PARALLELISM='false');env.update(s.get('env',{}))
 with log.open('a') as f:
  f.write('\nJob '+p.stem+' '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n');f.flush()
  proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
  atomic(state,{'status':'running','start':start,'argv':cmd,'pid':proc.pid})
  thread=threading.Thread(target=monitor,args=(proc,log.with_suffix('.gpu.csv')),daemon=True);thread.start()
  rc=proc.wait();thread.join(timeout=3)
 atomic(state,{'status':'done' if rc==0 else 'failed','start':start,'end':time.time(),'seconds':time.time()-start,'returncode':rc,'argv':cmd})
 print(p.stem,'completed rc',rc,flush=True)
while True:
 for p in sorted(QUEUE.glob('*.job.json')):
  try:execute(p,json.loads(p.read_text()))
  except Exception:
   traceback.print_exc();atomic(p.with_suffix('.state.json'),{'status':'failed','error':traceback.format_exc()})
 time.sleep(3)
