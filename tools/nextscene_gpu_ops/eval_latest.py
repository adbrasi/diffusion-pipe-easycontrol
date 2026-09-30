import sys,subprocess
from pathlib import Path
import toml
config,step,out,*extra=sys.argv[1:]
c=toml.load(config);paths=sorted(Path(c['output_dir']).glob('*/step'+step));assert paths,('No checkpoint',config,step)
cmd=['/venv/main/bin/python','tools/nextscene_eval.py','--dit',c['model']['transformer_path'],'--vae',c['model']['vae_path'],'--llm',c['model']['llm_path'],'--pairs','/workspace/heldout_short','--ckpt',str(paths[-1]),'--out',out,'--limit','12','--steps','20','--width','512','--height','512','--ref_cfg','1.0']+extra
print('Evaluation:',cmd,flush=True);subprocess.run(cmd,check=True)
