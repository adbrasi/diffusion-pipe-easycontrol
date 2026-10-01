#!/usr/bin/env python3
"""Three genuine epochs of native DiffSynth Qwen21 editing, from a fresh LoRA.

Encode ahead in stages during epoch one, then reuse lossless conditioning caches.
Each epoch has a deterministic permutation of the entire original inventory.
GPU training/sampling are serial; public backups are verified before pruning.
"""
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from huggingface_hub import HfApi
from anima1024_campaign import atomic_json, verified_upload

ROOT=Path('/workspace/qwen21')
CODE=Path(__file__).resolve().parents[1]
REPO='AdwolfCzar/qwen-image-21-nextscene-edit'
PY='/venv/main/bin/python'
NATIVE=CODE/'tools/qwen21_native.py'

def run(args,log):
    with log.open('a') as stream:
        p=subprocess.Popen([PY,str(NATIVE),*args],cwd=CODE,stdout=stream,stderr=subprocess.STDOUT)
        # save_quit is consumed by the native operational runner after an update;
        # never send a termination signal to a live training process.
        p.wait()
    if p.returncode: raise RuntimeError(f'Job failed ({p.returncode}); see {log}')

def main():
    summary=json.loads((ROOT/'dataset_summary.json').read_text())
    n=summary['pairs'];total=n*3
    if n!=11526 or summary['repeat']!=1 or summary['presentations']!=total:
        raise RuntimeError('Expected full dataset, repeat 1, three genuine epochs')
    smoke=json.loads((ROOT/'smoke1024/step000010/report.json').read_text())
    parity=json.loads((ROOT/'smoke512/step000010/report.json').read_text())['checkpoint_parity']
    resume=json.loads((ROOT/'resume_audit.json').read_text())
    if not smoke['changed_lora_B'] or not smoke['finite_gradients'] or not resume['tensor_bit_identical']:
        raise RuntimeError('Preflight validation failed')
    if parity['worst_relative_gradient']>.05:
        raise RuntimeError('Gradient checkpointing audit failed')
    for label in ('base','smoke10'):
        if not (ROOT/'samples'/label/'report.json').exists():
            raise RuntimeError(f'Missing native image smoke: {label}')
    output=ROOT/'checkpoints';output.mkdir(exist_ok=True)
    status_path=ROOT/'campaign_state.json'
    def status(name,**kwargs):
        atomic_json(status_path,dict(status=name,step=step,total_steps=total,
                    pairs=n,epochs=3,repo=REPO,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kwargs))
    step=0
    api=HfApi();api.create_repo(REPO,private=False,exist_ok=True)
    if api.repo_info(REPO).private: raise RuntimeError('Public repository required')
    receipts=ROOT/'verified_backups';receipts.mkdir(exist_ok=True)
    latest=output/'latest.json'
    if latest.exists(): step=json.loads(latest.read_text())['step']
    config=dict(framework='DiffSynth-Studio',model='Qwen/Qwen-Image-2.1',
                upstream_commit='974cfa37f27ac55eba3b6d10efa21f876900572d',
                resolution=1024,pairs=n,epochs=3,total_steps=total,
                batch_size=1,gradient_accumulation_steps=1,dataset_repeat=1,
                lora_rank=32,lora_alpha=32,lora_target_modules='native autodetected DiT block linears',
                base_quant='comfy_kitchen_fp8_w8a8',adapter_dtype='float32',compute_dtype='bfloat16',
                learning_rate=1e-4,warmup_steps=200,final_learning_rate=1e-5,
                scheduler='cosine',optimizer='AdamW',weight_decay=.01,clip_grad_norm=1,
                cache='lossless BF16 latents + multimodal features; Zstandard',
                initialization='fresh; smoke LoRA is not used',
                seed=76,inference_kv_cache=False,
                caption_policy='one unchanged original caption per pair')
    atomic_json(ROOT/'config.json',config)
    card='''---
base_model: Qwen/Qwen-Image-2.1
library_name: diffsynth
pipeline_tag: image-to-image
license: apache-2.0
tags:
- lora
- image-editing
- qwen-image-2.1
datasets:
- AdwolfCzar/proxima_cena_grounded_original_dataset
---
# Qwen Image 2.1 — NextScene Edit

Native DiffSynth-Studio editing LoRA: scene A + original description → scene B.
Fresh training on all 11,526 complete available pairs from four subsets, 1024-area, three epochs. 24 established heldout pairs stay separate.

LoRA rank/alpha 32, scaled FP8 DiT, BF16 compute, FP32 adapters, AdamW, peak LR 1e-4 with 200-step warmup and cosine decay. Real batch 1, accumulation 1. Checkpoints and samples are periodically uploaded with verified hashes.

The step-10 smoke verifies execution and learning, not final visual quality. Loading and sampling use the same native DiffSynth model and numerical policy; see `evidence/code/qwen21_native.py`.
'''
    api.upload_file(repo_id=REPO,path_or_fileobj=card.encode(),path_in_repo='README.md')
    perm={epoch:random.Random(76+epoch).sample(range(n),n) for epoch in range(3)}
    # Operational files are small. Full metadata stays in the existing public
    # source dataset; no arbitrary data subset is introduced in this run.
    evidence=ROOT/'evidence';(evidence/'code').mkdir(parents=True,exist_ok=True)
    for name in ('dataset_summary.json','model_provenance.json','config.json',
                 'resume_audit.json','upstream_text_encoder_hook_cleanup.patch'):
        shutil.copy2(ROOT/name,evidence/name)
    for name in ('qwen21_native.py','qwen21_prepare.py','qwen21_campaign.py'):
        shutil.copy2(CODE/'tools'/name,evidence/'code'/name)
    verified_upload(api,REPO,evidence,'evidence')
    while step<total:
        if (ROOT/'save_quit').exists():
            status('stopped');return
        # Do not cross an epoch boundary within a cache stage.
        boundary=(step//n+1)*n
        target=min(total,boundary,100 if step<100 else (step//250+1)*250)
        epoch=step//n
        indices=perm[epoch][step%n:(target-1)%n+1]
        if len(indices)!=target-step: raise RuntimeError('Stage pair inventory mismatch')
        index_file=ROOT/f'stage_indices_{target:06}.json'
        atomic_json(index_file,indices)
        status('caching',target_step=target,epoch=epoch+1)
        run(['cache','--resolution','1024','--indices',str(index_file),
             '--report',str(ROOT/f'cache_until_{target:06}.json')],ROOT/f'cache_until_{target:06}.log')
        if (ROOT/'save_quit').exists(): status('stopped');return
        status('training',target_step=target,epoch=epoch+1)
        run(['train','--resolution','1024','--until',str(target),
             '--output',str(output)],ROOT/f'train_until_{target:06}.log')
        info=json.loads(latest.read_text());step=info['step']
        if step<target and not (ROOT/'save_quit').exists():
            raise RuntimeError('Training ended before its genuine step target')
        folder=Path(info['adapter']).parent
        status('backup',target_step=target)
        receipt=receipts/(folder.name+'.json')
        if not receipt.exists():
            files=verified_upload(api,REPO,folder,'checkpoints/'+folder.name)
            atomic_json(receipt,dict(repo=REPO,files=files))
        status('sampling',target_step=target)
        if not (ROOT/'save_quit').exists():
            destination=ROOT/'samples'/folder.name
            run(['sample','--prepared',str(ROOT/'eval_cache'),'--adapter',info['adapter'],
                 '--output',str(destination)],ROOT/f'sample_{step:06}.log')
            files=verified_upload(api,REPO,destination,'samples/'+folder.name)
            atomic_json(receipts/(folder.name+'_samples.json'),dict(repo=REPO,files=files))
            if step%n==0:
                # Full matched/shuffled evaluation on the 24 established heldout
                # pairs at each epoch; the frequent preview uses three cases.
                prepared=ROOT/'eval_epoch_cache'
                if not (prepared/'manifest.json').exists():
                    run(['prepare_eval','--resolution','1024','--all-heldout',
                         '--output',str(prepared)],ROOT/'prepare_epoch_eval.log')
                baseline=ROOT/'samples/base_full_heldout'
                if not (baseline/'report.json').exists():
                    run(['sample','--prepared',str(prepared),'--output',str(baseline)],
                        ROOT/'sample_base_full_heldout.log')
                    verified_upload(api,REPO,baseline,'samples/base_full_heldout')
                epoch_out=ROOT/'samples'/f'epoch{step//n}_full_heldout'
                run(['sample','--prepared',str(prepared),'--adapter',info['adapter'],
                     '--output',str(epoch_out)],ROOT/f'sample_epoch{step//n}.log')
                verified_upload(api,REPO,epoch_out,'samples/'+epoch_out.name)
        # Full historical snapshots remain verified on the public Hub. Keep the
        # latest two locally so cache capacity does not shrink each sampling stage.
        old=sorted(output.glob('step*'))[:-2]
        for p in old:
            if (receipts/(p.name+'.json')).exists(): shutil.rmtree(p)
        for name in ('campaign_state.json',):
            api.upload_file(repo_id=REPO,path_or_fileobj=str(ROOT/name),path_in_repo=name)
        api.upload_file(repo_id=REPO,path_or_fileobj=str(output/'loss.jsonl'),path_in_repo='logs/loss.jsonl')
    status('complete')
    api.upload_file(repo_id=REPO,path_or_fileobj=str(status_path),path_in_repo='campaign_state.json')

if __name__=='__main__':
    try: main()
    except Exception as exc:
        path=ROOT/'campaign_state.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        state.update(status='failed',error=str(exc))
        atomic_json(path,state)
        print('CAMPAIGN_FAILED',str(exc),flush=True)
        # Leave a reviewable failure instead of an automatic GPU retry loop.
        # Supervisor may restart an unexpectedly killed process, but a diagnosed
        # failure exits normally with an explicit failed status, avoiding GPU loops.
