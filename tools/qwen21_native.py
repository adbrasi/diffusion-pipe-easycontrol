#!/usr/bin/env python3
"""Operational runner around the unmodified DiffSynth Qwen21 training module.

Model, multimodal conditioning, masks, scheduler and flow loss are upstream.
Adds lossless compact caching, full optimizer/RNG resume, atomic checkpoints,
finite-gradient audits, native heldout inference and graceful save_quit.
"""
import argparse
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('DIFFSYNTH_SKIP_DOWNLOAD','True')
import numpy as np
from PIL import Image
import torch
import zstandard as zstd
from safetensors.torch import save_file, load_file
from diffsynth.core import ModelConfig, QuantizeConfig
from diffsynth.pipelines.qwen_image_21 import QwenImage21Pipeline

ROOT=Path('/workspace/qwen21')
MODELS=Path('/workspace/models/qwen_image_21')
QUANT='comfy_kitchen_fp8_w8a8'
spec=importlib.util.spec_from_file_location('qwen21_upstream_train',
    '/workspace/DiffSynth-Studio/examples/qwen_image_21/model_training/train.py')
upstream=importlib.util.module_from_spec(spec)
spec.loader.exec_module(upstream)

def atomic_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=str))
    tmp.replace(path)

def rows():
    return [json.loads(s) for s in (ROOT/'dataset.jsonl').read_text().splitlines()]

def dimensions(image, resolution):
    from diffsynth.pipelines.qwen_image_21 import QwenImage21Unit_EditImageEmbedder
    return QwenImage21Unit_EditImageEmbedder.calculate_dimensions(resolution**2,image.width/image.height)

def item(row, resolution):
    target=Image.open(row['image']).convert('RGBA')
    ref=Image.open(row['edit_image']).convert('RGBA')
    target=target.resize(dimensions(target,resolution),Image.Resampling.LANCZOS)
    return dict(image=target,edit_image=ref,prompt=row['prompt'])

def paths(kind):
    pattern={'dit':'transformer/diffusion_pytorch_model-*.safetensors',
             'text_encoder':'text_encoder/model-*.safetensors',
             'vae':'vae/diffusion_pytorch_model.safetensors'}[kind]
    matches=sorted(str(p) for p in MODELS.glob(pattern))
    if not matches: raise RuntimeError(f'Missing weights: {pattern}')
    return matches if len(matches)>1 else matches[0]

def module(task, quant=QUANT):
    # Split native preprocessing and training avoids moving the 8B VLM each step.
    kinds=['text_encoder','vae'] if task.endswith('data_process') else ['dit']
    model_paths=json.dumps([paths(k) for k in kinds])
    quant_options=None
    if 'dit' in kinds and quant!='bf16':
        quant_options=json.dumps(paths('dit'))+':'+quant
    result=upstream.QwenImage21TrainingModule(
        model_paths=model_paths,processor_path=str(MODELS/'processor'),
        lora_base_model='dit',lora_target_modules='',lora_rank=32,
        use_gradient_checkpointing=True,extra_inputs='edit_image',
        quant_options=quant_options,device='cuda',task=task)
    if task.endswith(':train') and result.pipe.units:
        raise RuntimeError('Unexpected preprocessing units remain in cached training')
    result.to('cuda')
    return result

def cpu(data):
    if isinstance(data,torch.Tensor): return data.detach().cpu().contiguous()
    if isinstance(data,dict): return {k:cpu(v) for k,v in data.items()}
    if isinstance(data,(list,tuple)): return type(data)(cpu(v) for v in data)
    return data

def compact(inputs):
    shared,pos,neg=inputs
    # These values are no longer read by any unit in sft:train. The native
    # FlowMatchSFTLoss creates new Gaussian noise and a new timestep every step.
    ignored={'input_image','edit_image','prompt','noise','latents','rand_device'}
    shared={k:v for k,v in shared.items() if k not in ignored}
    pos={k:v for k,v in pos.items() if k not in ignored}
    return cpu((shared,pos,{}))

def write_cache(data,path):
    buffer=io.BytesIO()
    torch.save(data,buffer)
    raw=buffer.getvalue()
    compressed=zstd.ZstdCompressor(level=3).compress(raw)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_bytes(compressed)
    tmp.replace(path)
    # Lossless roundtrip includes both the VLM representation and VAE latents.
    restored=read_cache(path)
    def equal(a,b):
        if isinstance(a,torch.Tensor): return torch.equal(a,b)
        if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    if not equal(data,restored): raise RuntimeError('Cache is not lossless')
    return len(raw),len(compressed)

def read_cache(path):
    raw=zstd.ZstdDecompressor().decompress(Path(path).read_bytes())
    return torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)

def cache_path(index,resolution):
    for base in (ROOT/'cache',Path('/dev/shm/qwen21_cache')):
        p=base/str(resolution)/f'{index:05}.pt.zst'
        if p.exists(): return p
    return None

def verify_cache_identity():
    identity=dict(dataset_sha256=json.loads((ROOT/'dataset_summary.json').read_text())['dataset_sha256'],
                  model_revision='d26bb61231c349cf6b7896fa83353113880e1ba3',
                  format='native-edit-bfloat16-lossless-v1')
    path=ROOT/'cache/identity.json'
    if path.exists():
        if json.loads(path.read_text())!=identity:
            raise RuntimeError('Cache belongs to a different dataset or encoder; rebuild it explicitly')
    else: atomic_json(path,identity)

@torch.no_grad()
def cache(args):
    import shutil
    verify_cache_identity()
    data=rows()
    indices=json.loads(Path(args.indices).read_text())
    missing=[i for i in indices if cache_path(i,args.resolution) is None]
    if not missing: return
    model=module('sft:data_process')
    norm=model.pipe.text_encoder.model.model.language_model.norm
    initial_hooks=len(norm._forward_hooks)
    records=[]
    for count,i in enumerate(missing):
        start=time.monotonic()
        inputs=compact(model(item(data[i],args.resolution)))
        if len(norm._forward_hooks)!=initial_hooks:
            raise RuntimeError('Native VLM retained a per-image forward hook')
        # Disk keeps a 12 GiB checkpoint reserve; temporary RAM overflow can be
        # regenerated from the canonical dataset after an instance restart.
        base=ROOT/'cache'
        if shutil.disk_usage(ROOT).free < 12*1024**3+32*1024**2:
            base=Path('/dev/shm/qwen21_cache')
            if shutil.disk_usage('/dev/shm').free < 8*1024**3+32*1024**2:
                raise RuntimeError('Cache capacity exhausted; never truncate the dataset')
        path=base/str(args.resolution)/f'{i:05}.pt.zst'
        raw,size=write_cache(inputs,path)
        shared,pos,_=inputs
        record=dict(index=i,path=str(path),raw_bytes=raw,compressed_bytes=size,
                    prompt_shape=list(pos['prompt_embeds'].shape),
                    reference_latent_shapes=[list(t.shape) for t in shared['edit_latents']],
                    target_latent_shape=list(shared['input_latents'].shape),
                    seconds=time.monotonic()-start)
        records.append(record)
        print(json.dumps(record),flush=True)
        if (ROOT/'save_quit').exists(): break
    atomic_json(Path(args.report),dict(mode='cache',resolution=args.resolution,
                forward_hook_growth=0,lossless_roundtrip=True,records=records))

def trainable(model):
    return {n:p for n,p in model.named_parameters() if p.requires_grad}

def adapter_state(model):
    return {n.removeprefix('pipe.dit.'):p.detach().cpu().contiguous()
            for n,p in trainable(model).items()}

def load_adapter(model,path):
    state=load_file(str(path))
    expected=set(trainable(model))
    state={'pipe.dit.'+n:v for n,v in state.items()}
    if set(state)!=expected: raise RuntimeError('Adapter keys do not match the native module')
    model.load_state_dict(state,strict=False)

def rng_state():
    return dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),
                numpy=np.random.get_state(),python=random.getstate())

def restore_rng(state):
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])

def checkpoint_audit(model, params, inputs, optimizer):
    inputs=model.transfer_data_to_device(inputs,model.pipe.device,model.pipe.torch_dtype)
    shared,pos,_=inputs
    shared=dict(shared,latents=torch.randn_like(shared['input_latents']))
    timestep=torch.tensor([500.],device='cuda',dtype=torch.bfloat16)
    selected=[n for n in params if 'transformer_blocks.0.' in n or 'transformer_blocks.31.' in n]
    if not selected: raise RuntimeError('No first/last block gradients selected')
    results=[]
    def pack(t):
        # QuantizedTensor cannot be moved with the generic save_on_cpu hook in
        # this comfy-kitchen release. Frozen FP8 weights stay on GPU; only plain
        # graph tensors are offloaded for this uncheckpointed diagnostic.
        if hasattr(t,'_qdata'): return (t,None)
        return (t.detach().cpu().pin_memory(),t.device)
    def unpack(saved):
        value,device=saved
        return value if device is None else value.to(device,non_blocking=True)
    for enabled in (False,True):
        optimizer.zero_grad(set_to_none=True)
        shared['use_gradient_checkpointing']=enabled
        with torch.autograd.graph.saved_tensors_hooks(pack,unpack):
            result=model.pipe.model_fn(dit=model.pipe.dit,**shared,**pos,timestep=timestep)
            result.float().square().mean().backward()
        results.append((result.detach().cpu(),{n:params[n].grad.detach().cpu().clone() for n in selected}))
        del result
    output_error=float((results[0][0].float()-results[1][0].float()).abs().max())
    worst_grad=0.
    for name in selected:
        a,b=results[0][1][name],results[1][1][name]
        worst_grad=max(worst_grad,float((a-b).norm()/a.norm().clamp_min(1e-12)))
    report=dict(output_max_abs=output_error,worst_relative_gradient=worst_grad,
                layers=[0,31],gradient_tensors=len(selected),plain_saved_tensors='CPU')
    if output_error>0.01 or worst_grad>0.05: raise RuntimeError(f'Checkpointing parity failed: {report}')
    return report

def audit(args):
    model=module('sft:train',args.quant)
    params=trainable(model)
    for p in params.values(): p.data=p.data.float()
    latest=json.loads((Path(args.output)/'latest.json').read_text())
    load_adapter(model,latest['adapter'])
    optimizer=torch.optim.AdamW(list(params.values()),lr=1e-4)
    index=random.Random(76).sample(range(len(rows())),len(rows()))[0]
    report=checkpoint_audit(model,params,read_cache(cache_path(index,args.resolution)),optimizer)
    path=Path(latest['adapter']).parent/'report.json'
    previous=json.loads(path.read_text())
    previous['checkpoint_parity']=report
    atomic_json(path,previous)
    print('PARITY_AUDIT',json.dumps(report),flush=True)

def train(args):
    verify_cache_identity()
    torch.manual_seed(76);random.seed(76);np.random.seed(76)
    model=module('sft:train',args.quant)
    params=trainable(model)
    if not params or any('lora_' not in n for n in params):
        raise RuntimeError('Only native DiT LoRA parameters may be trained')
    # FP32 adapters are the normal PEFT policy; avoid rounding small Adam updates
    # into BF16 adapter weights. Base/activations remain native BF16 or scaled FP8.
    for p in params.values(): p.data=p.data.float()
    model.pipe.dit.train()
    optimizer=torch.optim.AdamW(list(params.values()),lr=1e-4,weight_decay=.01)
    n=len(rows())
    total=n*3
    def lr_factor(step):
        warmup=200
        if step<warmup: return (step+1)/warmup
        progress=min(1,(step-warmup)/(total-warmup))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    latest=output/'latest.json'
    start=0
    if latest.exists():
        info=json.loads(latest.read_text())
        state=torch.load(info['state'],map_location='cpu',weights_only=False)
        load_adapter(model,info['adapter'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'])
        start=state['step']
        if state['quant']!=args.quant or state['dataset_sha256']!=json.loads((ROOT/'dataset_summary.json').read_text())['dataset_sha256']:
            raise RuntimeError('Resume policy or dataset differs')
    original={n:p.detach().cpu().clone() for n,p in params.items()} if args.smoke else {}
    logs=[]
    torch.cuda.reset_peak_memory_stats()
    begin=time.monotonic()
    permutations={epoch:random.Random(76+epoch).sample(range(n),n) for epoch in range(3)}
    for step in range(start,min(args.until,total)):
        epoch,offset=divmod(step,n)
        index=permutations[epoch][offset]
        path=cache_path(index,args.resolution)
        if path is None: raise RuntimeError(f'Missing cache for pair {index}; never silently skip')
        data=read_cache(path)
        data[0]['use_gradient_checkpointing']=True
        data[0]['use_flex_attention']=True
        t=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss=model({},inputs=data)
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(list(params.values()),1.0,error_if_nonfinite=True)
        missing=[n for n,p in params.items() if p.grad is None]
        if missing: raise RuntimeError(f'Missing adapter gradients: {missing[:8]}')
        optimizer.step();scheduler.step()
        torch.cuda.synchronize()
        log=dict(step=step+1,epoch=epoch+1,pair_index=index,loss=float(loss.detach()),
                 gradient_norm=float(grad_norm),lr=optimizer.param_groups[0]['lr'],
                 seconds=time.monotonic()-t,samples_per_second=1/(time.monotonic()-t),
                 allocated_gib=torch.cuda.max_memory_allocated()/1024**3)
        print(json.dumps(log),flush=True)
        with (output/'loss.jsonl').open('a') as f: f.write(json.dumps(log)+'\n')
        if output==ROOT/'checkpoints':
            atomic_json(ROOT/'progress.json',dict(status='training',actual_step=step+1,
                        total_steps=total,epoch=epoch+1,pairs=n,**{k:v for k,v in log.items() if k not in ('step','epoch')}))
        logs.append(log)
        if (ROOT/'save_quit').exists(): break
    finish=logs[-1]['step'] if logs else start
    if finish==start: return
    checkpoint=output/f'step{finish:06}'
    checkpoint.mkdir(exist_ok=True)
    adapter=checkpoint/'adapter.safetensors'
    save_file(adapter_state(model),str(adapter),metadata=dict(
        base_model='Qwen/Qwen-Image-2.1',task='native_edit',framework='DiffSynth-Studio',
        upstream_commit='974cfa37f27ac55eba3b6d10efa21f876900572d',
        step=str(finish),rank='32',alpha='32',base_precision=args.quant))
    state=dict(optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
               rng=rng_state(),step=finish,quant=args.quant,
               dataset_sha256=json.loads((ROOT/'dataset_summary.json').read_text())['dataset_sha256'])
    state_path=checkpoint/'resume.pt'
    torch.save(state,state_path)
    atomic_json(latest,dict(step=finish,adapter=str(adapter),state=str(state_path)))
    report=dict(step=finish,starting_step=start,steps=finish-start,
                samples_per_second=(finish-start)/(time.monotonic()-begin),
                trainable_parameters=sum(p.numel() for p in params.values()),
                trainable_tensors=len(params),batch=1,accumulation=1,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                finite_gradients=True,logs=logs)
    if args.smoke:
        changed={n:float((p.detach().cpu()-original[n]).abs().max()) for n,p in params.items()}
        report['changed_lora_A']=sum(v>0 for n,v in changed.items() if 'lora_A' in n)
        report['changed_lora_B']=sum(v>0 for n,v in changed.items() if 'lora_B' in n)
        if not report['changed_lora_B']: raise RuntimeError('LoRA B did not learn')
        atomic_json(checkpoint/'report.json',report)
        # Checkpointing parity with a trained nonzero adapter, including real FP8.
        if not args.skip_parity:
            report['checkpoint_parity']=checkpoint_audit(model,params,
                read_cache(cache_path(permutations[0][0],args.resolution)),optimizer)
    atomic_json(checkpoint/'report.json',report)
    print('TRAIN_STAGE_COMPLETE',finish,flush=True)

@torch.no_grad()
def sample(args):
    heldout=json.loads((ROOT/'heldout.json').read_text())
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    pipe=QwenImage21Pipeline.from_pretrained(device='cuda',torch_dtype=torch.bfloat16,
        model_configs=[ModelConfig(path=paths('dit'),quantize=QuantizeConfig(method=args.quant))
                       if args.quant!='bf16' else ModelConfig(path=paths('dit')),
                       ModelConfig(path=paths('vae'))])
    pipe.scheduler.training=False
    if args.adapter:
        from diffsynth.diffusion.training_module import DiffusionTrainingModule
        helper=DiffusionTrainingModule()
        pipe.dit=helper.add_lora_to_model(pipe.dit,
            helper.parse_lora_target_modules(pipe.dit,''),32,upcast_dtype=torch.float32)
        state=load_file(args.adapter)
        keys={n for n,p in pipe.dit.named_parameters() if p.requires_grad}
        if set(state)!=keys: raise RuntimeError('Sampling LoRA keys differ from training')
        pipe.dit.load_state_dict(state,strict=False)
    pipe.dit.eval()
    prepared=Path(args.prepared)
    record=json.loads((prepared/'manifest.json').read_text())
    for entry in record['cases']:
        # Cached heldout prompts were produced through the same native VLM and
        # reference VAE units as training. The native pipeline skips only encoders.
        inputs=read_cache(Path(entry['cache']))
        shared,pos,_=helper.transfer_data_to_device(inputs,'cuda',torch.bfloat16) if args.adapter else upstream.DiffusionTrainingModule().transfer_data_to_device(inputs,'cuda',torch.bfloat16)
        h,w=shared['height'],shared['width']
        pipe.scheduler.set_timesteps(40,dynamic_shift_len=(h//16)*(w//16))
        generator=torch.Generator(device='cpu').manual_seed(76)
        latents=torch.randn(shared['input_latents'].shape,generator=generator,dtype=torch.bfloat16).to('cuda')
        # Tensorwise FP8 activation scales depend on the complete token group.
        # Disable the optional inference prefix cache to keep the same full-token
        # numerical policy as training, rather than quantizing a shortened group.
        kv_cache=None
        for i,t in enumerate(pipe.scheduler.timesteps):
            prediction=pipe.model_fn(dit=pipe.dit,latents=latents,
                timestep=t.reshape(1).to('cuda',torch.bfloat16),
                prompt_embeds=pos['prompt_embeds'],prompt_embeds_mask=pos['prompt_embeds_mask'],
                edit_image_pad_mask=pos['edit_image_pad_mask'],edit_latents=shared['edit_latents'],
                kv_cache=kv_cache,use_flex_attention=True)
            latents=pipe.scheduler.step(prediction,t,latents)
        image=pipe.vae_output_to_image(pipe.vae.decode(latents))
        name=entry['name']+'.png'
        image.save(output/name)
        if np.asarray(image.convert('RGB')).std()<1:
            raise RuntimeError('Flat image in native inference smoke')
        print('SAMPLE',str(output/name),flush=True)
    atomic_json(output/'report.json',dict(steps=40,seed=76,cfg=1,quant=args.quant,
                kv_cache=False,adapter=args.adapter,cases=record['cases']))

@torch.no_grad()
def prepare_eval(args):
    data=json.loads((ROOT/'heldout.json').read_text())
    model=module('sft:data_process')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    cases=[]
    # One image is generated first; the subsequent cases retain identical seed
    # and original caption, changing both native reference conditioning branches.
    cases_to_prepare=([(i,False) for i in range(len(data))]+[(i,True) for i in range(len(data))]
                      if args.all_heldout else [(0,False),(1,False),(0,True)])
    for index,swapped in cases_to_prepare:
        row=dict(data[index])
        if swapped: row['edit_image']=data[(index+1)%len(data)]['edit_image']
        name=f'heldout{index:02}_'+('shuffled' if swapped else 'true')
        inputs=compact(model(item(row,args.resolution)))
        path=out/(name+'.pt.zst')
        write_cache(inputs,path)
        cases.append(dict(name=name,cache=str(path),row=row))
    atomic_json(out/'manifest.json',dict(cases=cases))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['cache','train','sample','prepare_eval','audit'])
    p.add_argument('--indices');p.add_argument('--report')
    p.add_argument('--resolution',type=int,default=1024)
    p.add_argument('--quant',default=QUANT)
    p.add_argument('--until',type=int,default=10)
    p.add_argument('--output',default=str(ROOT/'checkpoints'))
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--skip-parity',action='store_true')
    p.add_argument('--all-heldout',action='store_true')
    p.add_argument('--adapter');p.add_argument('--prepared',default=str(ROOT/'eval_cache'))
    args=p.parse_args()
    torch.set_num_threads(8)
    globals()[args.mode](args)

if __name__=='__main__': main()
