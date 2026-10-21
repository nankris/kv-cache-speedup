import csv, gc, hashlib, json, os, platform, time, ctypes, resource
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import torch, transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'results'; R.mkdir(exist_ok=True)
torch.set_num_threads(2); torch.set_num_interop_threads(1)
torch.manual_seed(121)
model_id='Qwen/Qwen3-0.6B'
dtype=torch.float32
print('Loading model',flush=True)
revision=json.loads((R/'model_source.json').read_text())['sha']
tok=AutoTokenizer.from_pretrained(model_id,revision=revision)
model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,dtype=dtype,attn_implementation='sdpa').eval()
gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
print('Loaded RSS KiB',resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,flush=True)
config=model.config.to_dict()
(R/'config.json').write_text(json.dumps(config,indent=2))
meta=dict(model=model_id,revision=revision,torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),platform=platform.platform(),threads=torch.get_num_threads(),dtype=str(next(model.parameters()).dtype),quantization='none',attention='sdpa',cpu=platform.processor(),parameters=sum(p.numel() for p in model.parameters()),start_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
if (R/'timings.csv').exists() and (R/'environment.json').exists():
    prior=json.loads((R/'environment.json').read_text())
    for key in ['model','revision','torch','transformers','threads','dtype','quantization','attention']:
        assert prior.get(key,'none' if key=='quantization' else None)==meta.get(key), f'Resume metadata differs: {key}'
(R/'environment.json').write_text(json.dumps(meta,indent=2))
base=tok.encode('Explain how a library organizes books. Describe shelves, catalogues, readers, and borrowing in clear language. ',add_special_tokens=False)
prompts={p:torch.tensor([(base*((p+len(base)-1)//len(base)))[:p]]) for p in [64,512]}
(R/'prompts.json').write_text(json.dumps({p:{'ids':x[0].tolist(),'text':tok.decode(x[0])} for p,x in prompts.items()},indent=2))
def generate(ids,n,cache):
    return model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=n,min_new_tokens=n,do_sample=False,use_cache=cache,eos_token_id=None,pad_token_id=tok.pad_token_id)
with torch.inference_mode():
    for cache in [True,False]:
        print('Warmup',cache,flush=True)
        generate(prompts[64],8,cache)
        print('Warmup done',cache,flush=True)
    memory=[]
    for p,ids in prompts.items():
        out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
        print('Memory cache',p,[(i, None if layer.keys is None else list(layer.keys.shape)) for i,layer in enumerate(out.past_key_values.layers)],flush=True)
        tensors=[t for layer in out.past_key_values.layers for t in (layer.keys,layer.values)]
        assert len(tensors)==2*config['num_hidden_layers'] and all(t is not None for t in tensors)
        formula=config['num_hidden_layers']*config['num_key_value_heads']*config['head_dim']*2*next(model.parameters()).element_size()*p
        memory.append(dict(tokens=p,formula_bytes=formula,tensor_bytes=sum(t.numel()*t.element_size() for t in tensors),storage_bytes=sum(t.untyped_storage().nbytes() for t in tensors),key_shape=list(tensors[0].shape),value_shape=list(tensors[1].shape)))
        del out,tensors
    (R/'memory.json').write_text(json.dumps(memory,indent=2))
    path=R/'timings.csv'
    rows=list(csv.DictReader(path.open())) if path.exists() else []
    done={(int(r['prompt_tokens']),int(r['output_tokens']),int(r['repeat']),r['use_cache']=='True') for r in rows}
    fields=['prompt_tokens','output_tokens','repeat','use_cache','seconds','tokens_per_second','generated_tokens','output_sha256','utc']
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        if not rows: writer.writeheader(); f.flush()
        for rep in range(3):
            for n in [32,64,128,256]:
                for p in [64,512]:
                    for cache in ([True,False] if rep%2==0 else [False,True]):
                        if (p,n,rep,cache) in done: continue
                        gc.collect()
                        started=time.perf_counter(); out=generate(prompts[p],n,cache); elapsed=time.perf_counter()-started
                        row=dict(prompt_tokens=p,output_tokens=n,repeat=rep,use_cache=cache,seconds=elapsed,tokens_per_second=n/elapsed,generated_tokens=out.shape[1]-p,output_sha256=hashlib.sha256(out.numpy().tobytes()).hexdigest(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                        writer.writerow(row); f.flush(); os.fsync(f.fileno())
                        print(json.dumps(row),flush=True)
                        del out
print('Done',flush=True)
