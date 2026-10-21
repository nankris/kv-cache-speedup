"""Run on Colab: python gpu_benchmark.py; checkpoints under results/gpu/."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import csv,gc,hashlib,json,platform,subprocess,time,fcntl
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from huggingface_hub import model_info
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'/'gpu'; R.mkdir(parents=True,exist_ok=True)
lock=(R/'benchmark.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
torch.set_num_threads(2); torch.manual_seed(121)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
print(subprocess.getoutput('nvidia-smi'),flush=True)
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--models',nargs='+',default=['0.6B'],choices=['0.6B','1.7B'])
for size in parser.parse_args().models:
    mid='Qwen/Qwen3-'+size; tag=size.replace('.','_'); D=R/tag; D.mkdir(exist_ok=True)
    envpath=D/'environment.json'
    rev=(json.loads(envpath.read_text())['revision'] if envpath.exists() else ('c1899de289a04d12100db370d81485cdf75e47ca' if size=='0.6B' else model_info(mid).sha))
    tok=AutoTokenizer.from_pretrained(mid,revision=rev)
    model=AutoModelForCausalLM.from_pretrained(mid,revision=rev,torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
    meta=dict(model=mid,revision=rev,torch=torch.__version__,transformers=transformers.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(),dtype='float32',tf32=False,attention='sdpa',threads=2,config=model.config.to_dict())
    if envpath.exists():
        old=json.loads(envpath.read_text()); assert old==meta, 'Environment changed on resume'
    envpath.write_text(json.dumps(meta,indent=2))
    base=tok.encode('Explain how a library organizes books. Describe shelves, catalogues, readers, and borrowing in clear language. ',add_special_tokens=False)
    prompts={p:torch.tensor([(base*((p+len(base)-1)//len(base)))[:p]],device='cuda') for p in [64,512]}
    (D/'prompts.json').write_text(json.dumps({p:x[0].tolist() for p,x in prompts.items()}))
    def gen(p,n,cache):
        return model.generate(input_ids=prompts[p],attention_mask=torch.ones_like(prompts[p]),max_new_tokens=n,min_new_tokens=n,do_sample=False,use_cache=cache,eos_token_id=None,pad_token_id=tok.pad_token_id)
    with torch.inference_mode():
        for cache in [True,False]:
            gen(64,16,cache); gen(512,16,cache)
        torch.cuda.synchronize()
        path=D/'timings.csv'; rows=list(csv.DictReader(path.open())) if path.exists() else []
        done={(int(r['prompt_tokens']),int(r['output_tokens']),int(r['repeat']),r['use_cache']=='True') for r in rows}
        fields=['prompt_tokens','output_tokens','repeat','use_cache','seconds','tokens_per_second','generated_tokens','output_sha256','utc']
        with path.open('a',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields)
            if not rows: w.writeheader(); f.flush()
            for rep in range(3):
                for n in [32,64,128,256,512]:
                    for p in [64,512]:
                        for cache in ([True,False] if rep%2==0 else [False,True]):
                            if (p,n,rep,cache) in done: continue
                            gc.collect(); torch.cuda.synchronize()
                            start=time.perf_counter(); out=gen(p,n,cache); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                            ids=out[0,p:].cpu().tolist(); assert len(ids)==n
                            (D/f'output_p{p}_n{n}_r{rep}_c{int(cache)}.json').write_text(json.dumps(ids))
                            row=dict(prompt_tokens=p,output_tokens=n,repeat=rep,use_cache=cache,seconds=elapsed,tokens_per_second=n/elapsed,generated_tokens=len(ids),output_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
                            w.writerow(row); f.flush(); os.fsync(f.fileno()); print(size,json.dumps(row),flush=True); del out
        memory=[]
        for t in [64,319,512,767,1023]:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
            ids=torch.tensor([(base*((t+len(base)-1)//len(base)))[:t]],device='cuda')
            baseline=torch.cuda.memory_allocated(); torch.cuda.reset_peak_memory_stats()
            out=model(input_ids=ids,use_cache=True,logits_to_keep=1); torch.cuda.synchronize()
            tensors=[x for layer in out.past_key_values.layers for x in (layer.keys,layer.values)]
            c=model.config; per=c.num_hidden_layers*c.num_key_value_heads*c.head_dim*2*4
            memory.append(dict(tokens=t,formula_bytes=per*t,tensor_bytes=sum(x.numel()*x.element_size() for x in tensors),storage_bytes=sum(x.untyped_storage().nbytes() for x in tensors),baseline_bytes=baseline,live_bytes=torch.cuda.memory_allocated(),delta_bytes=torch.cuda.memory_allocated()-baseline,peak_bytes=torch.cuda.max_memory_allocated(),reserved_bytes=torch.cuda.memory_reserved(),key_shape=list(tensors[0].shape)))
            del tensors,out,ids
        (D/'memory.json').write_text(json.dumps(memory,indent=2))
    del model,prompts,tok; gc.collect(); torch.cuda.empty_cache()
    print('MODEL COMPLETE',size,flush=True)
print('ALL COMPLETE',flush=True)
