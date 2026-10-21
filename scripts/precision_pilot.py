"""Untimed loading, then matched 64/512-token single forwards to choose CPU precision."""
import os
os.environ['OMP_NUM_THREADS']='2'
import time,json,gc
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
R=Path(__file__).resolve().parents[1]/'results'
torch.set_num_threads(2); torch.set_num_interop_threads(1)
rev=json.loads((R/'model_source.json').read_text())['sha']
m=AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B',revision=rev,dtype=torch.float32,attn_implementation='sdpa').eval()
prompts=json.loads((R/'prompts.json').read_text())
rows=[]
with torch.inference_mode():
    for dtype in [torch.float32,torch.bfloat16]:
        m.to(dtype=dtype)
        for p in [64,512]:
            ids=torch.tensor([prompts[str(p)]['ids']])
            m(input_ids=ids,use_cache=False,logits_to_keep=1)
            for repeat in range(2):
                start=time.perf_counter(); out=m(input_ids=ids,use_cache=False,logits_to_keep=1); elapsed=time.perf_counter()-start
                rows.append(dict(dtype=str(dtype),prompt_tokens=p,repeat=repeat,seconds=elapsed))
                print(rows[-1],flush=True)
                (R/'precision_pilot.json').write_text(json.dumps(rows,indent=2))
print(torch.__config__.show(),flush=True)
