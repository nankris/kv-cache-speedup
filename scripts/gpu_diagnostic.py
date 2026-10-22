"""Untimed mechanism check, run after timing to avoid GPU contention."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
import json,torch,transformers
from pathlib import Path
from transformers import AutoModelForCausalLM
R=Path('/content/kvcache/results/gpu'); d=R/'0_6B'
meta=json.loads((d/'environment.json').read_text())
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
model=AutoModelForCausalLM.from_pretrained(meta['model'],revision=meta['revision'],torch_dtype=torch.float32,attn_implementation='sdpa').to('cuda').eval()
ids=torch.tensor([json.loads((d/'prompts.json').read_text())['64']],device='cuda')
records=[]
for cache in [True,False]:
    calls=[]
    def hook(module,args,kwargs):
        x=kwargs.get('input_ids')
        calls.append(dict(input_shape=list(x.shape),use_cache=kwargs.get('use_cache'),past_present=kwargs.get('past_key_values') is not None,device=str(x.device)))
    h=model.register_forward_pre_hook(hook,with_kwargs=True)
    with torch.inference_mode():
        out=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=4,min_new_tokens=4,do_sample=False,use_cache=cache,eos_token_id=None,pad_token_id=151643)
    torch.cuda.synchronize(); h.remove()
    records.append(dict(requested_cache=cache,calls=calls,output=out[0].tolist()))
assert [x['input_shape'][1] for x in records[0]['calls']]==[64,1,1,1]
assert [x['input_shape'][1] for x in records[1]['calls']]==[64,65,66,67]
assert records[0]['output']==records[1]['output']
(R/'mechanism_check.json').write_text(json.dumps(records,indent=2)); print(json.dumps(records,indent=2))
