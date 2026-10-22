"""Report incomplete data honestly, or apply the fixed analysis to all 48 runs."""
import csv,json,runpy,itertools
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'
rows=list(csv.DictReader((R/'timings.csv').open()))
if len(rows)==48:
    runpy.run_path(str(ROOT/'scripts'/'analyze_complete.py'),run_name='__main__')
    raise SystemExit
assert len(rows)<48
keys={(int(r['prompt_tokens']),int(r['output_tokens']),int(r['repeat']),r['use_cache']=='True') for r in rows}
assert len(keys)==len(rows)
assert all(int(r['output_tokens'])==int(r['generated_tokens']) for r in rows)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
memory=json.loads((R/'memory.json').read_text())
for m in memory:
    assert m['formula_bytes']==m['tensor_bytes']==m['storage_bytes']
lookup={(int(r['prompt_tokens']),int(r['output_tokens']),r['use_cache']=='True'):r for r in rows}
a=lookup[64,32,True]; b=lookup[64,32,False]
ratio=float(b['seconds'])/float(a['seconds'])
result=dict(verdict='INCONCLUSIVE',reason='256-token endpoint not measured; feasibility failure',headline_number=None,ci95=None,completed_runs=len(rows),required_runs=48,completed_256_token_runs=sum(int(r['output_tokens'])==256 for r in rows),pilot_32_token_ratio=ratio,pilot_output_hashes_match=a['output_sha256']==b['output_sha256'],fits=None,gpu=None,colab_cu=0)
(R/'analysis.json').write_text(json.dumps(result,indent=2))
summary=[]
for p,n,c in itertools.product([64,512],[32,64,128,256],[True,False]):
    rs=[r for r in rows if int(r['prompt_tokens'])==p and int(r['output_tokens'])==n and (r['use_cache']=='True')==c]
    summary.append(dict(prompt_tokens=p,output_tokens=n,use_cache=c,completed_repeats=len(rs),mean_seconds=sum(float(r['seconds']) for r in rs)/len(rs) if rs else '',mean_tokens_per_second=sum(float(r['tokens_per_second']) for r in rs)/len(rs) if rs else '',ci95='not estimable'))
with (R/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=summary[0]); w.writeheader(); w.writerows(summary)
fig,axes=plt.subplots(1,2,figsize=(11,4.8))
labels=['64 prompt\ncache on','64 prompt\ncache off','512 prompt\ncache on']
values=[float(lookup[64,32,True]['seconds']),float(lookup[64,32,False]['seconds']),float(lookup[512,32,True]['seconds'])]
axes[0].bar(labels,values,color=['#167d9a','#cc592a','#167d9a'])
for i,v in enumerate(values): axes[0].text(i,v+1,f'{v:.2f}s',ha='center')
axes[0].set(title='32-token float32 pilot only (one run per bar)',ylabel='Generation time, seconds',ylim=(0,max(values)*1.2))
xs=range(len(memory))
axes[1].bar([i-.18 for i in xs],[m['formula_bytes']/2**20 for m in memory],width=.36,label='Formula',color='#167d9a')
axes[1].bar([i+.18 for i in xs],[m['tensor_bytes']/2**20 for m in memory],width=.36,label='Measured tensors',color='#55aa79')
axes[1].set(xticks=list(xs),xticklabels=[f"{m['tokens']} tokens" for m in memory],title='Float32 KV cache memory',ylabel='MiB'); axes[1].legend()
fig.suptitle('KV cache: 256-token endpoint unmeasured — INCONCLUSIVE')
fig.tight_layout(); fig.savefig(ROOT/'chart.png',dpi=180)
lines=['# KV cache CPU experiment: INCONCLUSIVE','','**Requested headline: unavailable. No 256-token cache-on/off pair completed; no N× value can honestly be given.**','','The original >=5× hypothesis is INCONCLUSIVE because the required endpoint is missing. There is no 95% CI, and no valid linear/quadratic growth fit: only one output length was measured. This is a feasibility failure, not evidence against KV caching.','','## Completed float32 pilot results','','Qwen/Qwen3-0.6B, CPU, float32, two Torch threads, greedy, model.generate including prefill. Each row below is ONE completed run, not the requested three-repeat estimate.','','| Prompt tokens | Output tokens | Cache | Time (s) | Tokens/s | Completed repeats |','|---:|---:|:---|---:|---:|---:|']
for r in rows: lines.append(f"| {r['prompt_tokens']} | {r['output_tokens']} | {'on' if r['use_cache']=='True' else 'off'} | {float(r['seconds']):.2f} | {float(r['tokens_per_second']):.3f} | 1/3 |")
lines+=['',f'The one matched 64-prompt/32-output pair gives a **{ratio:.2f}× pilot slowdown**, with identical output hashes. This is not the requested headline, has no repeat-based CI, and must not be extrapolated to 256 tokens.','','## Requested matrix completion','','| Prompt | Output | Cache-on repeats | Cache-off repeats |','|---:|---:|---:|---:|']
for p,n in itertools.product([64,512],[32,64,128,256]):
    counts=[sum(int(r['prompt_tokens'])==p and int(r['output_tokens'])==n and (r['use_cache']=='True')==c for r in rows) for c in [True,False]]
    lines.append(f'| {p} | {n} | {counts[0]}/3 | {counts[1]}/3 |')
lines+=['','## Why the experiment stopped','','The original float32 uncached 32-token pilot already took 57.72 seconds. A separate forward-only precision pilot measured 18.48 and 19.00 seconds for a 512-token prompt in float32, and BF16 was slower at 64 tokens. These are feasibility measurements, not substitutes for generation. Completing the full original matrix was projected to take hours, beyond the small experiment scope.','','An int8 feasibility amendment was attempted before observing the primary endpoint. Two setup attempts were OOM-killed; a lower-memory conversion succeeded, but its 32-token uncached run took 75.17 seconds versus 12.63 seconds cached. These are different numerical representations and output hashes differ in that int8 pair. Int8 results are archived separately and excluded from the float32 table and inference. No partial interrupted generation was treated as a completed run.','','## KV cache memory: successfully checked','','The [pinned Qwen config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json) has 28 layers, 8 KV heads (16 query heads), and a 128-dimensional head. At batch size 1 in float32:','','**layers × KV heads × head_dim × 2 × bytes × tokens = 28 × 8 × 128 × 2 × 4 × T = 229,376T bytes = 224 KiB/token.**','','| Cached tokens | Formula MiB | Tensor MiB | Backing storage MiB |','|---:|---:|---:|---:|']
for m in memory: lines.append(f"| {m['tokens']} | {m['formula_bytes']/2**20:.2f} | {m['tensor_bytes']/2**20:.2f} | {m['storage_bytes']/2**20:.2f} |")
lines+=['','The formula exactly matches populated K/V tensors in all 28 layers. Each tensor shape is [1, 8, T, 128]. This measures cache tensor storage, excluding weights, other activations, and allocator overhead; process RSS cannot isolate the cache. At 256 new tokens, prompt + output − 1 predicts 319 cached tokens (69.78 MiB) or 767 (167.78 MiB); those generation-time sizes are formula estimates, not measured endpoints. BF16 KV tensors would use half the bytes, but the original benchmark uses float32.','','## Growth fits and limits','','Linear and quadratic growth fits are unavailable: four output lengths with three repeats each were requested, and only one length has observations. [Hugging Face explains](https://huggingface.co/docs/transformers/en/cache_explanation) that KV caching reuses earlier keys and values; cached attention still reads the growing context. Without caching, each step recomputes the prefix, but this experiment did not measure the requested scaling curves.','','Public model and synthetic prompts only; no chat template, EOS stopping disabled, fixed output lengths. Pinned revision, library versions, prompt IDs, raw completed rows, failed setup logs, and separate feasibility pilots are saved in results/. Main scripts resume float32 from results/timings.csv and apply the predeclared bootstrap/decision rule only once all 48 rows exist. GPU: none. Colab CU: **0**.','','## Two-sentence plain-English takeaway','','The short pilot showed a clear speed benefit from remembering earlier attention data, but this run did not finish the 256-token comparison needed for the headline. We did verify the memory cost: this model stores 224 KiB of float32 attention data per cached token.','']
(ROOT/'writeup.md').write_text('\n'.join(lines))
print(json.dumps(result,indent=2))
