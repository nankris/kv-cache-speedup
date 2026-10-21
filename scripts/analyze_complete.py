import csv,json,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results'
rows=list(csv.DictReader((R/'timings.csv').open()))
assert len(rows)==48, f'Expected 48 runs, got {len(rows)}'
d={(int(r['prompt_tokens']),int(r['output_tokens']),int(r['repeat']),r['use_cache']=='True'):float(r['seconds']) for r in rows}
assert len(d)==48
assert all(int(r['generated_tokens'])==int(r['output_tokens']) for r in rows)
rng=np.random.default_rng(121); indices=rng.integers(0,3,size=(20000,3))
def stats(ratios):
    logs=np.log(ratios); return dict(ratio=float(np.exp(logs.mean())),ci95=np.exp(logs[indices].mean(axis=1)).__array__().tolist())
def ratio_summary(ratios):
    logs=np.log(ratios); ci=np.quantile(np.exp(logs[indices].mean(axis=1)),[.025,.975]); return dict(ratio=float(np.exp(logs.mean())),ci95=ci.tolist(),paired_ratios=list(map(float,ratios)))
per_prompt={p:ratio_summary([d[p,256,r,False]/d[p,256,r,True] for r in range(3)]) for p in [64,512]}
pooled=ratio_summary([np.sqrt((d[64,256,r,False]/d[64,256,r,True])*(d[512,256,r,False]/d[512,256,r,True])) for r in range(3)])
lo,hi=pooled['ci95']; verdict='PROVEN' if lo>=5 else ('DISPROVEN' if hi<5 else 'INCONCLUSIVE')
summary=[]; fits=[]
fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
for ax,p in zip(axes,[64,512]):
    for cache,color in [(True,'#167d9a'),(False,'#cc592a')]:
        x=np.array([32,64,128,256],dtype=float)
        values=np.array([[d[p,int(n),r,cache] for r in range(3)] for n in x]); y=values.mean(axis=1)
        ax.errorbar(x,y,yerr=values.std(axis=1,ddof=1),marker='o',color=color,label='Cache on' if cache else 'Cache off',capsize=3)
        for n,ts in zip(x,values):
            summary.append(dict(prompt_tokens=p,output_tokens=int(n),use_cache=cache,mean_seconds=float(ts.mean()),sd_seconds=float(ts.std(ddof=1)),mean_tokens_per_second=float(np.mean(n/ts))))
        fit=dict(prompt_tokens=p,use_cache=cache,loglog_slope=float(np.polyfit(np.log(x),np.log(y),1)[0]))
        for degree,name in [(1,'linear'),(2,'quadratic')]:
            coeff=np.polyfit(x,y,degree); pred=np.polyval(coeff,x); fit[name]=dict(coefficients=coeff.tolist(),r2=float(1-((y-pred)**2).sum()/((y-y.mean())**2).sum()))
        fits.append(fit)
    ax.set(title=f'{p}-token prompt',xlabel='New tokens'); ax.set_xticks([32,64,128,256]); ax.grid(alpha=.2); ax.legend()
axes[0].set_ylabel('Generation time (seconds; mean ± SD, n=3)')
fig.suptitle(f'Qwen3-0.6B · CPU float32 · cache off: {pooled["ratio"]:.1f}× slower at 256 tokens')
fig.tight_layout();fig.savefig(ROOT/'chart.png',dpi=180)
with (R/'summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=summary[0].keys()); w.writeheader();w.writerows(summary)
matching=sum(next(r['output_sha256'] for r in rows if int(r['prompt_tokens'])==p and int(r['output_tokens'])==n and int(r['repeat'])==rep and r['use_cache']=='True')==next(r['output_sha256'] for r in rows if int(r['prompt_tokens'])==p and int(r['output_tokens'])==n and int(r['repeat'])==rep and r['use_cache']=='False') for p,n,rep in itertools.product([64,512],[32,64,128,256],range(3)))
result=dict(verdict=verdict,primary=pooled,per_prompt=per_prompt,fits=fits,matched_output_hash_pairs=matching,total_pairs=24,total_timed_seconds=sum(d.values()))
(R/'analysis.json').write_text(json.dumps(result,indent=2))
lines=[f'# KV cache: {verdict}', '',f'**Headline: Turning off the KV cache made the model {pooled["ratio"]:.1f}× slower at 256 tokens.**','',f'Primary geometric-mean slowdown across both prompt lengths: {pooled["ratio"]:.2f}× (paired-repeat bootstrap 95% CI {lo:.2f}–{hi:.2f}×). Predeclared threshold: at least 5×. Three matched repeats; 20,000 bootstrap draws, seed 121.','', '| Prompt tokens | New tokens | Cache | Total seconds, mean ± SD | Tokens/s, mean |','|---:|---:|:---|---:|---:|']
for r in summary:
    lines.append(f'| {r["prompt_tokens"]} | {r["output_tokens"]} | {"on" if r["use_cache"] else "off"} | {r["mean_seconds"]:.2f} ± {r["sd_seconds"]:.2f} | {r["mean_tokens_per_second"]:.2f} |')
lines+=['','At 256 new tokens:']
for p,s in per_prompt.items(): lines.append(f'- {p}-token prompt: {s["ratio"]:.2f}× slower (95% CI {s["ci95"][0]:.2f}–{s["ci95"][1]:.2f}×).')
lines+=['','## Growth with output length','','Fits use the four cell means, with an intercept. Linear: t = aN + b; quadratic: t = aN² + bN + c. Coefficients in results/analysis.json. Quadratic fits have only one residual degree of freedom; R² is descriptive, not a test of complexity.','','| Prompt | Cache | Log-log slope | Linear R² | Quadratic R² |','|---:|:---|---:|---:|---:|']
for f in fits: lines.append(f'| {f["prompt_tokens"]} | {"on" if f["use_cache"] else "off"} | {f["loglog_slope"]:.3f} | {f["linear"]["r2"]:.5f} | {f["quadratic"]["r2"]:.5f} |')
lines+=['','## Cache memory','','The [model config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json) has 28 layers, 8 KV heads (16 query heads), and head dimension 128. In this float32 run: **28 × 8 × 128 × 2 × 4 × T = 229,376T bytes = 224 KiB per cached token** (batch size 1). BF16 would use half these bytes.','','| Cached tokens | Formula MiB | Measured tensor MiB | Backing storage MiB |','|---:|---:|---:|---:|']
for m in json.loads((R/'memory.json').read_text()): lines.append(f'| {m["tokens"]} | {m["formula_bytes"]/2**20:.2f} | {m["tensor_bytes"]/2**20:.2f} | {m["storage_bytes"]/2**20:.2f} |')
lines+=['','At 256 generated tokens the final cache normally covers 319 tokens for the short prompt (69.78 MiB) and 767 for the long prompt (167.78 MiB), since the last selected token has not yet been forwarded. These are formula estimates; the table directly measures prompt-forward tensors. Storage may exceed logical tensor bytes because tensors can be views; neither measure includes weights, activations, allocator overhead, or process RSS.','','## Interpretation and limits','',f'All 48 runs generated the requested lengths. Output hashes matched in {matching}/24 cache-on/off pairs; floating-point differences can change greedy sequences. Timed generation includes prefill and Python overhead; throughput is new tokens divided by total time. Fixed synthetic token sequences are saved in results/prompts.json; no chat template was used. This tests computational cost, not answer quality.','', 'With cache, projections and feed-forward work for previous tokens are reused, while attention still reads a growing context. Without cache, each step recomputes the prompt and generated prefix: summing that work gives roughly PN + N² behavior, with an additional attention contribution; large fixed prompts and efficient matrix operations can obscure a quadratic curve across only 32–256 tokens. See [Hugging Face’s caching explanation](https://huggingface.co/docs/transformers/en/cache_explanation).','','Only three timing repeats, two related synthetic prompts, one CPU and one model were tested. The narrow bootstrap CI measures repeat variability, not uncertainty across workloads; shared-host contention and fixed run order may affect timings. CPU float32 is not representative of optimized GPU serving. GPU: none; Colab CU: 0.','','## Two-sentence takeaway','',f'Turning off the KV cache made this small model {pooled["ratio"]:.1f} times slower when writing 256 tokens on our CPU. The cache buys that speed by keeping earlier attention data in memory, using 224 KiB per token in this float32 setup.','']
(ROOT/'writeup.md').write_text('\n'.join(lines))
print(json.dumps(result,indent=2))
