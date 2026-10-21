"""Analyze only GPU checkpoints; retain the CPU pilot unchanged."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; R=ROOT/'results/gpu'; F=ROOT/'figures'; F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.titlesize':13,'figure.dpi':150,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
summary=[]; fits=[]; endpoints=[]; checks=[]; memories=[]
def boot(x):
    x=np.array(x); rng=np.random.default_rng(121)
    draws=np.exp(np.log(rng.choice(x,(20000,len(x)),replace=True)).mean(axis=1))
    return float(np.exp(np.log(x).mean())),*np.quantile(draws,[.025,.975]).tolist()
for d in sorted(R.glob('*')):
    if not (d/'timings.csv').exists(): continue
    df=pd.read_csv(d/'timings.csv'); model=json.loads((d/'environment.json').read_text())['model'].split('/')[-1]
    assert len(df)==60 and not df.duplicated(['prompt_tokens','output_tokens','repeat','use_cache']).any()
    assert (df.generated_tokens==df.output_tokens).all()
    checks.append(dict(model=model,rows=len(df),all_outputs_identical=bool((df.groupby(['prompt_tokens','output_tokens']).output_sha256.nunique()==1).all())))
    wide=df.pivot(index=['prompt_tokens','output_tokens','repeat'],columns='use_cache',values='seconds')
    wide['ratio']=wide[False]/wide[True]
    for (p,n),g in df.groupby(['prompt_tokens','output_tokens']):
        rat=wide.loc[(p,n),'ratio'].to_numpy(); ratio,lo,hi=boot(rat)
        on=g[g.use_cache]; off=g[~g.use_cache]
        summary.append(dict(model=model,prompt_tokens=p,output_tokens=n,cached_seconds=on.seconds.mean(),uncached_seconds=off.seconds.mean(),cached_tps=on.tokens_per_second.mean(),uncached_tps=off.tokens_per_second.mean(),slowdown=ratio,ci_low=lo,ci_high=hi))
    ratios=wide.xs(256,level='output_tokens')['ratio'].unstack('prompt_tokens')
    endpoint=np.exp(np.log(ratios).mean(axis=1)); val,lo,hi=boot(endpoint)
    identical=checks[-1]['all_outputs_identical']
    verdict=('PROVEN' if lo>=5 else 'DISPROVEN' if hi<5 else 'INCONCLUSIVE') if identical else 'INCONCLUSIVE'
    endpoints.append(dict(model=model,slowdown=val,ci_low=lo,ci_high=hi,repeat_ratios=endpoint.tolist(),verdict=verdict))
    for (p,cache),g in df.groupby(['prompt_tokens','use_cache']):
        means=g.groupby('output_tokens').seconds.mean(); x=means.index.to_numpy(float); y=means.to_numpy()
        for degree in [1,2]:
            co=np.polynomial.polynomial.polyfit(x,y,degree); pred=np.polynomial.polynomial.polyval(x,co)
            errors=[]
            for i in range(len(x)):
                mask=np.arange(len(x))!=i
                c=np.polynomial.polynomial.polyfit(x[mask],y[mask],degree)
                errors.append((y[i]-np.polynomial.polynomial.polyval(x[i],c))**2)
            fits.append(dict(model=model,prompt_tokens=p,use_cache=bool(cache),degree=degree,r2=1-sum((y-pred)**2)/sum((y-y.mean())**2),loo_rmse_seconds=float(np.sqrt(np.mean(errors))),coefficients=co.tolist(),log_log_slope=float(np.polyfit(np.log(x),np.log(y),1)[0])))
    for m in json.loads((d/'memory.json').read_text()): memories.append(dict(model=model,**m))
s=pd.DataFrame(summary); s.to_csv(R/'summary.csv',index=False)
pd.DataFrame(fits).to_json(R/'growth_fits.json',orient='records',indent=2)
(R/'analysis.json').write_text(json.dumps(dict(endpoints=endpoints,checks=checks),indent=2))
pd.DataFrame(memories).to_csv(R/'memory_summary.csv',index=False)
models=list(s.model.unique()); colors=['#167d9a','#df7735']
fig,axs=plt.subplots(1,len(models),figsize=(6*len(models),4.4),squeeze=False)
for ax,model in zip(axs[0],models):
    for p,c in zip([64,512],colors):
        g=s[(s.model==model)&(s.prompt_tokens==p)]
        ax.errorbar(g.output_tokens,g.slowdown,yerr=[g.slowdown-g.ci_low,g.ci_high-g.slowdown],marker='o',capsize=4,label=f'{p}-token prompt',color=c)
    ax.axhline(5,ls='--',color='gray',label='5× hypothesis'); ax.set(title=f'{model} · L4 · FP32',xlabel='Generated tokens',ylabel='No-cache / cached total time'); ax.legend(); ax.grid(alpha=.2)
fig.suptitle('How much slower is generation without the KV cache?'); fig.text(.5,.01,'Error bars: 95% paired bootstrap CI over 3 repeats',ha='center',fontsize=9); fig.tight_layout(rect=[0,.04,1,1]); fig.savefig(F/'slowdown.png'); plt.close(fig)
fig,axs=plt.subplots(1,len(models),figsize=(6*len(models),4.4),squeeze=False)
for ax,model in zip(axs[0],models):
    for p,c in zip([64,512],colors):
        g=s[(s.model==model)&(s.prompt_tokens==p)]
        ax.plot(g.output_tokens,g.cached_seconds,'o-',color=c,label=f'Cache on, prompt {p}')
        ax.plot(g.output_tokens,g.uncached_seconds,'s--',color=c,label=f'Cache off, prompt {p}')
    ax.set(title=f'{model} · L4 · FP32',xlabel='Generated tokens',ylabel='Mean total generation time (s)'); ax.legend(fontsize=9); ax.grid(alpha=.2)
fig.suptitle('Generation time includes prompt prefill and all decode steps'); fig.tight_layout(); fig.savefig(F/'timing.png'); plt.close(fig)
fig,axs=plt.subplots(1,len(models),figsize=(6*len(models),4.4),squeeze=False)
m=pd.DataFrame(memories)
for ax,model in zip(axs[0],models):
    g=m[m.model==model]
    ax.plot(g.tokens,g.formula_bytes/2**20,'-',lw=3,label='KV formula'); ax.scatter(g.tokens,g.tensor_bytes/2**20,marker='o',s=45,label='Live KV tensor bytes'); ax.scatter(g.tokens,g.delta_bytes/2**20,marker='x',s=60,label='CUDA allocation increase')
    ax.set(title=f'{model} · L4 · FP32',xlabel='Cached context tokens',ylabel='Memory (MiB)'); ax.legend(fontsize=9); ax.grid(alpha=.2)
fig.suptitle('FP32 cache memory: formula versus measured CUDA allocation'); fig.tight_layout(); fig.savefig(F/'memory.png'); plt.close(fig)
# Complete main results table as an image for Substack.
labels=['Model','Prompt','Output','On (s)','Off (s)','On tok/s','Off tok/s','Slowdown [95% CI]']
rows=[]
for r in summary:
    rows.append([r['model'],str(r['prompt_tokens']),str(r['output_tokens']),f"{r['cached_seconds']:.2f}",f"{r['uncached_seconds']:.2f}",f"{r['cached_tps']:.1f}",f"{r['uncached_tps']:.1f}",f"{r['slowdown']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"])
fig,ax=plt.subplots(figsize=(13,1.1+.36*len(rows))); ax.axis('off'); table=ax.table(cellText=rows,colLabels=labels,loc='center',cellLoc='center',colWidths=[.09,.07,.07,.09,.09,.11,.11,.25]); table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1,1.5)
for (i,j),cell in table.get_celld().items():
    cell.set_edgecolor('#dddddd')
    if i==0: cell.set_facecolor('#163c51'); cell.set_text_props(color='white',weight='bold')
    elif i%2==0: cell.set_facecolor('#edf3f6')
primary=endpoints[0]
ax.set_title(f"L4 · FP32 · batch 1 · greedy · mean of 3 repeats\n256-token combined slowdown: {primary['slowdown']:.2f}× [95% CI {primary['ci_low']:.2f}–{primary['ci_high']:.2f}]",pad=15); fig.tight_layout(); fig.savefig(F/'table.png',bbox_inches='tight'); plt.close(fig)
e=endpoints[0]
lines=[f"# Turning off the KV cache made the model {e['slowdown']:.2f}× slower at 256 tokens",'',f"**Verdict: {e['verdict']} for the ≥5× hypothesis. Headline number: {e['slowdown']:.2f}×** (95% paired bootstrap CI {e['ci_low']:.2f}–{e['ci_high']:.2f}×).",'', 'The headline is Qwen3-0.6B on one NVIDIA L4, FP32 SDPA with TF32 disabled: the geometric mean of two prompt-length ratios within each repeat, then across three repeats. Each timing includes prompt prefill and greedy decoding; outputs have fixed lengths. The upper CI bound falls below the predeclared 5× threshold.', '', '## Main results', '', '| Model | Prompt | Output | Cache on s | Cache off s | On tok/s | Off tok/s | Slowdown [95% CI] |','|---|---:|---:|---:|---:|---:|---:|---:|']
for row in rows: lines.append('| '+' | '.join(row)+' |')
lines+=['','Times and tokens/s are arithmetic means; slowdown is the geometric mean of paired time ratios. CIs resample the three repeats. Use figures/table.png for publication.','', '## Primary endpoint','']
for e in endpoints: lines.append(f"- {e['model']}: {e['slowdown']:.2f}× [{e['ci_low']:.2f}, {e['ci_high']:.2f}], {e['verdict']}; paired repeat endpoints {', '.join(f'{v:.3f}' for v in e['repeat_ratios'])}.")
lines+=['',f"All 60 rows per model completed. Exact generated-token hashes match across both modes and all repeats in every cell: {all(c['all_outputs_identical'] for c in checks)}. Token ID files are retained for independent checking.",'','## Growth fits','','| Model | Prompt | Cache | Linear R² / LOO RMSE s | Quadratic R² / LOO RMSE s | Log-log slope |','|---|---:|---|---:|---:|---:|']
for model in models:
    for p in [64,512]:
        for cache in [True,False]:
            fs=[f for f in fits if f['model']==model and f['prompt_tokens']==p and f['use_cache']==cache]; a,b=fs
            lines.append(f"| {model} | {p} | {'on' if cache else 'off'} | {a['r2']:.4f} / {a['loo_rmse_seconds']:.3f} | {b['r2']:.4f} / {b['loo_rmse_seconds']:.3f} | {a['log_log_slope']:.2f} |")
lines+=['','Cached generation is nearly linear over this tested range (log-log slopes 0.99–1.00). For uncached generation, the quadratic fit predicts held-out lengths better at both prompt lengths; this describes curvature in these measurements.', '', 'Fits use mean total seconds over the five output lengths. LOO predicts a held-out output length; quadratic has one extra parameter. Finite lengths, prompt cost, GPU utilization and generation overhead prevent a claim of asymptotic linear or quadratic complexity. Coefficients are in results/gpu/growth_fits.json.','','## KV memory','','Formula: batch × layers × KV heads × head_dim × 2 (K,V) × dtype bytes × cached tokens. FP32 uses 4 bytes per element.','', '| Model | Tokens | Formula MiB | Tensor MiB | CUDA allocation delta MiB |','|---|---:|---:|---:|---:|']
for r in memories: lines.append(f"| {r['model']} | {r['tokens']} | {r['formula_bytes']/2**20:.3f} | {r['tensor_bytes']/2**20:.3f} | {r['delta_bytes']/2**20:.3f} |")
lines+=['','The direct forward retains logits for one token; the CUDA allocation increase includes those and any other persistent outputs. Peak and reserved allocation are stored separately and include activations or allocator pools. K/V backing storage is also recorded. Final generation cache length is prompt + output − 1, hence 319 and 767 tokens at the headline endpoint.','','## Two-sentence takeaway','',f"On this L4, turning off the model’s memory of earlier attention data made 256-token generation {endpoints[0]['slowdown']:.2f} times slower. Keeping that memory uses GPU RAM proportional to the cached context length.",'','## Caveats and CPU pilot','','Only three timing repeats on one GPU, one fixed synthetic repeated paragraph at two lengths, batch size 1, greedy decoding and FP32 SDPA. Bootstrap intervals reflect repeat noise only and are especially coarse with n=3; they do not establish population uncertainty. Forced output lengths may differ from normal EOS-terminated chat. FP32 is a correctness-driven choice and is not representative of typical BF16 serving; the incomplete BF16 pilot had much smaller early speed ratios. Neither quality nor multi-user serving was tested; results depend on hardware, kernels, precision and sequence lengths. Mode order alternates across repeats but output lengths run in ascending order. No-cache recomputes prefixes; cached attention still reads an increasing context.','','The earlier float32 CPU pilot completed one matched 64-prompt/32-output pair: 9.29 s cached versus 57.72 s uncached (6.21×), with matching outputs. It has no CI and is excluded from all GPU summaries; original files and feasibility failures are preserved.','','Reproduction: run scripts/gpu_benchmark.py on the pinned GPU environment, then scripts/gpu_analyze.py locally. The benchmark resumes by cell and checks environment metadata. No private data or paid external APIs. See README for the predeclared decision rule, environment.json for exact model revisions, and figures/colab_run.txt for execution evidence.']
(ROOT/'writeup.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(endpoints=endpoints,checks=checks),indent=2))

with (ROOT/'writeup.md').open('a') as f:
    f.write("\n## Correctness-driven precision amendment\n\nThe initial BF16 run stopped after its 512-prompt/256-output pair differed at generated token 212 (43 of 256 positions differed). Its 17 completed rows are preserved in results/gpu_bf16_exploratory and excluded from the headline. The optional Qwen3-1.7B matrix was not run: the FP32 primary timings implied a substantially longer run, so it did not meet the ‘if quick’ condition. The full publication matrix was restarted in FP32 with TF32 disabled under the same decision threshold. This is an amended FP32 result, not a BF16 performance claim.\n\nKV caching reuses previously computed keys and values; see the [version-pinned Transformers explanation](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation). Model dimensions come from the saved config and [pinned Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json). The untimed forward-hook mechanism check is saved in results/gpu/mechanism_check.json.\n")

env=json.loads((R/'0_6B/environment.json').read_text())
c=env['config']; per=c['num_hidden_layers']*c['num_key_value_heads']*c['head_dim']*2*4
with (ROOT/'writeup.md').open('a') as f:
    f.write(f"\nExact primary setup: Python {env['python']}, PyTorch {env['torch']}, Transformers {env['transformers']}, {env['gpu']}, FP32, TF32 disabled, SDPA, two CPU threads. Model revision: `{env['revision']}`. No quantization or compilation.\n\nFor this model, KV bytes/token = {c['num_hidden_layers']} layers × {c['num_key_value_heads']} KV heads × {c['head_dim']} head dimensions × 2 × 4 bytes = {per:,} bytes = {per/1024:.0f} KiB/token. The generation-time cache at 256 output tokens therefore occupies {319*per/2**20:.2f} MiB for a 64-token prompt and {767*per/2**20:.2f} MiB for a 512-token prompt; the direct-forward memory checks measure these same context lengths.\n")

if (R/'cost.json').exists():
    cost=json.loads((R/'cost.json').read_text())
    with (ROOT/'writeup.md').open('a') as f:
        f.write(f"\nGPU and cost: {cost['gpu']}, stopped after {cost['session_seconds']/60:.2f} minutes. Estimated session usage **{cost['estimated_session_cu']:.2f} CU** from the observed 1.54 CU/hour account-rate drop on stopping this session. Account balance declined by {cost['account_cu_delta']:.2f} CU ({cost['account_balance_before']:.2f} → {cost['account_balance_after']:.2f}); another concurrent assignment makes that a shared-account total, not this experiment’s exact bill. Exact per-session CU metering was unavailable. Lifecycle timestamps, balance excerpts and the estimate are saved in results/gpu/.\n")
