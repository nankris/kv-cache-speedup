# Turning off the KV cache made the model 1.93× slower at 256 tokens

**Verdict: DISPROVEN for the ≥5× hypothesis. Headline number: 1.93×** (95% paired bootstrap CI 1.92–1.94×).

The headline is Qwen3-0.6B on one NVIDIA L4, FP32 SDPA with TF32 disabled: the geometric mean of two prompt-length ratios within each repeat, then across three repeats. Each timing includes prompt prefill and greedy decoding; outputs have fixed lengths. The upper CI bound falls below the predeclared 5× threshold.

## Main results

| Model | Prompt | Output | Cache on s | Cache off s | On tok/s | Off tok/s | Slowdown [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 64 | 32 | 1.23 | 1.36 | 25.9 | 23.5 | 1.103 [1.096, 1.117] |
| Qwen3-0.6B | 64 | 64 | 2.44 | 2.69 | 26.2 | 23.8 | 1.099 [1.096, 1.104] |
| Qwen3-0.6B | 64 | 128 | 4.93 | 5.40 | 25.9 | 23.7 | 1.094 [1.080, 1.108] |
| Qwen3-0.6B | 64 | 256 | 9.81 | 11.42 | 26.1 | 22.4 | 1.164 [1.163, 1.165] |
| Qwen3-0.6B | 64 | 512 | 19.58 | 31.62 | 26.1 | 16.2 | 1.615 [1.611, 1.617] |
| Qwen3-0.6B | 512 | 32 | 1.26 | 2.97 | 25.4 | 10.8 | 2.358 [2.339, 2.372] |
| Qwen3-0.6B | 512 | 64 | 2.48 | 6.10 | 25.8 | 10.5 | 2.461 [2.454, 2.470] |
| Qwen3-0.6B | 512 | 128 | 4.93 | 13.48 | 26.0 | 9.5 | 2.734 [2.667, 2.783] |
| Qwen3-0.6B | 512 | 256 | 9.82 | 31.42 | 26.1 | 8.1 | 3.199 [3.173, 3.242] |
| Qwen3-0.6B | 512 | 512 | 19.65 | 79.60 | 26.1 | 6.4 | 4.052 [4.045, 4.061] |

Times and tokens/s are arithmetic means; slowdown is the geometric mean of paired time ratios. CIs resample the three repeats. The same table is in figures/table.png.

## Primary endpoint

- Qwen3-0.6B: 1.93× [1.92, 1.94], DISPROVEN; paired repeat endpoints 1.942, 1.921, 1.925.

All 60 rows per model completed. Exact generated-token hashes match across both modes and all repeats in every cell: True. Token ID files are retained for independent checking.

## Growth fits

| Model | Prompt | Cache | Linear R² / LOO RMSE s | Quadratic R² / LOO RMSE s | Log-log slope |
|---|---:|---|---:|---:|---:|
| Qwen3-0.6B | 64 | on | 1.0000 / 0.029 | 1.0000 / 0.081 | 1.00 |
| Qwen3-0.6B | 64 | off | 0.9807 / 4.364 | 0.9996 / 2.746 | 1.12 |
| Qwen3-0.6B | 512 | on | 1.0000 / 0.018 | 1.0000 / 0.012 | 0.99 |
| Qwen3-0.6B | 512 | off | 0.9898 / 7.960 | 1.0000 / 0.614 | 1.19 |

Cached generation is nearly linear over this tested range (log-log slopes 0.99–1.00). For uncached generation, the quadratic fit predicts held-out lengths better at both prompt lengths; this describes curvature in these measurements.

Fits use mean total seconds over the five output lengths. LOO predicts a held-out output length; quadratic has one extra parameter. Finite lengths, prompt cost, GPU utilization and generation overhead prevent a claim of asymptotic linear or quadratic complexity. Coefficients are in results/gpu/growth_fits.json.

## KV memory

Formula: batch × layers × KV heads × head_dim × 2 (K,V) × dtype bytes × cached tokens. FP32 uses 4 bytes per element.

| Model | Tokens | Formula MiB | Tensor MiB | CUDA allocation delta MiB |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 64 | 14.000 | 14.000 | 14.580 |
| Qwen3-0.6B | 319 | 69.781 | 69.781 | 71.208 |
| Qwen3-0.6B | 512 | 112.000 | 112.000 | 112.580 |
| Qwen3-0.6B | 767 | 167.781 | 167.781 | 168.361 |
| Qwen3-0.6B | 1023 | 223.781 | 223.781 | 224.568 |

The direct forward retains logits for one token; the CUDA allocation increase includes those and any other persistent outputs. Peak and reserved allocation are stored separately and include activations or allocator pools. K/V backing storage is also recorded. Final generation cache length is prompt + output − 1, hence 319 and 767 tokens at the headline endpoint.

## Two-sentence takeaway

On this L4, turning off the model’s memory of earlier attention data made 256-token generation 1.93 times slower. Keeping that memory uses GPU RAM proportional to the cached context length.

## Caveats and CPU pilot

Only three timing repeats on one GPU, one fixed synthetic repeated paragraph at two lengths, batch size 1, greedy decoding and FP32 SDPA. Bootstrap intervals reflect repeat noise only and are especially coarse with n=3; they do not establish population uncertainty. Forced output lengths may differ from normal EOS-terminated chat. FP32 is a correctness-driven choice and is not representative of typical BF16 serving; the incomplete BF16 pilot had much smaller early speed ratios. Neither quality nor multi-user serving was tested; results depend on hardware, kernels, precision and sequence lengths. Mode order alternates across repeats but output lengths run in ascending order. No-cache recomputes prefixes; cached attention still reads an increasing context.

The earlier float32 CPU pilot completed one matched 64-prompt/32-output pair: 9.29 s cached versus 57.72 s uncached (6.21×), with matching outputs. It has no CI and is excluded from all GPU summaries; original files and feasibility failures are preserved.

Reproduction: run scripts/gpu_benchmark.py on the pinned GPU environment, then scripts/gpu_analyze.py locally. The benchmark resumes by cell and checks environment metadata. No private data or paid external APIs. See README for the predeclared decision rule, environment.json for exact model revisions, and figures/colab_run.txt for execution evidence.

## Correctness-driven precision amendment

The initial BF16 run stopped after its 512-prompt/256-output pair differed at generated token 212 (43 of 256 positions differed). Its 17 completed rows are preserved in results/gpu_bf16_exploratory and excluded from the headline. The optional Qwen3-1.7B matrix was not run: the FP32 primary timings implied a substantially longer run, so it did not meet the ‘if quick’ condition. The full publication matrix was restarted in FP32 with TF32 disabled under the same decision threshold. This is an amended FP32 result, not a BF16 performance claim.

KV caching reuses previously computed keys and values; see the [version-pinned Transformers explanation](https://huggingface.co/docs/transformers/v4.57.1/en/cache_explanation). Model dimensions come from the saved config and [pinned Qwen3-0.6B config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json). The untimed forward-hook mechanism check is saved in results/gpu/mechanism_check.json.

Exact primary setup: Python 3.13.15, PyTorch 2.11.0+cu130, Transformers 4.57.1, NVIDIA L4, FP32, TF32 disabled, SDPA, two CPU threads. Model revision: `c1899de289a04d12100db370d81485cdf75e47ca`. No quantization or compilation.

For this model, KV bytes/token = 28 layers × 8 KV heads × 128 head dimensions × 2 × 4 bytes = 229,376 bytes = 224 KiB/token. The generation-time cache at 256 output tokens therefore occupies 69.78 MiB for a 64-token prompt and 167.78 MiB for a 512-token prompt; the direct-forward memory checks measure these same context lengths.

GPU and cost: NVIDIA L4, stopped after 18.17 minutes. Estimated session usage **0.47 CU** from the observed 1.54 CU/hour account-rate drop on stopping this session. Account balance declined by 0.83 CU (163.67 → 162.84); another job running on the same account makes that a shared-account total, not this experiment’s exact bill. Exact per-session CU metering was unavailable. Lifecycle timestamps, balance excerpts and the estimate are saved in results/gpu/.
