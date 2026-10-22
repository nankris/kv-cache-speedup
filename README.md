# KV cache speed on an NVIDIA L4

Question: How much does disabling KV caching slow greedy Qwen3 generation, and what memory does caching cost?

Hypothesis: no-cache generation is at least 5× slower at 256 new tokens on Qwen/Qwen3-0.6B. The GPU protocol below was written down before any GPU timing on 2026-10-17. The earlier CPU pilot and its original protocol remain in results/cpu_writeup.md and results/cpu_protocol.md; they do not enter GPU inference.

## Fixed GPU protocol and decision rule

Primary model Qwen/Qwen3-0.6B, pinned revision c1899de289a04d12100db370d81485cdf75e47ca; optionally Qwen/Qwen3-1.7B as a separately reported replication if time permits. NVIDIA L4, batch 1, FP32 weights/activations/KV with TF32 disabled (precision amendment below), Transformers SDPA, no compilation, greedy model.generate, use_cache=True versus False. Fixed synthetic prompts of exactly 64 and 512 tokens (the CPU pilot's repeated library paragraph), outputs 32/64/128/256/512 tokens, three paired repeats per cell. EOS stopping disabled. Warm up both modes. Alternate mode order across repeats. Synchronize CUDA around total generation time, including prefill and Python generation overhead, excluding tokenization/loading. Save every completed row immediately; resume by cell identity after checking environment. Verify complete generated token IDs are identical across modes and repeats for every cell. Any mismatch is reported explicitly and prevents a clean identical-output claim.

Primary endpoint: within each repeat, geometric mean of uncached/cached time ratios across the two prompt lengths at 256 output tokens. Report the geometric mean of the three repeat endpoints and a paired percentile bootstrap 95% CI (20,000 resamples, seed 121). PROVEN if lower CI bound >=5; DISPROVEN if upper bound <5; otherwise INCONCLUSIVE. Three repeats only measure local timing variability, not variation across prompts, machines or workloads. Report prompt-specific ratios and CIs, total times, and generated tokens/s for every cell. Require all 60 primary rows and matching outputs for the primary verdict; otherwise INCONCLUSIVE.

Descriptive growth analysis: fit intercept + N and intercept + N + N² to mean total seconds at the five output lengths for every model/prompt/mode; report R², leave-one-length-out RMSE and log-log slope. These five points cannot establish asymptotic complexity. Quadratic has an extra parameter; use held-out error when comparing fits.

KV formula: batch × layers × KV heads × head dimension × 2 × bytes per element × cached tokens. Compare formula with populated K/V tensor bytes, backing storage bytes, and torch.cuda.memory_allocated delta after a synchronized forward with logits_to_keep=1 at context lengths 64, 319, 512, 767 and 1023. Record baseline, live and peak CUDA allocation; the allocation delta includes logits/other persistent outputs and is not a pure KV tensor measurement. Reserved allocator memory is reported separately. Generation retains prompt + output − 1 cached tokens at the final forward.

Public Hugging Face model weights and synthetic prompts only; no private data or external paid APIs. Raw CPU pilot data remain unchanged. GPU results live in results/gpu; scripts and logs support independent reproduction. The Colab session was stopped on completion or failure; GPU time, CU and the final outcome are reported at the end of this file.

## Numerical-equivalence amendment (before the FP32 run)

The first BF16 512-prompt/256-output pair diverged in generated token hashes. The BF16 run was stopped, preserving completed rows in results/gpu_bf16_exploratory. It cannot support an identical-output speed comparison. Repeat the same fixed matrix and >=5× decision rule in **FP32**, with TF32 disabled, using separate results/gpu files; this becomes the publication comparison if outputs match. No BF16 timing is pooled into FP32. This change was motivated by the failed correctness check, not by selecting a favorable ratio (early BF16 ratios were close to 1). FP32 KV formula uses 4 bytes per element. If FP32 still diverges, report the hypothesis as INCONCLUSIVE and expose the mismatch rather than concealing it.

## Reproduction and artifacts

Local analysis needs Python with numpy and matplotlib. GPU dependencies: PyTorch from the Colab runtime (exact version recorded in each environment.json), transformers==4.57.1, accelerate, huggingface_hub. Upload scripts/gpu_benchmark.py to /content/kvcache/scripts/gpu_benchmark.py and run it as a Python process; all generation runs on CUDA. GPU script acquires an experiment-local lock to prevent duplicate benchmark processes. For interruption recovery, restore results/gpu to that experiment directory before relaunching in the same environment. Each saved generated-token JSON can be checked against its CSV hash. Run scripts/validate_gpu.py and scripts/gpu_analyze.py on downloaded results.

Figures: slowdown.png, timing.png, memory.png, and table.png in figures/. Historical chart.png is the original incomplete CPU pilot chart. Legacy scripts/benchmark.py and scripts/analyze.py reproduce only that original CPU protocol; use the gpu_ scripts for the current publication result. The colab_ scripts are operational helpers for the Colab session and work under /content/kvcache.

The usage CLI exposes only the account-level balance and rate, and other jobs on the same account may run at the same time. I recorded the balance before and after and disclose below that CU cannot be attributed precisely.

Optional-model scope: the FP32 primary timing (31.74 s uncached for the 512-prompt/256-output pair on its first repeat) implies the larger model's full matrix is not quick. Qwen3-1.7B is omitted; the full 0.6B matrix remains mandatory. scripts/gpu_benchmark.py defaults to the primary model; --models 0.6B 1.7B enables a separate larger-model replication for future work. A completion watcher terminates only this experiment's original two-model process after its primary timing and memory files finish, to avoid beginning the optional model.

## Final outcome and resource use

**DISPROVEN for the amended FP32 ≥5× hypothesis:** the 256-output-token slowdown is **1.93×**, paired repeat-bootstrap 95% CI **1.92–1.94×**. All 60 primary GPU runs completed and saved generated tokens match exactly across modes and repeats. The hook check confirms CUDA inputs of lengths 64,1,1,1 with caching and 64,65,66,67 without. KV backing storage exactly matches 224 KiB per cached FP32 token; the CUDA allocation increase also includes persistent non-KV outputs and allocator rounding. The CPU 6.21× short pilot remains a separate side note.

GPU: **NVIDIA L4**. The Colab session was created at 2026-10-17 05:37:19 UTC and terminated at 05:55:29 UTC (18.17 minutes). **Estimated experiment usage: 0.47 CU**, using session duration × the observed 1.54 CU/hour reduction in account rate after stopping this session. The account balance fell 163.67 → 162.84 (**0.83 CU total**), including another job running on the same account. Exact per-session CU metering is unavailable, so 0.47 CU is explicitly an estimate; raw account/lifecycle evidence and calculation are in results/gpu/cost.json and usage_*.txt. No A100, paid external API or private data was used.
