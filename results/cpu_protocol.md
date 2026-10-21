# KV cache CPU experiment

KV cache saves the attention keys and values from earlier tokens so generation can reuse them.

Hypothesis: turning off the KV cache makes 256-token generation at least 5× slower on Qwen/Qwen3-0.6B on CPU.

## Fixed setup and decision rule (written before measurements)

Use Hugging Face Transformers, Qwen/Qwen3-0.6B, CPU float32, two PyTorch threads, greedy decoding, synthetic fixed prompts of exactly 64 and 512 tokens, and exactly 32, 64, 128, or 256 new tokens. Run three timed repeats per cell (48 runs); alternate cache order within matched repeats. Disable EOS stopping to keep output lengths fixed. Time model.generate only, including prompt prefill, excluding model loading/tokenization. Warm up both modes. Save each completed run immediately and skip it on resume.

Primary endpoint: for each repeat, take the geometric mean of the no-cache/cache total-time ratios across the two prompt lengths at 256 output tokens; summarize with their geometric mean. Bootstrap the three paired repeats with replacement (20,000 draws, seed 121) for a percentile 95% CI. PROVEN if the lower bound is >=5; DISPROVEN if the upper bound is <5; otherwise INCONCLUSIVE. This interval describes timing-repeat variability on this machine, not generalization across machines/prompts. Report each prompt length separately as well.

Fit descriptive linear and quadratic total-time curves versus output length for each prompt/cache setting and report R² and log-log slope. Four lengths cannot establish asymptotic complexity: cached attention still grows with context and uncached transformer work includes prompt recomputation.

Calculate logical KV tensor bytes as layers × KV heads × head dimension × 2 (K and V) × dtype bytes × cached tokens, using model config. Compare with actual tensor storage bytes and shapes from a direct forward pass; process RSS is not a reliable isolated KV measurement. Generation typically retains prompt + output − 1 tokens after its last forward pass.

Public model and synthetic data only. No paid APIs. CPU only; GPU: none; Colab CU: 0. Target wall time under 60 minutes. Model/library versions, raw timings, memory measurements, results and limitations will be recorded here or in results/ and writeup.md.

## Reproduce

```bash
source /workspace/tools/setup.sh
cd /workspace/work/tig/121-kv-cache-speedup
/workspace/tools/mlenv/bin/python scripts/benchmark.py > results/benchmark.log 2>&1
/workspace/tools/mlenv/bin/python scripts/analyze.py
```

The benchmark resumes from `results/timings.csv`. Preserve the environment and model revision when resuming. The complete endpoint analysis requires all 48 runs; otherwise the analysis reports missing data. It checks lengths, compares available paired output hashes, and generates `chart.png`, `writeup.md`, `results/summary.csv`, and `results/analysis.json`. Raw output text is not needed for timing, so output hashes are retained instead.

Sources: [Qwen model config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/config.json), [Transformers caching explanation](https://huggingface.co/docs/transformers/en/cache_explanation).

## Outcome: feasibility failure; INCONCLUSIVE

The requested 256-token endpoint was **not measured**. Only 3 of the original 48 float32 timings completed: a cache on/off pair for a 64-token prompt with 32 output tokens, plus one cached run for a 512-token prompt with 32 output tokens. There is no valid 256-token headline multiplier, no repeat-based CI, and insufficient lengths for growth fits. See `writeup.md` and `chart.png`; missing cells are explicit. Do not publish the 32-token pilot ratio as a 256-token result.

Resumed this folder on 2026-10-16 with its committed protocol intact and no completed timed rows. The scripts pin model/tokenizer revision `c1899de289a04d12100db370d81485cdf75e47ca`. Actual environment: aarch64 Neoverse-V2, two CPU threads, Torch 2.14.1+cpu, Transformers 5.19.0, Python 3.12.15. Generation was stopped to honor the small, cheap experiment scope: the first 32-token uncached float32 run took 57.72 seconds, and a separate 512-token prompt forward took 18–19 seconds. Library loading alone took several minutes on repeated starts. Longer runs were interrupted; partial elapsed times are not recorded as completed generations.

A documented feasibility amendment was attempted before any 256-token measurement: BF16 forwards, then dynamic int8 transformer linear weights with qnnpack (float32 activations/KV). BF16 was slower; two int8 setup processes were OOM-killed, confirmed by cgroup counters. A third int8 attempt succeeded by loading published BF16 weights, quantizing one layer at a time, trimming the allocator, and keeping the tied output head in float32. It was also slower at 32 tokens (75.17 s uncached), so it was abandoned. Int8 and precision-pilot data remain separate under `results/int8_pilot/` and `results/precision_pilot.*`; they are excluded from the original float32 result. `scripts/int8_feasibility.py` reproduces the final int8 attempt in that separate results folder. The final resumable `scripts/benchmark.py` implements the original float32 protocol.

`analyze.py` reports an honest incomplete experiment when fewer than 48 rows exist; once all 48 are present it calls `analyze_complete.py` to apply the original fixed decision rule. Cache memory was successfully verified: logical bytes and tensor backing storage exactly match 28 × 8 × 128 × 2 × 4 × T = 229,376T bytes at 64 and 512 cached tokens (14 and 112 MiB).

GPU: none. Colab CU: **0**. No paid APIs. Runtime status tooling and the network-policy snapshot were unavailable; public Hugging Face access used the inherited network setup. No other experiment folders or sessions were modified. Setup-failure logs and all completed pilot rows are retained.
