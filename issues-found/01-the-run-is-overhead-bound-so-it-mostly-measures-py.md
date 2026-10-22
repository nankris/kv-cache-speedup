# 01: The run is overhead-bound, so it mostly measures Python/launch overhead, not attention ...

- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**The run is overhead-bound, so it mostly measures Python/launch overhead, not attention compute.** 26 tok/s for a 0.6B model on an L4 is ~38 ms/step, far below what the GPU can do. Cached speed is identical at 64 and 512 prompt tokens. The BF16 exploratory run (results/gpu_bf16_exploratory/0_6B/timings.csv) shows ratios of **1.01-1.02x** in every cell, with uncached as fast as cached at 512/256 (9.75 vs 9.54 s). The per-step cost is dominated by HF generate's Python loop and kernel launches (2 CPU threads). The FP32 switch made each forward more compute-heavy, which *raised* the ratio. The headline 1.93x is a property of this inefficient harness, not a general "KV cache is only ~2x" fact. The post's framing ("short chats barely notice it") overgeneralizes.

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (the pre-registration rule), not an edit to these numbers.
