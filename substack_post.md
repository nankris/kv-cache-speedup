TITLE: What happens if you turn off the KV cache? 1.93x slower at 256 tokens, not 5x
SUBTITLE: A measured test on Qwen3-0.6B and an NVIDIA L4. The 5x claim is disproven, but the slowdown grows with context.

You will often hear that the KV cache makes LLM generation "many times faster". I wanted a number. So I turned the cache off, kept everything else fixed, and timed it. My predeclared claim was a slowdown of at least 5x at 256 output tokens.

Verdict: DISPROVEN for 5x. The measured slowdown was 1.93x (95% CI 1.92 to 1.94). The upper bound sits well below my 5x threshold.

[IMAGE table.png]

Setup

Model: Qwen3-0.6B. GPU: one NVIDIA L4 on Colab, about 18 minutes, roughly 0.47 CU. Precision: FP32 with TF32 disabled and SDPA attention. Batch size 1, greedy decoding, fixed output lengths, so the work is the same in both modes. I tested two prompt lengths (64 and 512 tokens) and five output lengths (32 to 512), with three repeats.

Each timing includes prompt prefill plus decoding. I checked that the generated token hashes match between cache on and cache off in every cell, so the speed comparison is fair.

The headline is the geometric mean of the two prompt-length ratios at 256 output tokens, then across the three repeats: 1.93x. The three repeats gave 1.942, 1.921 and 1.925.

What the numbers say

With the cache on, speed is flat at about 26 tokens per second whatever the length. With the cache off, it drops as the context grows.

At a 64-token prompt and 32 output tokens, the slowdown is only 1.103x (CI 1.096 to 1.117). At 64 prompt and 256 output it is 1.164x (CI 1.163 to 1.165). At 512 prompt and 256 output it is 3.199x (CI 3.173 to 3.242). The worst cell was 512 prompt and 512 output: 19.65 s with the cache versus 79.60 s without, a 4.052x slowdown (CI 4.045 to 4.061).

So the cache is not a flat "Nx" trick. The benefit depends on how much context you carry. Short chats barely notice it. Long prompts and long answers pay a lot.

[IMAGE slowdown.png]

[IMAGE timing.png]

The growth shape

Cached generation is nearly linear in output length (log-log slopes 0.99 to 1.00). Uncached generation curves upward: a quadratic fit predicted held-out lengths better than a linear one at both prompt lengths. I only claim curvature in these measurements, not asymptotic complexity.

What the cache costs in memory

For this model the cache needs 28 layers x 8 KV heads x 128 head dimensions x 2 x 4 bytes, which is 229,376 bytes or 224 KiB per token in FP32. At 512 tokens that is 112.000 MiB, and at 1023 tokens 223.781 MiB. The measured CUDA allocation delta was 112.580 MiB and 224.568 MiB, so the formula holds.

[IMAGE memory.png]

Verdict

DISPROVEN for the 5x hypothesis. The honest summary is that the KV cache gave 1.93x at 256 output tokens in this setup, and up to 4.05x at the longest cell I tested. It costs GPU memory proportional to context length.

Caveats

Only three repeats on one GPU, so the bootstrap intervals reflect repeat noise only and are coarse. One synthetic repeated paragraph, batch size 1, greedy decoding. FP32 is not typical serving precision. My first BF16 run stopped because outputs diverged at token 212 in one pair, so I reran in FP32 and kept the BF16 rows out of the headline. Quality and multi-user serving were not tested. The optional Qwen3-1.7B run was skipped because it would have run too long.

A side note: an earlier CPU pilot with one matched pair (64 prompt, 32 output) gave 9.29 s cached versus 57.72 s uncached, 6.21x. It has no CI and is excluded from all GPU summaries. It suggests the ratio depends heavily on hardware.

What to try next

Repeat on BF16 with a fixed kernel, try longer contexts (4K and up), test batch sizes above 1, and compare a bigger model where attention is a larger share of the work.

All code and raw results are in this repo. If you want more measured tests like this, subscribe to Tech in General.
