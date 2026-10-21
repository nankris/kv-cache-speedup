# KV cache CPU experiment: INCONCLUSIVE

**Requested headline: unavailable. No 256-token cache-on/off pair completed; no N× value can honestly be given.**

The original >=5× hypothesis is INCONCLUSIVE because the required endpoint is missing. There is no 95% CI, and no valid linear/quadratic growth fit: only one output length was measured. This is a feasibility failure, not evidence against KV caching.

## Completed float32 pilot results

Qwen/Qwen3-0.6B, CPU, float32, two Torch threads, greedy, model.generate including prefill. Each row below is ONE completed run, not the requested three-repeat estimate.

| Prompt tokens | Output tokens | Cache | Time (s) | Tokens/s | Completed repeats |
|---:|---:|:---|---:|---:|---:|
| 64 | 32 | on | 9.29 | 3.444 | 1/3 |
| 64 | 32 | off | 57.72 | 0.554 | 1/3 |
| 512 | 32 | on | 10.79 | 2.966 | 1/3 |

The one matched 64-prompt/32-output pair gives a **6.21× pilot slowdown**, with identical output hashes. This is not the requested headline, has no repeat-based CI, and must not be extrapolated to 256 tokens.

## Requested matrix completion

| Prompt | Output | Cache-on repeats | Cache-off repeats |
|---:|---:|---:|---:|
| 64 | 32 | 1/3 | 1/3 |
| 64 | 64 | 0/3 | 0/3 |
| 64 | 128 | 0/3 | 0/3 |
| 64 | 256 | 0/3 | 0/3 |
| 512 | 32 | 1/3 | 0/3 |
| 512 | 64 | 0/3 | 0/3 |
| 512 | 128 | 0/3 | 0/3 |
| 512 | 256 | 0/3 | 0/3 |

## Why the experiment stopped

The original float32 uncached 32-token pilot already took 57.72 seconds. A separate forward-only precision pilot measured 18.48 and 19.00 seconds for a 512-token prompt in float32, and BF16 was slower at 64 tokens. These are feasibility measurements, not substitutes for generation. Completing the full original matrix was projected to take hours, beyond the small experiment scope.

An int8 feasibility amendment was attempted before observing the primary endpoint. Two setup attempts were OOM-killed; a lower-memory conversion succeeded, but its 32-token uncached run took 75.17 seconds versus 12.63 seconds cached. These are different numerical representations and output hashes differ in that int8 pair. Int8 results are archived separately and excluded from the float32 table and inference. No partial interrupted generation was treated as a completed run.

## KV cache memory: successfully checked

The [pinned Qwen config](https://huggingface.co/Qwen/Qwen3-0.6B/blob/c1899de289a04d12100db370d81485cdf75e47ca/config.json) has 28 layers, 8 KV heads (16 query heads), and a 128-dimensional head. At batch size 1 in float32:

**layers × KV heads × head_dim × 2 × bytes × tokens = 28 × 8 × 128 × 2 × 4 × T = 229,376T bytes = 224 KiB/token.**

| Cached tokens | Formula MiB | Tensor MiB | Backing storage MiB |
|---:|---:|---:|---:|
| 64 | 14.00 | 14.00 | 14.00 |
| 512 | 112.00 | 112.00 | 112.00 |

The formula exactly matches populated K/V tensors in all 28 layers. Each tensor shape is [1, 8, T, 128]. This measures cache tensor storage, excluding weights, other activations, and allocator overhead; process RSS cannot isolate the cache. At 256 new tokens, prompt + output − 1 predicts 319 cached tokens (69.78 MiB) or 767 (167.78 MiB); those generation-time sizes are formula estimates, not measured endpoints. BF16 KV tensors would use half the bytes, but the original benchmark uses float32.

## Growth fits and limits

Linear and quadratic growth fits are unavailable: four output lengths with three repeats each were requested, and only one length has observations. [Hugging Face explains](https://huggingface.co/docs/transformers/en/cache_explanation) that KV caching reuses earlier keys and values; cached attention still reads the growing context. Without caching, each step recomputes the prefix, but this experiment did not measure the requested scaling curves.

Public model and synthetic prompts only; no chat template, EOS stopping disabled, fixed output lengths. Pinned revision, library versions, prompt IDs, raw completed rows, failed setup logs, and separate feasibility pilots are saved in results/. Main scripts resume float32 from results/timings.csv and apply the predeclared bootstrap/decision rule only once all 48 rows exist. GPU: none. Colab CU: **0**.

## Two-sentence plain-English takeaway

The short pilot showed a clear speed benefit from remembering earlier attention data, but this run did not finish the 256-token comparison needed for the post’s headline. We did verify the memory cost: this model stores 224 KiB of float32 attention data per cached token.
