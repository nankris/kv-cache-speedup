# 02 (121): The CPU pilot contradicts the GPU number in the same cell

- **Experiment:** 121 (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `121-kv-cache-speedup.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**The CPU pilot contradicts the GPU number in the same cell:** 6.21x at 64/32 on CPU vs 1.10x on GPU. That is consistent with the GPU being overhead-bound, not with the cache being weak.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
