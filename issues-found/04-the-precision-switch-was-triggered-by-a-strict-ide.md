# 04: The precision switch was triggered by a strict identical-output rule

- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST. Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**The precision switch was triggered by a strict identical-output rule.** BF16 divergence between cached and uncached paths is expected numerical nondeterminism (different kernels and reduction orders), not a bug. The rule forced FP32, which is "not representative of typical serving" (the writeup admits this).

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (the pre-registration rule), not an edit to these numbers.
