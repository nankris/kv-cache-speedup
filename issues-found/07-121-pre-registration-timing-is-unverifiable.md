# 07 (121): Pre-registration timing is unverifiable

- **Experiment:** 121 (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `121-kv-cache-speedup.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Pre-registration timing is unverifiable (see section 1). The protocol was amended twice (CPU -> GPU BF16 -> GPU FP32), but each amendment is documented with its reason. The BF16 rows are preserved and excluded (17 rows).

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
