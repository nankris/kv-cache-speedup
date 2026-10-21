# 03 (121): The verdict depends on the arbitrary 5x threshold and on the endpoint definition

- **Experiment:** 121 (this repo)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `121-kv-cache-speedup.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**The verdict depends on the arbitrary 5x threshold and on the endpoint definition**: geomean of the 64- and 512-token prompts at 256 out. With longer contexts (4K+) or larger models the ratio would plausibly exceed 5x; the 512/512 cell is already 4.05x.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
