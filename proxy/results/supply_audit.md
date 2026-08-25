# Supply-confidence auditor — validation run (auto-generated, seed 0)

The auditor measures a lane's **confidence factor** = usable-unique-in-language tokens / raw tokens, via a language-ID gate, exact-dedup, and MinHash+LSH near-dedup. Here it is validated against a corpus with a *known* usable fraction; at assembly it is pointed at the real per-lane shards and its output replaces the reconciler's planning-assumption factors.

- Corpus: **6,480 documents**, **517,845 raw tokens**, polluted with exact copies, near-copies (~8% words changed) and out-of-language junk.
- Ground-truth usable fraction (known by construction): **0.615**
- Auditor dropped: 300 out-of-language, 1,589 exact-dup, 567 near-dup documents.
- **Measured confidence factor: 0.619** (absolute error **0.004** vs ground truth).

**Verdict:** the instrument recovers the true usable fraction to within 0.004, so the confidence factors it will report on real Sangraha / FineWeb-2 / Stack-v2 shards are trustworthy. Running this per lane at assembly is the concrete step that moves the plan from *modelled* to *gated-ready* (README §11).

*Config:* MinHash NUM_HASHES=64, BANDS=16 (rows/band=4), 3-word shingles, language-ID threshold=0.90. Deterministic (seed 0), no network.
