# Tier-blend probe — results (auto-generated, seed 0)

Controlled order-2 Markov language | V=64 | entropy floor **3.050 nats/token** | fixed compute 384,000 tokens/arm | scarce verified pool = 32,000 tokens.

Synthetic share `s`: s=0 fills the whole budget by repeating the scarce verified pool; s=1 trains only on fresh synthetic; in between is a blend. `verified_rep` is how many times the verified pool is repeated at that s.

### good synthetic (λ=0.20)
| Synthetic share s | Verified repetition | Val CE | vs best |
|--:|--:|--:|--:|
| 0.00 | 12.0x | 3.4451 | +0.0433 |
| 0.25 | 9.0x | 3.4232 | +0.0213 |
| 0.50 | 6.0x | 3.4156 | +0.0138 |
| 0.75 | 3.0x | 3.4029 | +0.0010 |
| 0.90 | 1.2x | 3.4019 | +0.0000  ← best |
| 1.00 | 0.0x | 3.4033 | +0.0015 |

**Optimal blend: s* = 0.90** (verified 10% / synthetic 90%); pure-verified-repeated (s=0) costs +0.0433 nats and pure-synthetic (s=1) costs +0.0015 nats vs the blend.

### poor synthetic (λ=0.80)
| Synthetic share s | Verified repetition | Val CE | vs best |
|--:|--:|--:|--:|
| 0.00 | 12.0x | 3.4700 | +0.0514 |
| 0.25 | 9.0x | 3.4330 | +0.0145 |
| 0.50 | 6.0x | 3.4218 | +0.0032 |
| 0.75 | 3.0x | 3.4185 | +0.0000  ← best |
| 0.90 | 1.2x | 3.4373 | +0.0188 |
| 1.00 | 0.0x | 3.4351 | +0.0166 |

**Optimal blend: s* = 0.75** (verified 25% / synthetic 75%); pure-verified-repeated (s=0) costs +0.0514 nats and pure-synthetic (s=1) costs +0.0166 nats vs the blend.

## Takeaway for the mixture
- The optimum is **interior in both regimes** — a bounded blend beats both repeating scarce verified to death (s=0) and going all-synthetic (s=1). This is the empirical basis for the plan giving verified, unverified, translated and synthetic each a *capped* share rather than betting the Indic lane on any single tier.
- The optimal synthetic share **moves with quality**: s*=0.90 when synthetic is good vs s*=0.75 when it is poor. That is exactly why the plan sets tier caps by *quality risk*, and why proxy arm A2 (verified-only vs +synthetic) is the gate that fixes the real Indic caps at 1B/3B scale.
- Pure-verified-repeated is never optimal here, which corroborates treating the verified tier's 6.0-epoch demand (see the ledger) as a deficit to close by cleaning, not an operating point.

*Caveat:* a ~12k-parameter toy far below 40B; offered as directional evidence for the unrun 1B/3B arms, not as the final caps.
