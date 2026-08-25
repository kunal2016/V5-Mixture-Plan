#!/usr/bin/env python3
"""
V5 (India-first 40B) budget reconciler — the cheap check that runs *before* any
GPU is touched. It does one job: turn every budget share into a demand for
confidence-adjusted supply, reconcile that demand against the supply from the inventory,
and print the number of epochs each lane would have to be repeated to hit its
share. A lane that needs more than ~4 epochs is flagged, because Muennighoff et
al. (2023, "Scaling Data-Constrained Language Models") show that repeating data
beyond ~4 epochs buys almost nothing. This is the reconciliation that catches
"wishful" lanes — a large share sitting on top of almost no real data — which is
the exact failure mode the V5 mixture spec is written to avoid.

Run:  python3 budget_reconciler.py
Outputs: results/lane_ledger.md, results/indic_tier_ledger.md,
         results/lane_headroom.png, results/curriculum_ramps.png
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# --------------------------------------------------------------------------
# 0. Global budget
# --------------------------------------------------------------------------
TOTAL_B = 6000.0          # 6.0T total training tokens for the 40B run
ANNEAL_FRAC = 0.12        # 12% held back for the cooldown/anneal
STABLE_FRAC = 1 - ANNEAL_FRAC
SAFE_EPOCHS = 4.0         # repetition ceiling (Muennighoff et al., 2023)
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Lane mixture. stable% is share of the 5.28T stable phase;
#    anneal% is share of the 0.72T anneal reserve. Effective share of the
#    full 6T budget = stable*0.88 + anneal*0.12.
# --------------------------------------------------------------------------
# raw_B  = headline supply from the inventory / dataset cards (billions of tokens).
# conf   = a *supply-confidence discount* in (0,1]: how much of the raw figure we
#          trust as genuinely unique, high-quality, usable supply, after near-dedup,
#          quality filtering and source reliability. Raw sizes overstate real supply
#          because heavy near-duplication is the norm on web crawl; MinHash/SemDedup
#          studies (Lee et al. 2022; Abbas et al. 2023; the FineWeb report) retain
#          only ~0.5-0.8 of raw crawl, more for curated or freshly generated text.
#          Web crawl gets the steepest discount, curated/generated data the mildest,
#          and epochs are computed against the confidence-adjusted supply = raw*conf.
#          These factors are our own conservative planning assumptions until
#          re-derived from the assembled shards (see the readiness gate in the README).
# Figures tagged (est.) are labelled estimates, not inventory-measured.
lanes = [
    # name                     stable% anneal%  raw_B  conf  supply_note
    ("English general web",       34.0,  12.0,  5000.0, 0.85, "FineWeb 15T / DCLM-baseline 3.8T; high-quality subset"),
    ("Indic (all tiers)",         26.0,  24.0,  1245.0, 0.74, "sum of the four tiers below; blended discount"),
    ("Code",                      12.0,  14.0,   775.0, 0.80, "The Stack v2 dedup ~775B; further near-dup/license filtering"),
    ("Reference / knowledge",      9.0,   0.0,   350.0, 0.75, "books, Wikipedia, arXiv, S2ORC (est.)"),
    ("Math & STEM",                6.0,  18.0,   200.0, 0.70, "Nemotron-CC-Math 133B + OpenWebMath/FineMath (est.)"),
    ("Long-context",               4.0,   5.0,   150.0, 0.60, "native long docs; partly re-packed from other lanes (est.)"),
    ("Reasoning (long CoT)",       4.0,  16.0,    45.0, 0.55, "OpenThoughts3 / OpenR1 / distilled traces; heavy near-dup (est.)"),
    ("Multilingual (non-Indic)",   3.0,   0.0,  2000.0, 0.60, "CulturaX / MADLAD (est.)"),
    ("Agentic / tool-use",         2.0,   8.0,     2.0, 0.50, "xLAM-60k, Glaive, ToolBench, APIGen-MT seed (est.)"),
    ("Instruction / chat",         0.0,   3.0,    40.0, 0.65, "Tulu, OpenHermes, Aya, Indic SFT (est.)"),
]

def effective_share(stable_pct, anneal_pct):
    return stable_pct/100*STABLE_FRAC + anneal_pct/100*ANNEAL_FRAC

rows = []
for name, s, a, raw, conf, note in lanes:
    eff = effective_share(s, a)                 # fraction of 6T
    consumed = eff * TOTAL_B                     # B tokens fed to the model
    usable = raw * conf                        # confidence-adjusted supply
    epochs = consumed / usable if usable > 0 else float("inf")
    status = "OK" if epochs <= SAFE_EPOCHS else "WISHFUL"
    rows.append(dict(name=name, stable=s, anneal=a, eff=eff*100,
                     consumed=consumed, raw=raw, conf=conf, uniq=usable,
                     epochs=epochs, status=status, note=note))

# sanity: effective shares must sum to 100
tot = sum(r["eff"] for r in rows)
assert abs(tot - 100.0) < 1e-6, f"effective shares sum to {tot}, not 100"

# --------------------------------------------------------------------------
# 2. Indic tier accounting (share of the Indic lane).
# --------------------------------------------------------------------------
indic_lane = next(r for r in rows if r["name"].startswith("Indic"))
INDIC_CONSUMED = indic_lane["consumed"]         # ~1.55T
tiers = [
    # tier         share%  raw_B  conf  supply_note
    ("Verified",     28.0,  103.0, 0.70, "Sangraha-Verified 64.3B + IndicCorp v2 ~21B + Indic Wikipedia/curated ~18B; cross-source dup"),
    ("Unverified",   24.0,  400.0, 0.60, "Sangraha-Unverified 24.3B + Indic FineWeb-2 / CulturaX / MADLAD / HPLT; heavy near-dup so steepest discount"),
    ("Translated",   22.0,  340.0, 0.85, "EN->Indic MT of high-quality EN; near-unique by construction, capped for quality"),
    ("Synthetic",    26.0,  402.0, 0.80, "Sangraha-Synthetic 162.7B + model-generated Indic QA/CoT/romanized; dedup of generations"),
]
tier_rows = []
for name, share, raw, conf, note in tiers:
    consumed = share/100 * INDIC_CONSUMED
    usable = raw * conf                          # confidence-adjusted supply
    epochs = consumed / usable
    status = "OK" if epochs <= SAFE_EPOCHS else "WISHFUL"
    tier_rows.append(dict(name=name, share=share, consumed=consumed, raw=raw,
                          conf=conf, uniq=usable, epochs=epochs,
                          status=status, note=note))
assert abs(sum(t[1] for t in tiers) - 100.0) < 1e-6

# --------------------------------------------------------------------------
# 3. Write markdown tables
# --------------------------------------------------------------------------
def w(path, text):
    with open(os.path.join(RES, path), "w") as f:
        f.write(text)

lane_md = ["# Lane ledger (auto-generated by budget_reconciler.py)\n",
           f"Total budget **{TOTAL_B/1000:.1f}T tokens** | anneal reserve "
           f"**{ANNEAL_FRAC*100:.0f}%** ({TOTAL_B*ANNEAL_FRAC:.0f}B) | "
           f"safe repetition ceiling **{SAFE_EPOCHS:.0f} epochs**. "
           f"Supply is discounted by a per-lane supply-confidence factor.\n",
           "| Lane | Stable % | Anneal % | Eff % of 6T | Tokens fed (B) | "
           "Raw supply (B) | Confidence | Conf-adj supply (B) | Epochs | Verdict |",
           "|---|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
for r in rows:
    ep = "∞" if r["epochs"] == float("inf") else f"{r['epochs']:.2f}"
    flag = "✅ OK" if r["status"] == "OK" else "⚠️ **WISHFUL**"
    lane_md.append(f"| {r['name']} | {r['stable']:.0f} | {r['anneal']:.0f} | "
                   f"{r['eff']:.2f} | {r['consumed']:.0f} | {r['raw']:.0f} | "
                   f"{r['conf']:.2f} | {r['uniq']:.0f} | {ep} | {flag} |")
lane_md.append("\n**Supply notes**\n")
for r in rows:
    lane_md.append(f"- *{r['name']}*: {r['note']}")
w("lane_ledger.md", "\n".join(lane_md) + "\n")

tier_md = ["# Indic tier ledger (auto-generated by budget_reconciler.py)\n",
           f"Indic lane feeds **{INDIC_CONSUMED:.0f}B tokens** "
           f"({indic_lane['eff']:.1f}% of the 6T budget).\n",
           "| Tier | Share of Indic % | Tokens fed (B) | Raw supply (B) | "
           "Confidence | Conf-adj supply (B) | Epochs | Verdict |",
           "|---|--:|--:|--:|--:|--:|--:|:--|"]
for t in tier_rows:
    flag = "✅ OK" if t["status"] == "OK" else "⚠️ **WISHFUL**"
    tier_md.append(f"| {t['name']} | {t['share']:.0f} | {t['consumed']:.0f} | "
                   f"{t['raw']:.0f} | {t['conf']:.2f} | {t['uniq']:.0f} | "
                   f"{t['epochs']:.2f} | {flag} |")
tier_md.append("\n**Supply notes**\n")
for t in tier_rows:
    tier_md.append(f"- *{t['name']}*: {t['note']}")
w("indic_tier_ledger.md", "\n".join(tier_md) + "\n")

# --------------------------------------------------------------------------
# 4. Chart: supply vs demand (log scale), coloured by verdict
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
names = [r["name"] for r in rows]
y = range(len(names))
demand = [r["consumed"] for r in rows]
supply = [r["uniq"] for r in rows]
colors = ["#c0392b" if r["status"] == "WISHFUL" else "#2e86c1" for r in rows]
ax.barh([i+0.2 for i in y], demand, height=0.4, color=colors, label="Tokens fed (demand)")
ax.barh([i-0.2 for i in y], supply, height=0.4, color="#aab7b8", label="Confidence-adjusted supply")
ax.set_yticks(list(y)); ax.set_yticklabels(names)
ax.set_xscale("log"); ax.set_xlabel("Billions of tokens (log scale)")
ax.set_title("V5 mixture — demand vs. confidence-adjusted supply per lane\n(red = exceeds 4-epoch safe repetition)")
ax.invert_yaxis(); ax.legend(loc="lower right")
for i, r in enumerate(rows):
    if r["status"] == "WISHFUL":
        ep = "∞" if r["epochs"]==float("inf") else f"{r['epochs']:.0f}x"
        ax.text(r["consumed"]*1.1, i+0.2, ep, va="center", color="#c0392b", fontweight="bold")
plt.tight_layout(); plt.savefig(os.path.join(RES, "lane_headroom.png"), dpi=130)
plt.close()

# --------------------------------------------------------------------------
# 5. Chart: curriculum schedule (difficulty + reasoning-length ramp)
# --------------------------------------------------------------------------
import numpy as np
p = np.linspace(0, 1, 200)             # training progress 0..1
# difficulty band weights shift from foundational -> frontier
b1 = np.clip(1.0 - 1.3*p, 0.05, 1)     # foundational
b2 = np.exp(-((p-0.35)**2)/0.05)       # intermediate
b3 = np.exp(-((p-0.7)**2)/0.05)        # advanced
b4 = np.clip((p-0.55)*1.8, 0, 1)       # frontier (ramps late + anneal)
stack = np.vstack([b1, b2, b3, b4])
stack = stack/stack.sum(0)
fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5))
axA.stackplot(p, stack, labels=["B1 foundational","B2 intermediate","B3 advanced","B4 frontier"],
              colors=["#a9cce3","#5dade2","#2e86c1","#1b4f72"])
axA.axvspan(0.88, 1.0, color="#f9e79f", alpha=0.5)
axA.text(0.94, 0.5, "anneal", ha="center", rotation=90, fontsize=9)
axA.set_title("Difficulty-band schedule"); axA.set_xlabel("training progress")
axA.set_ylabel("share of tokens"); axA.legend(loc="upper center", ncol=2, fontsize=8)
axA.set_xlim(0,1); axA.set_ylim(0,1)
# reasoning-length ramp
med = np.clip(0.2+0.6*np.exp(-((p-0.4)**2)/0.08),0,1)
lng = np.clip((p-0.4)*1.5,0,1)
xln = np.clip((p-0.7)*2.5,0,1)
shrt= np.clip(1.1-1.6*p,0.05,1)
st2 = np.vstack([shrt,med,lng,xln]); st2 = st2/st2.sum(0)
axB.stackplot(p, st2, labels=["≤256","256–1k","1k–4k","4k–16k+"],
              colors=["#d5f5e3","#82e0aa","#28b463","#186a3b"])
axB.axvspan(0.88, 1.0, color="#f9e79f", alpha=0.5)
axB.set_title("Reasoning-length ramp (co-scheduled with context extension)")
axB.set_xlabel("training progress"); axB.set_ylabel("share of reasoning tokens")
axB.legend(loc="upper center", ncol=2, fontsize=8); axB.set_xlim(0,1); axB.set_ylim(0,1)
plt.tight_layout(); plt.savefig(os.path.join(RES, "curriculum_ramps.png"), dpi=130)
plt.close()

# --------------------------------------------------------------------------
# 6. Console summary
# --------------------------------------------------------------------------
print(f"TOTAL {TOTAL_B/1000:.1f}T | stable {STABLE_FRAC*TOTAL_B:.0f}B | anneal {ANNEAL_FRAC*TOTAL_B:.0f}B")
print(f"effective shares sum = {tot:.4f}%")
print("\nLANES flagged WISHFUL (>%.0f epochs):" % SAFE_EPOCHS)
for r in rows:
    if r["status"] == "WISHFUL":
        ep = "inf" if r["epochs"]==float("inf") else f"{r['epochs']:.1f}"
        print(f"  - {r['name']:26s} needs {ep} epochs on {r['uniq']:.0f}B conf-adj supply")
print("\nINDIC tiers flagged:")
for t in tier_rows:
    mark = "" if t["status"]=="OK" else "  <-- WISHFUL"
    print(f"  {t['name']:11s} {t['share']:.0f}%  {t['consumed']:.0f}B / {t['uniq']:.0f}B conf-adj = {t['epochs']:.2f} epochs{mark}")
print("\nwrote results/ -> lane_ledger.md, indic_tier_ledger.md, lane_headroom.png, curriculum_ramps.png")
