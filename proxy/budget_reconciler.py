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
    ("Indic (all tiers)",         26.0,  24.0,  1232.0, 0.74, "sum of the four tiers below; blended discount"),
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
    ("Verified",     28.0,   90.0, 0.70, "Sangraha Indic-verified 51.5B (card's 64.3B MINUS 12.76B English) + IndicCorp v2 ~21B + Indic Wikipedia/curated ~18B; cross-source dup"),
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

# --------------------------------------------------------------------------
# 7. Per-language Indic split (capstone deepening).
#    The Indic lane (INDIC_CONSUMED tokens) is split across the 22 scheduled
#    languages, not just the 4 tiers. Allocation rule:
#      q_i   = real_i / sum(real)                 natural proportion
#              real_i = Sangraha verified_i + unverified_i  (genuinely-sourced;
#              synthetic is generated-to-order so it does NOT drive proportions)
#      p_i  ~ q_i ** ALPHA                         temperature sampling (ALPHA=0.3,
#              i.e. T~3.3; flattens the Hindi-heavy head toward low-resource langs,
#              as in XLM-R / mBERT / NLLB)
#      then a per-language FLOOR guarantees coverage, and we renormalise.
#    Each language is then checked against the 4-epoch ceiling on its VERIFIED
#    supply: how much of the 28% verified-tier target can it actually back from
#    real verified tokens at <=4 epochs? The shortfall is necessarily synthetic/
#    translated. This surfaces, per language, exactly how synthetic-dependent it is.
#    Sangraha per-language token counts (millions), Indic only (English excluded).
#    Fertility = IndicSuperTokenizer tokens/word (arXiv 2511.03237 Table 3).
ALPHA = 0.30
FLOOR_SHARE = 0.005          # >=0.5% of the Indic lane per scheduled language
CONF_V = 0.70                # verified-tier supply-confidence (matches the tier table)
TIER_V = 0.28                # verified tier target share of the Indic lane

#               name          verified_M  unverified_M  fertility
INDIC_LANG = [
    ("Hindi",       12617.3, 12348.3, 1.23),
    ("Bengali",     10604.4,  5608.8, 1.74),
    ("Tamil",        3985.1,  1515.9, 2.12),
    ("Urdu",         3658.1,  1328.2, 1.44),
    ("Telugu",       3706.8,   647.4, 1.88),
    ("Gujarati",     3647.9,   597.0, 1.77),
    ("Marathi",      2827.0,   652.1, 1.63),
    ("Malayalam",    2730.8,   547.8, 2.30),
    ("Nepali",       1822.5,   485.5, 1.62),
    ("Kannada",      1778.3,   388.8, 2.19),
    ("Sanskrit",     1329.0,     9.8, 2.59),
    ("Punjabi",      1075.3,   136.9, 1.39),
    ("Odia",         1177.1,    23.7, 1.65),
    ("Assamese",      292.1,    17.5, 1.85),
    # Sindhi 258.2 + Maithili 14.6 + Konkani 10.1 + Manipuri 7.4 + Bodo 1.5
    #  + Kashmiri 0.5 + Santali 0.3 + Dogri 0.06
    ("Rest (8 langs)", 292.66,   0.0, 2.50),
]

real = [v + u for (_, v, u, _f) in INDIC_LANG]
sum_real = sum(real)
# temperature-smoothed shares
tw = [r ** ALPHA for r in real]
p_temp = [wi / sum(tw) for wi in tw]
# apply floor then renormalise
p_floored = [max(pi, FLOOR_SHARE) for pi in p_temp]
p = [pi / sum(p_floored) for pi in p_floored]

lang_rows = []
for (name, ver_M, unv_M, fert), q_real, share in zip(
        INDIC_LANG, [r / sum_real for r in real], p):
    tokens_b = share * INDIC_CONSUMED                 # B tokens allocated
    usable_ver_b = (ver_M / 1000.0) * CONF_V          # B usable verified tokens
    # verified epochs if we insisted on the full 28% verified tier for this language
    vep = (TIER_V * tokens_b) / usable_ver_b if usable_ver_b > 0 else float("inf")
    # the most verified SHARE this language can back at <=4 epochs
    max_ver_share = min(TIER_V, (SAFE_EPOCHS * usable_ver_b) / tokens_b) if tokens_b > 0 else 0.0
    synth_trans_floor = TIER_V - max_ver_share        # gap that must be synth/translated
    verdict = "verified-OK" if max_ver_share >= TIER_V - 1e-9 else "synthetic-lean"
    lang_rows.append(dict(name=name, ver_M=ver_M, unv_M=unv_M, fert=fert,
                          real_b=(ver_M + unv_M) / 1000.0, q_real=q_real,
                          share=share, tokens_b=tokens_b, usable_ver_b=usable_ver_b,
                          vep=vep, max_ver_share=max_ver_share,
                          synth_trans_floor=synth_trans_floor, verdict=verdict))

# content-weighted share (fertility-adjusted): content_i ~ tokens_i / fertility_i
content_raw = [r["tokens_b"] / r["fert"] for r in lang_rows]
sum_content = sum(content_raw)
for r, c in zip(lang_rows, content_raw):
    r["content_share"] = c / sum_content

# ---- write per-language ledger ----
pl = ["# Per-language Indic ledger (auto-generated by budget_reconciler.py)\n",
      f"Indic lane = **{INDIC_CONSUMED:.0f}B tokens**, split across 22 scheduled "
      f"languages. Rule: temperature sampling (alpha={ALPHA}) on real "
      f"(verified+unverified) Sangraha supply, a {FLOOR_SHARE*100:.1f}% per-language "
      f"floor, then each language checked against the 4-epoch ceiling on its VERIFIED "
      f"supply (tier target {TIER_V*100:.0f}%). Sangraha English (12.76B) excluded.\n",
      "| Language | Real V+U (B) | Verified (B) | Fertility | Target share | "
      "Target tokens (B) | Content share | Verified epochs @28% | Max verified share @<=4ep | Verdict |",
      "|---|--:|--:|--:|--:|--:|--:|--:|--:|:--|"]
for r in lang_rows:
    vepx = "inf" if r["vep"] == float("inf") else f"{r['vep']:.1f}"
    flag = "OK" if r["verdict"] == "verified-OK" else "**synthetic-lean**"
    pl.append(f"| {r['name']} | {r['real_b']:.1f} | {r['ver_M']/1000:.2f} | "
              f"{r['fert']:.2f} | {r['share']*100:.1f}% | {r['tokens_b']:.0f} | "
              f"{r['content_share']*100:.1f}% | {vepx} | {r['max_ver_share']*100:.0f}% | {flag} |")
tot_tokens = sum(r["tokens_b"] for r in lang_rows)
# token-weighted verified share actually backable from real verified at <=4 epochs
agg_ver_backable = sum(r["max_ver_share"] * r["tokens_b"] for r in lang_rows) / tot_tokens
pl.append(f"| **Total** | {sum_real/1000:.1f} | {sum(r['ver_M'] for r in lang_rows)/1000:.1f} "
          f"| - | {sum(r['share'] for r in lang_rows)*100:.0f}% | {tot_tokens:.0f} | 100% | - | "
          f"**{agg_ver_backable*100:.0f}%** | - |")
n_lean = sum(1 for r in lang_rows if r["verdict"] != "verified-OK")
pl += ["",
       f"**The headline finding.** At a flat 28% verified tier, *no* language can be fully "
       f"verified-backed at <=4 epochs — the per-language ceiling runs from ~{lang_rows[0]['max_ver_share']*100:.0f}% "
       f"(Hindi) down to ~{lang_rows[-1]['max_ver_share']*100:.0f}% (low-resource). Token-weighted, only "
       f"**~{agg_ver_backable*100:.0f}% of the Indic lane is real-verified-backable** today; the single "
       f"'28% verified' in the aggregate tier table is optimistic. The actionable fix is a "
       f"**per-language verified cap** (the 'Max verified share' column): each language's verified "
       f"tier is set to what its real supply can sustain, and the remainder is explicitly "
       f"translated + synthetic. Low-resource languages are therefore *structurally* "
       f"synthetic-first; only data-rich Hindi/Bengali approach a verified-backed tier.",
       "",
       f"**And the aggregate was optimistic twice over.** The §2 pooled view says 28% verified "
       f"needs 6.87 epochs, so only ~{SAFE_EPOCHS/6.87*28:.0f}% is backable at the ceiling. But that "
       f"treats verified as one pool — it implicitly lets Hindi's surplus back Odia's demand, which "
       f"is not real. Once each language's verified demand must be met by *that language's* verified "
       f"supply, the backable share drops further to **~{agg_ver_backable*100:.0f}%**. Per-language "
       f"matching is the stricter, honest number. Growing verified supply via cleaning (README §9) "
       f"lifts every language's ceiling together.",
       "",
       f"*Allocation knob:* alpha={ALPHA} (temperature sampling) flattens Hindi from its "
       f"~{lang_rows[0]['q_real']*100:.0f}% natural share to {lang_rows[0]['share']*100:.0f}% for "
       f"low-resource coverage; a production India-first model may fold in speaker population. "
       f"alpha is itself a hypothesis a proxy arm can test.",
       "",
       "*Supply: Sangraha per-language (AI4Bharat). Fertility: IndicSuperTokenizer Table 3 "
       "(arXiv 2511.03237). 'Rest' = Sindhi, Maithili, Konkani, Manipuri, Bodo, Kashmiri, "
       "Santali, Dogri. Sangraha English (12.76B) excluded from verified.*"]
w("per_language_ledger.md", "\n".join(pl) + "\n")

# ---- chart: allocated tokens vs verified-backable ceiling, per language ----
fig, ax = plt.subplots(figsize=(10, 6.5))
names = [r["name"] for r in lang_rows]
yy = range(len(names))
demand = [r["tokens_b"] for r in lang_rows]
ver_ceiling = [SAFE_EPOCHS * r["usable_ver_b"] for r in lang_rows]   # max verified tokens @4ep
cols = ["#c0392b" if r["verdict"] != "verified-OK" else "#2e86c1" for r in lang_rows]
ax.barh([i+0.2 for i in yy], demand, height=0.4, color=cols, label="Allocated tokens")
ax.barh([i-0.2 for i in yy], ver_ceiling, height=0.4, color="#aab7b8",
        label="Verified-backable ceiling (4 x usable verified)")
ax.set_yticks(list(yy)); ax.set_yticklabels(names)
ax.set_xscale("log"); ax.set_xlabel("Billions of tokens (log scale)")
ax.set_title("Per-language Indic: allocation vs. verified-backable ceiling\n"
             "(red = verified-starved -> must be synthetic/translated)")
ax.invert_yaxis(); ax.legend(loc="lower right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(RES, "per_language_indic.png"), dpi=130)
plt.close()

# ---- console summary ----
print("\nPER-LANGUAGE Indic split (alpha=%.2f, floor=%.1f%%):" % (ALPHA, FLOOR_SHARE*100))
for r in lang_rows:
    vepx = "inf" if r["vep"] == float("inf") else f"{r['vep']:4.1f}"
    print(f"  {r['name']:14s} share {r['share']*100:4.1f}%  tokens {r['tokens_b']:5.0f}B  "
          f"vepochs@28% {vepx}  maxVer {r['max_ver_share']*100:2.0f}%  {r['verdict']}")
print(f"  -> {n_lean}/{len(lang_rows)} languages verified-starved")
print("wrote results/ -> per_language_ledger.md, per_language_indic.png")
