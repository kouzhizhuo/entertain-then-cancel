"""Fig 2 (fig_hero_strip): per-model thinking effect on 'some', sorted; Fig 4 (fig_major_forest): major models, some/or. Data: \\res keys."""
import re, json, matplotlib; matplotlib.use("pdf"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False})
V = {k: float(v) for k, v in re.findall(r"\\resdef\{([^}]+)\}\{(-?\d+(?:\.\d+)?)\}", open("build/results_full.tex").read())}
L = json.load(open("../exp/results/DERIVED/models_lists.json")); roster, counted, rise = L["main_tested"], set(L["main_counted"]), set(L["main_rise"])
PC = [("e1", "nat"), ("e1s", "nat"), ("e1q", "nat"), ("e1b", "nat"), ("e1p", "nat"), ("e1pangu", "nat"), ("e1r1", "nat"), ("e3", "high")]
NAMES = {"qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B", "qwen3p5-9b": "Qwen3.5-9B", "qwen3p8-27b": "Qwen3.8-27B", "mimo-7b-rl": "MiMo-7B-RL",
         "openpangu-embedded-7b-v1p1": "openPangu-7B-V1.1", "r1-0528-qwen3-8b": "R1-0528-Qwen3-8B", "gpt-oss-20b": "gpt-oss-20b", "phi4-reasoning": "Phi-4-reasoning", "olmo3-32b-think": "OLMo-3-32B-Think"}
def eff(m, w):
    for p, c in PC:
        k = f"{p}.{m}.{w}.inf.{c}.dR"
        if k + ".mean" in V: return V[k + ".mean"], V.get(k + ".lo", V[k + ".mean"]), V.get(k + ".hi", V[k + ".mean"])
O, B, G = "#D55E00", "#0072B2", "#999999"
pts = sorted([(eff(m, "a3-quant") or eff(m, "a3-conn"), m) for m in roster if eff(m, "a3-quant") or eff(m, "a3-conn")])
fig, ax = plt.subplots(figsize=(6.6, 2.2))
ax.axhline(0, color="0.5", lw=0.8)
for i, ((y, lo, hi), m) in enumerate(pts):
    c = O if m in counted else B if m in rise else G
    ax.vlines(i, lo, hi, color=c, lw=0.9, alpha=0.7); ax.plot(i, y, "o", color=c, ms=3.6, zorder=3)
    if m in NAMES:
        up = (i % 2 == 0); ty = (hi + 9) if up else (lo - 9)
        ax.annotate(NAMES[m], (i, hi if up else lo), (i, ty), fontsize=5.6, ha="center", va="bottom" if up else "top", arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"))
ax.text(0.01, 0.97, f"{len(counted)} of {len(roster)} models more literal", transform=ax.transAxes, color=O, fontweight="bold", va="top")
ax.text(0.01, 0.86, f"{len(rise)} more pragmatic", transform=ax.transAxes, color=B, fontweight="bold", va="top")
ax.set_xticks([]); ax.set_xlabel("open reasoning models, sorted by effect"); ax.set_ylabel("thinking effect on 'some' (pp)")
ax.set_xlim(-1, len(pts)); fig.tight_layout(pad=0.3); fig.savefig("figures/fig_hero_strip.pdf")
SMALL = {"qwen3-8b", "qwen3p8-27b", "mimo-7b-rl", "r1-0528-qwen3-8b", "gpt-oss-20b"}
pts = sorted([(eff(m, "a3-quant") or eff(m, "a3-conn"), m) for m in roster if eff(m, "a3-quant") or eff(m, "a3-conn")])
fig, ax = plt.subplots(figsize=(3.3, 2.1))
ax.axhline(0, color="0.5", lw=0.8)
for i, ((y, lo, hi), m) in enumerate(pts):
    c = O if m in counted else B if m in rise else G
    ax.vlines(i, lo, hi, color=c, lw=0.9, alpha=0.7); ax.plot(i, y, "o", color=c, ms=2.6, zorder=3)
    if m in SMALL:
        up = (i % 2 == 0); ty = (hi + 9) if up else (lo - 9)
        ax.annotate(NAMES[m], (i, hi if up else lo), (i, ty), fontsize=5.2, ha="center", va="bottom" if up else "top", arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"))
ax.text(0.01, 0.97, f"{len(counted)} of {len(roster)} models more literal", transform=ax.transAxes, color=O, fontweight="bold", va="top")
ax.text(0.01, 0.84, f"{len(rise)} more pragmatic", transform=ax.transAxes, color=B, fontweight="bold", va="top")
ax.set_xticks([]); ax.set_xlabel("open reasoning models, sorted by effect"); ax.set_ylabel("thinking effect on 'some' (pp)")
ax.set_xlim(-1, len(pts)); fig.tight_layout(pad=0.3); ax.text(0.99, 0.03, "each dot = one model; below 0 = more literal with thinking", transform=ax.transAxes, ha="right", fontsize=5.3, style="italic", color="0.35"); fig.savefig("figures/fig_hero_strip_small.pdf")
fig, ax = plt.subplots(figsize=(3.15, 2.5)); ax.axvline(0, color="0.5", lw=0.8); ax.axvspan(-200, 0, color="0.94", zorder=0)
rows = [m for m in NAMES if eff(m, "a3-quant")]; lims = []
for r, m in enumerate(rows):
    y = len(rows) - 1 - r
    for w, dy, fc in (("a3-quant", 0.15, O), ("a3-conn", -0.15, "white")):
        e = eff(m, w)
        if not e: continue
        ax.hlines(y + dy, e[1], e[2], color=O, lw=0.9); ax.plot(e[0], y + dy, "o", mfc=fc, mec=O, ms=3.8, zorder=3); lims += [e[1], e[2]]
ax.set_yticks(range(len(rows))); ax.set_yticklabels([NAMES[m] for m in rows][::-1]); ax.set_xlim(min(lims) - 5, max(5, max(lims) + 5))
ax.text(0.98, 1.02, "● some   ○ or", transform=ax.transAxes, ha="right", color=O, fontsize=6.5)
ax.text(0.02, 0.01, "more literal", transform=ax.transAxes, style="italic", fontsize=6, color="0.4")
ax.set_xlabel("thinking effect (pp)"); fig.tight_layout(pad=0.3); fig.savefig("figures/fig_major_forest.pdf"); print("wrote")
