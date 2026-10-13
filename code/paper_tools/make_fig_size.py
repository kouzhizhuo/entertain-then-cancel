
"""Fig: thinking effect on some/or vs model size, by family; plus forest plot of major models."""
import re
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rcParams["font.family"] = "DejaVu Sans"

V = {k: float(v.replace("$", "").replace("\\text{-}", "-").replace("\u2212", "-"))
     for k, v in re.findall(r"\\resdef\{([^}]+)\}\{(-?\d+(?:\.\d+)?)\}",
                             open("build/results_full.tex").read())}

FAM = {
    "Qwen3": [(0.6, "e1s.qwen3-0p6b"), (1.7, "e1s.qwen3-1p7b"), (4, "e1.qwen3-4b"),
              (8, "e1.qwen3-8b"), (14, "e1.qwen3-14b"), (32, "e1s.qwen3-32b")],
    "Qwen3.5": [(0.8, "e1q.qwen3p5-0p8b"), (2, "e1q.qwen3p5-2b"), (4, "e1q.qwen3p5-4b"),
                (9, "e1q.qwen3p5-9b"), (35, "e1q.qwen3p5-35b-a3b")],
    "R1-Distill-Qwen": [(7, "e1s.r1distill-7b"), (14, "e1s.r1distill-14b")],
}
COL = {"Qwen3": "#0072B2", "Qwen3.5": "#009E73", "R1-Distill-Qwen": "#CC79A7"}

fig = plt.figure(figsize=(6.6, 2.5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1.25], figure=fig)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1], sharey=ax_a)
ax_c = fig.add_subplot(gs[0, 2])

panels = [(ax_a, "a3-quant", "some \u2192 not all"), (ax_b, "a3-conn", "or \u2192 not both")]

all_y = []
for ax, w, title in panels:
    ax.axhline(0, color="0.5", lw=0.9, zorder=1)
    for f, pts in FAM.items():
        xs, ys, es = [], [], []
        for x, k in pts:
            m = V.get(f"{k}.{w}.inf.nat.dR.mean")
            if m is None:
                continue
            lo, hi = V.get(f"{k}.{w}.inf.nat.dR.lo", m), V.get(f"{k}.{w}.inf.nat.dR.hi", m)
            xs.append(x); ys.append(m); es.append((m - lo, hi - m))
        if not xs:
            continue
        all_y.extend(ys)
        ax.errorbar(xs, ys, yerr=list(zip(*es)), fmt="o-", color=COL[f], ms=3.2,
                     lw=1.1, capsize=1.5, elinewidth=0.8, zorder=3)
        ax.annotate(f, (xs[-1], ys[-1]), fontsize=6, fontweight="bold", color=COL[f],
                     xytext=(4, 0), textcoords="offset points", va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xticks([1, 4, 16])
    ax.set_xticklabels(["1", "4", "16"])
    ax.set_title(title, fontsize=7)
    ax.set_xlabel("parameters (B)", fontsize=7)
    ax.tick_params(labelsize=6.5)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)

if all_y:
    ymin, ymax = min(all_y), max(all_y)
    pad = max(2.0, 0.15 * (ymax - ymin))
    ylo, yhi = ymin - pad, ymax + pad
else:
    ylo, yhi = -5, 5
ax_a.set_ylim(ylo, yhi)
for ax in (ax_a, ax_b):
    ax.axhspan(ylo, 0, color="0.92", zorder=0)

ax_a.text(0.03, 0.06, "more literal", fontsize=5.5, style="italic", color="0.4",
           transform=ax_a.transAxes, ha="left", va="bottom")
ax_a.text(0.03, 0.94, "more pragmatic", fontsize=5.5, color="0.3",
           transform=ax_a.transAxes, ha="left", va="top")

ax_a.set_ylabel("thinking effect (pp)", fontsize=7)
ax_b.tick_params(labelleft=False)

# --- Panel (c): forest plot of major models ---
rows = [
    ("Qwen3-8B", "e1.qwen3-8b", "nat"),
    ("Qwen3.5-9B", "e1q.qwen3p5-9b", "nat"),
    ("Qwen3.8-27B", "e1q.qwen3p8-27b", "nat"),
    ("MiMo-7B-RL", "e1b.mimo-7b-rl", "nat"),
    ("openPangu-7B-V1.1", "e1pangu.openpangu-embedded-7b-v1p1", "nat"),
    ("DeepSeek-R1-0528-Qwen3-8B", "e1r1.r1-0528-qwen3-8b", "nat"),
    ("gpt-oss-20b", "e3.gpt-oss-20b", "high"),
]
COL_C = "#D55E00"
ax_c.axvline(0, color="0.5", lw=0.9, zorder=1)

ylabels, yticks = [], []
n = len(rows)
for i, (name, k, cond) in enumerate(rows):
    y = n - 1 - i
    yticks.append(y)
    ylabels.append(name)
    for word, dy, filled in [("a3-quant", 0.14, True), ("a3-conn", -0.14, False)]:
        mean = V.get(f"{k}.{word}.inf.{cond}.dR.mean")
        if mean is None:
            continue
        lo = V.get(f"{k}.{word}.inf.{cond}.dR.lo", mean)
        hi = V.get(f"{k}.{word}.inf.{cond}.dR.hi", mean)
        ax_c.errorbar(mean, y + dy, xerr=[[mean - lo], [hi - mean]], fmt="o",
                       mfc=COL_C if filled else "none", mec=COL_C, ecolor=COL_C,
                       ms=4, mew=1.0, lw=1.0, capsize=2, zorder=3)

ax_c.set_yticks(yticks)
ax_c.set_yticklabels(ylabels, fontsize=6.5)
ax_c.set_ylim(-0.7, n - 0.3)
ax_c.set_xlabel("thinking effect (pp)", fontsize=7)
ax_c.set_title("Major models", fontsize=7)
ax_c.tick_params(labelsize=6.5)
for s_ in ("top", "right"):
    ax_c.spines[s_].set_visible(False)
ax_c.text(0.5, 1.14, "\u25cf some    \u25cb or", fontsize=6, color=COL_C,
           transform=ax_c.transAxes, ha="center", va="bottom")

for ax, lab in zip((ax_a, ax_b, ax_c), ("(a)", "(b)", "(c)")):
    ax.text(-0.12, 1.12, lab, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")

fig.tight_layout(pad=0.3)
fig.savefig("figures/fig_size_family.pdf")
print("wrote")
