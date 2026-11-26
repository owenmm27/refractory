"""README chart: the refractory day in one picture (model-free check from Test 15b).

Reads only results/test15b.json, so it runs without the downloaded data.
Usage: python3 -I src/figures_hero.py <project_root>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
RES = ROOT / "results"
J = json.loads((RES / "test15b.json").read_text())

# same palette as figures.py
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"], "font.size": 12.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "legend.frameon": False, "legend.labelcolor": INK2,
})

ERAS = [("pre-2000", "Before 2000"), ("2000-2012", "2000–2012"), ("2013-2026", "2013–2026")]
pooled = J["intl_pooled"]
idx = [pooled[e]["calmer_net_move"] for e, _ in ERAS]
null_lo = min(pooled[e]["null_p05"] for e, _ in ERAS) * 100
null_hi = max(pooled[e]["null_p95"] for e, _ in ERAS) * 100
stk = [J["stocks"]["own_2005-2012"]["calmer_net_move"], J["stocks"]["own_2013-2026"]["calmer_net_move"]]
n_mkts = [pooled[e]["n_markets"] for e, _ in ERAS]

fig = plt.figure(figsize=(7.2, 5.0))
ax = fig.add_axes([0.085, 0.13, 0.8, 0.56])

ax.axhspan(null_lo, null_hi, color=GRID, alpha=0.8, lw=0, zorder=0)
ax.text(-0.42, null_lo + 0.3, "no-effect range\n(simulated)", fontsize=10.5, color=MUTED, va="bottom", linespacing=1.2)
for y in (45, 55, 60):
    ax.axhline(y, color=GRID, lw=0.6, zorder=0)


def series(xs, pts, color, ms):
    ys = [p["est"] * 100 for p in pts]
    ax.plot(xs, ys, color=color, lw=2, zorder=3, solid_capstyle="round")
    for x, p in zip(xs, pts):
        ax.plot([x, x], [p["lo"] * 100, p["hi"] * 100], color=color, lw=2.2, alpha=0.4, zorder=2,
                solid_capstyle="round")
    ax.plot(xs, ys, "o", ms=ms, color=color, mec=SURF, mew=2, zorder=4)
    return ys


yi = series([0, 1, 2], idx, ORANGE, 11)
ys = series([1, 2], stk, BLUE, 9)

ax.annotate(f"{yi[1]:.1f}%", (1, idx[1]["hi"] * 100), xytext=(0, 6), textcoords="offset points", ha="center",
            va="bottom", fontsize=13, fontweight="bold", color=INK)
for x in (0, 2):
    ax.annotate(f"{yi[x]:.1f}%", (x, yi[x]), xytext=(-13, 0), textcoords="offset points", ha="right",
                va="center", fontsize=11, color=INK2)
ax.annotate(f"{ys[0]:.1f}%", (1, stk[0]["lo"] * 100), xytext=(0, -5), textcoords="offset points", ha="center",
            va="top", fontsize=11, color=INK2)
ax.annotate("Stock indices", (2, yi[2]), xytext=(14, 0), textcoords="offset points", va="center", fontsize=12,
            color=INK2)
ax.annotate("Single stocks", (2, ys[1]), xytext=(14, 0), textcoords="offset points", va="center", fontsize=12,
            color=INK2)

ax.set_xlim(-0.45, 2.75)
ax.set_ylim(44, 61)
ax.set_xticks([0, 1, 2], [lab for _, lab in ERAS], fontsize=12.5)
ax.set_yticks([45, 50, 55, 60], ["45%", "50%", "55%", "60%"])
ax.tick_params(axis="both", length=0, pad=6)

leg = [plt.Line2D([], [], marker="o", ls="-", lw=2, ms=9, color=ORANGE, mec=SURF,
                  label=f"Stock indices worldwide ({min(n_mkts)}–{max(n_mkts)} markets pooled)"),
       plt.Line2D([], [], marker="o", ls="-", lw=2, ms=8, color=BLUE, mec=SURF,
                  label="237 single US stocks (from 2005)")]
fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.06, 0.785), ncol=1, fontsize=11, handletextpad=0.5,
           labelspacing=0.3)

fig.text(0.06, 0.965, "Markets took a ‘refractory’ day after big shocks in 2000–2012", fontsize=16, fontweight="bold",
         color=INK, va="top")
fig.text(0.06, 0.902, "How often the day after a 2-SD+ move is calmer than the day after that.\n"
                      "Without a refractory effect, volatility just decays and this stays near 50%.",
         fontsize=11.5, color=INK2, va="top", linespacing=1.35)
fig.text(0.06, 0.025, "Vertical bars: 95% intervals, resampled by month. Daily closes, Yahoo Finance. Test 15 in FINDINGS.md.",
         fontsize=9.5, color=MUTED)

out = RES / "figures" / "hero_refractory.png"
fig.savefig(out, dpi=200)
print("saved", out.name)
