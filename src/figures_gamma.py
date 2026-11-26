"""Chart for Test 11 (dealer gamma vs the refractory period). Usage: python3 -I src/figures_gamma.py <project_root>"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import figures as F

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
RES = ROOT / "results"
d = json.loads((RES / "test11.json").read_text())
ml = d["model_light_thr1"]
pl = d["stock_placebo_thr1"]
T = ["low", "mid", "high"]
TL = ["Low", "Mid", "High"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
fig.subplots_adjust(left=0.07, right=0.88, top=0.76, bottom=0.15, wspace=0.32)

# ---- A. refractory ratio by gamma tercile and by VIX tercile (index only)
ax1.axhline(1, color=F.BASE, lw=1)
for x0, key in [(0, "ratio_big_over_calm_by_gamma_tercile"), (3.8, "ratio_big_over_calm_by_VIX_tercile")]:
    v = ml[key]
    xs = x0 + np.arange(3)
    y = np.array([v["ratio"][t] for t in T])
    lo = np.array([v["ci95"][t][0] for t in T])
    hi = np.array([v["ci95"][t][1] for t in T])
    ax1.vlines(xs, lo, hi, color=F.BLUE, lw=1.6, alpha=0.45)
    ax1.plot(xs, y, color=F.BLUE, lw=1.6, marker="o", ms=8, mec=F.SURF, mew=1.5, zorder=5)
ax1.set_xticks(list(range(3)) + list(3.8 + np.arange(3)), TL + TL)
ax1.text(1, -0.15, "Dealer gamma (GEX) at the close", transform=ax1.get_xaxis_transform(), ha="center", va="top",
         fontsize=9.5, color=F.INK2)
ax1.text(4.8, -0.15, "VIX at the close", transform=ax1.get_xaxis_transform(), ha="center", va="top", fontsize=9.5,
         color=F.INK2)
ax1.set_xlim(-0.5, 6.3)
ax1.set_ylim(0.45, 1.55)
ax1.set_ylabel("Next-day variance after a big day\n÷ after a calm day")
ax1.text(6.25, 1.02, "1.0 = no refractory effect", color=F.MUTED, fontsize=9, ha="right", va="bottom")
ax1.set_title("S&P 500: the dip doesn't deepen with dealer gamma (if anything the reverse);\nit is deepest at low gamma and at high VIX, which go together (r = −0.49)",
              loc="left", fontsize=10, color=F.INK)

# ---- B. calm-day variance vs model by gamma tercile: index vs stocks' own moves
xs = np.arange(3)
ax2.axhline(1, color=F.BASE, lw=1)
for src, col, name in [(pl["index_after_calm_by_gamma"], F.BLUE, "S&P 500 index"),
                       (pl["idio_after_calm_by_gamma"], F.ORANGE, "Stocks' own moves\n(market removed)")]:
    y = np.array([src["mean"][t] for t in T])
    lo = np.array([src["ci95"][t][0] for t in T])
    hi = np.array([src["ci95"][t][1] for t in T])
    ax2.fill_between(xs, lo, hi, color=col, alpha=0.13, lw=0)
    ax2.plot(xs, y, color=col, lw=2, marker="o", ms=6, mec=F.SURF, mew=1.2, label=name.replace("\n", " "))
    ax2.annotate(name, (xs[-1], y[-1]), xytext=(9, 0 if col == F.ORANGE else -4), textcoords="offset points",
                 va="center", fontsize=9.5, color=F.INK2, annotation_clip=False)
ax2.set_xticks(xs, TL)
ax2.set_xlim(-0.2, 2.2)
ax2.set_ylim(0.7, 1.3)
ax2.set_xlabel("Dealer gamma (GEX) at the close")
ax2.set_ylabel("Today's variance ÷ model's forecast\n(after a calm day)")
ax2.text(0.03, 0.993, "1.0 = as expected", color=F.MUTED, fontsize=9, va="top")
ax2.set_title("High gamma marks calmer days for stocks' own moves too,\nwhich index-option hedging can't cause",
              loc="left", fontsize=10, color=F.INK)
ax2.legend(loc="lower left", fontsize=8.5, handletextpad=0.4, borderaxespad=0.3)

F.title(fig, "Dealer hedging doesn't explain the market's refractory period",
        "S&P 500, Oct 2011–Nov 2026, split into thirds by gamma or VIX. Big day = more than 1 SD above the model; "
        "bars and bands are 95% block-bootstrap intervals.")
fig.savefig(RES / "figures" / "fig11_gamma.png", dpi=200)
print("saved fig11")
