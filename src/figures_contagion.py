"""Chart for Test 7 (contagion). Usage: python3 -I src/figures_contagion.py <project_root>"""
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
r = json.loads((RES / "test7.json").read_text())
fu = json.loads((RES / "test7_followup.json").read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.8), gridspec_kw={"width_ratios": [1.05, 1]})
fig.subplots_adjust(left=0.17, right=0.98, top=0.78, bottom=0.17, wspace=0.75)

# left: sector leadership in volatility surprises, both halves
sec = r["real_sectors"]["magnitude"]["by_sector_[half1,half2,full]"]
items = sorted(sec.items(), key=lambda kv: kv[1][2])
for y, (name, (h1, h2, _)) in enumerate(items):
    ax1.plot([h1 * 1000, h2 * 1000], [y, y], color=F.GRID, lw=2.5, zorder=1)
    ax1.plot([h1 * 1000], [y], "o", ms=7, color=F.BLUE, mec=F.SURF, mew=1.2, zorder=3, label="1990–2008" if y == 0 else None)
    ax1.plot([h2 * 1000], [y], "o", ms=7, color=F.ORANGE, mec=F.SURF, mew=1.2, zorder=3, label="2008–2026" if y == 0 else None)
ax1.axvline(0, color=F.BASE, lw=1)
ax1.set_yticks(range(len(items)), [k for k, _ in items])
ax1.set_xlabel("Net leadership in volatility surprises (×1000)\n← follows other sectors      leads other sectors →")
ax1.grid(axis="y", visible=False)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_title("Sectors: the same ones lead in both halves (r = %.2f)" % r["real_sectors"]["magnitude"]["halves_corr"],
              loc="left", fontsize=10.5, color=F.INK)

# right: how stable is stock-level leadership? calibrated against synthetic panels
rows = [("Synthetic: common factors only", r["synthetic"]["factor_only"]["magnitude"]["stability"], None),
        ("Synthetic: planted contagion", r["synthetic"]["planted_contagion"]["magnitude"]["stability"], None),
        ("Real: volatility surprises", r["real"]["magnitude"]["stability"], fu["magnitude"]["within_sector_leadership_halves_corr"]),
        ("Real: return direction", r["real"]["direction"]["stability"], fu["direction"]["within_sector_leadership_halves_corr"])]
for y, (lab, s, within) in enumerate(rows):
    band = 2 * s["leadership_null_sd"]
    ax2.plot([-band, band], [y, y], color=F.GRID, lw=7, solid_capstyle="butt", zorder=1)
    ax2.plot([s["leadership_corr"]], [y], "o", ms=8, color=F.BLUE, mec=F.SURF, mew=1.2, zorder=3,
             label="All stocks" if y == 0 else None)
    if within is not None:
        ax2.plot([within], [y], "D", ms=6.5, color=F.ORANGE, mec=F.SURF, mew=1.2, zorder=3,
                 label="Within sectors only" if y == 2 else None)
ax2.axvline(0, color=F.BASE, lw=1)
ax2.set_yticks(range(len(rows)), [x[0] for x in rows])
ax2.invert_yaxis()
ax2.set_xlim(-0.2, 0.52)
ax2.set_xlabel("Does a stock's leadership repeat across halves?\n(correlation; gray = range expected by chance)")
ax2.grid(axis="y", visible=False)
ax2.legend(loc="upper right", fontsize=9)
ax2.set_title("Stocks: does each one's leadership repeat?", loc="left", fontsize=10.5, color=F.INK)

F.title(fig, "Volatility doesn't spread stock to stock; return information does",
        "237 S&P 500 stocks, 1990–2026. One-way (lagged, asymmetric) links in surprises, measured separately in each half.")
fig.savefig(RES / "figures" / "fig7_contagion.png", dpi=200)
print("saved fig7")
