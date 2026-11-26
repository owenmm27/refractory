"""Chart for Test 13 (why collective bursts persist). Usage: python3 -I src/figures_bursts.py <project_root>"""
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
d = json.loads((RES / "test13b.json").read_text())
V = [("v1", "Version 1", F.BLUE), ("v2", "Version 2: + shared gain", F.ORANGE), ("v3", "Version 3: + two-day earnings", F.AQUA)]

ROWS = [("burst_slope_all", "Slope, all days"),
        ("burst_slope_ex_crisis", "Slope, outside 2008–09 and early 2020"),
        ("burst_rank_corr", "Rank correlation, all days"),
        ("burst_slope_repeat_ex_crisis", "Same stocks firing again\n(outside crises)"),
        ("burst_slope_new_ex_crisis", "New stocks joining\n(outside crises)")]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"width_ratios": [1.25, 1]})
fig.subplots_adjust(left=0.2, right=0.97, top=0.78, bottom=0.12, wspace=0.28)

# ---- A. persistence measures
for i, (k, lab) in enumerate(ROWS):
    y0 = len(ROWS) - 1 - i
    for j, (v, name, col) in enumerate(V):
        o = d["variants"][v][k]
        y = y0 + 0.22 - j * 0.22
        ax1.plot([o["p5"], o["p95"]], [y, y], color=col, lw=2.2, alpha=0.55, solid_capstyle="round")
        ax1.plot(o["mean"], y, "o", color=col, ms=7, mec=F.SURF, mew=1.2, label=name if i == 0 else None)
    ax1.plot(d["real"][k], y0, "D", color=F.INK, ms=8, mec=F.SURF, mew=1.2, label="Real data" if i == 0 else None, zorder=5)
ax1.set_yticks(range(len(ROWS)), [r[1] for r in ROWS][::-1], fontsize=9.5)
ax1.set_ylim(-0.6, len(ROWS) - 0.4)
ax1.set_xlim(0, 0.62)
ax1.set_xlabel("Day-to-day persistence of collective firing")
ax1.grid(axis="y", visible=False)
ax1.legend(loc="lower right", fontsize=8.5, handletextpad=0.4, borderaxespad=0.3)
ax1.set_title("Everyday bursting is reproduced once stocks share a volatility gain;\nthe gap on all days comes from the crisis windows",
              loc="left", fontsize=10, color=F.INK)

# ---- B. crisis intensity: worst month of collective firing per simulated history
rng = np.random.default_rng(0)
for j, (v, name, col) in enumerate(V):
    vals = np.array(d["paths"][v]["peak_month_firing"])
    yj = 2 - j
    ax2.scatter(vals, yj + rng.uniform(-0.15, 0.15, len(vals)), s=28, color=col, alpha=0.8, edgecolor=F.SURF, linewidth=0.8)
for xv, lab, dy in [(123.7, "Mar 2020", 0), (46.2, "Oct 2008", 0), (38.6, "Aug 2011", 0.28)]:
    ax2.axvline(xv, color=F.INK2, lw=1.1, ls="--" if lab != "Mar 2020" else "-")
    ax2.text(xv + 1.5, 2.55 - dy, lab, fontsize=9, color=F.INK2, va="center")
ax2.set_yticks([2, 1, 0], ["Version 1", "Version 2", "Version 3"], fontsize=9.5)
ax2.set_ylim(-0.6, 2.8)
ax2.set_xlim(0, 140)
ax2.set_xlabel("Worst month: stocks firing per day (of 237)")
ax2.grid(axis="y", visible=False)
ax2.set_title("Worst month of each simulated 26-year history vs real crises:\nonly March 2020 is at the edge (1 of 60 histories reaches it)",
              loc="left", fontsize=10, color=F.INK)

F.title(fig, "Collective bursts persist because stocks share a volatility state; no stock-to-stock excitation is needed",
        "Collective firing = stocks moving more than 4× their typical size that day. Bars: 5–95% range of 20 simulated "
        "2000–2026 histories per model version.")
fig.savefig(RES / "figures" / "fig13_bursts.png", dpi=200)
print("saved fig13", {v: max(d["paths"][v]["peak_month_firing"]) for v, _, _ in V})
