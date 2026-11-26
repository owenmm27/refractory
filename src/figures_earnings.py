"""Chart for Test 8 (earnings calendar). Usage: python3 -I src/figures_earnings.py <project_root>"""
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
ev = json.loads((RES / "test8_event_study.json").read_text())
r8 = json.loads((RES / "test8.json").read_text())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1.15, 1]})
fig.subplots_adjust(left=0.2, right=0.98, top=0.76, bottom=0.17, wspace=0.62)

groups = [("same_sector_not_yet", "Same sector,\nnot yet reported"), ("same_sector_already", "Same sector,\nalready reported"),
          ("other_sector", "Other sectors")]
for gi, (key, lab) in enumerate(groups):
    for li, (lag, c, name) in enumerate([(0, F.BLUE, "Same day"), (1, F.ORANGE, "Next day")]):
        v = ev[f"lag{lag}/{key}"]
        se = abs(v["diff_pct"] / v["t"]) if v["t"] else 0
        y = gi + (li - 0.5) * 0.32
        ax1.plot([v["diff_pct"] - 1.96 * se, v["diff_pct"] + 1.96 * se], [y, y], color=c, lw=2.5, solid_capstyle="round", alpha=0.45)
        ax1.plot([v["diff_pct"]], [y], "o", ms=7, color=c, mec=F.SURF, mew=1.2, label=name if gi == 0 else None)
ax1.axvline(0, color=F.BASE, lw=1)
ax1.set_yticks(range(len(groups)), [g[1] for g in groups])
ax1.invert_yaxis()
ax1.set_xlabel("Extra peer volatility after a big vs a small\nearnings surprise (%, with 95% interval)")
ax1.grid(axis="y", visible=False)
ax1.legend(loc="lower right", fontsize=9)
ax1.set_title("Earnings news reaches peers the same day", loc="left", fontsize=10.5, color=F.INK)

d = r8["D_firing_persistence"]
nul = d["nulls"]["shocks_from_non_earnings_days"]
rows = [("All days", d["real_all_days"], nul["lag1_all_days"]),
        ("Earnings days removed", d["real_earnings_windows_removed"], nul["lag1_earnings_windows_removed"])]
for y, (lab, real, (m, lo, hi)) in enumerate(rows):
    ax2.plot([lo, hi], [y, y], color=F.BLUE, lw=3, solid_capstyle="round")
    ax2.plot([m], [y], "o", ms=7, color=F.BLUE, mec=F.SURF, mew=1.2, label="Null: real volatility, random shocks" if y == 0 else None)
    ax2.plot([real], [y], "D", ms=7, color=F.INK, label="Real stocks" if y == 0 else None)
ax2.set_yticks(range(len(rows)), [x[0] for x in rows])
ax2.set_ylim(1.7, -0.7)
ax2.set_xlim(0.40, 0.56)
ax2.set_xlabel("Day-to-day persistence of collective firing\n(lag-1 slope; 95% range over 20 runs)")
ax2.grid(axis="y", visible=False)
ax2.legend(loc="lower center", fontsize=9, bbox_to_anchor=(0.5, -0.02))
ax2.set_title("Removing earnings days leaves the gap intact", loc="left", fontsize=10.5, color=F.INK)

F.title(fig, "The earnings calendar isn't the hidden structure",
        "237 S&P 500 stocks, 2005–2026, earnings dates from SEC 8-K Item 2.02 filings (20,713 releases).")
fig.savefig(RES / "figures" / "fig8_earnings.png", dpi=200)
print("saved fig8")
