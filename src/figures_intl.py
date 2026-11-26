"""Chart for Test 14 (the market unit outside the US). Usage: python3 -I src/figures_intl.py <project_root>"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import figures as F

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
RES = ROOT / "results"
d = pd.read_csv(RES / "test14_intl.csv")
ERAS = [("pre-2000", "Before 2000", F.BLUE), ("2000-2012", "2000–2012", F.ORANGE), ("2013-2026", "2013–2026", F.AQUA)]
DEV = ["GSPC", "FTSE", "DAX", "CAC", "SMI", "AEX", "N225", "HSI", "ASX200", "TSX"]
EM = ["KOSPI", "TAIEX", "SENSEX", "BOVESPA", "IPC"]
w_dip = d.pivot(index="code", columns="era", values="next_day_after_3sd") * 100
w_vr = d.pivot(index="code", columns="era", values="VR21")
order = ["GSPC"] + sorted([c for c in DEV if c != "GSPC"], key=lambda c: w_dip.loc[c, "2000-2012"]) + \
        sorted(EM, key=lambda c: w_dip.loc[c, "2000-2012"])
labels = d.drop_duplicates("code").set_index("code")["name"]
ypos = {c: len(order) - 1 - i + (0 if c in DEV else -0.6) for i, c in enumerate(order)}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 7.4), sharey=True)
fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.09, wspace=0.08)
for ax, w, ref, xl, lim in [(ax1, w_dip, 100, "Next-day volatility after a 3-SD surprise (% of the state's forecast)", (30, 140)),
                            (ax2, w_vr, 1.0, "Monthly variance ratio VR(21): above 1 = trending, below 1 = self-correcting", (0.6, 1.72))]:
    ax.axvline(ref, color=F.BASE, lw=1.2)
    for c in order:
        y = ypos[c]
        vals = [w.loc[c, e] if e in w.columns and pd.notna(w.loc[c, e]) else np.nan for e, _, _ in ERAS]
        fin = [v for v in vals if np.isfinite(v)]
        ax.plot([min(fin), max(fin)], [y, y], color=F.GRID, lw=2.5, zorder=1)
        for (e, lab, col), v in zip(ERAS, vals):
            if np.isfinite(v):
                ax.plot(np.clip(v, *lim), y, "o", color=col, ms=8, mec=F.SURF, mew=1.3, zorder=3,
                        label=lab if c == "GSPC" else None)
    ax.set_xlim(*lim)
    ax.set_xlabel(xl, fontsize=9.5)
    ax.grid(axis="y", visible=False)
ax1.set_yticks([ypos[c] for c in order], [labels[c] for c in order], fontsize=9.5)
first_em = [c for c in order if c in EM][0]
ax1.text(31, ypos[first_em] + 0.75, "EMERGING MARKETS", fontsize=8, color=F.MUTED, fontweight="bold", va="center")
ax1.text(101.5, ypos["GSPC"] + 0.7, "100% = no refractory effect", fontsize=8.5, color=F.MUTED)
ax2.text(1.01, ypos["GSPC"] + 0.7, "1 = random walk", fontsize=8.5, color=F.MUTED)
ax1.set_ylim(-1.3, len(order) + 0.3)
ax1.set_title("Refractory period: strongest in 2000–2012 almost everywhere", loc="left", fontsize=10, color=F.INK)
ax2.set_title("Feedback: trending before 2000, self-correcting after", loc="left", fontsize=10, color=F.INK)
h, l = ax1.get_legend_handles_labels()
fig.legend(h, l, loc="upper left", fontsize=9, ncol=3, bbox_to_anchor=(0.19, 0.865), handletextpad=0.3, columnspacing=1.2,
           frameon=False)
F.title(fig, "The market unit replicates across 15 markets, and its inhibition rose and fell worldwide in step",
        "Daily index closes (Yahoo). The two-speed volatility state is significant in every market; era differences in "
        "the refractory period survive controlling for stress.")
fig.savefig(RES / "figures" / "fig14_intl.png", dpi=200)
print("saved fig14")
