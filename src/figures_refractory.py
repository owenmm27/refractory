"""Chart for Test 9 (refractory period). Usage: python3 -I src/figures_refractory.py <project_root>"""
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
df = pd.read_csv(RES / "test9c_rolling.csv")
r = np.corrcoef(df["next_day_var_after_3sd_pct"], df["ret_ac1"])[0, 1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.08, hspace=0.38)
ax1.axhline(100, color=F.BASE, lw=1)
ax1.plot(df["mid"], df["next_day_var_after_3sd_pct"], color=F.BLUE, marker="o", ms=4.5, mec=F.SURF, mew=1)
ax1.set_ylim(25, 135)
ax1.set_ylabel("% of expected")
ax1.text(1996, 104, "100% = no refractory effect", color=F.MUTED, fontsize=9)
ax1.set_title("Next-day volatility after a 3-SD surprise, as a share of what the volatility state implies (lower = stronger refractory period)",
              loc="left", fontsize=10, color=F.INK)
ax2.axhline(0, color=F.BASE, lw=1)
ax2.plot(df["mid"], df["ret_ac1"], color=F.ORANGE, marker="o", ms=4.5, mec=F.SURF, mew=1)
ax2.set_ylabel("Autocorrelation")
ax2.set_title("Lag-1 autocorrelation of daily returns (negative = today tends to reverse yesterday)", loc="left", fontsize=10, color=F.INK)
ax2.set_xlim(1926, 2025)
ax2.set_xlabel("Middle year of each 6-year window")
F.title(fig, "The market grew a refractory period as it turned self-correcting",
        f"US market (CRSP), rolling 6-year windows. Across windows the two measures move together (r = {r:.2f}).")
fig.savefig(RES / "figures" / "fig9_refractory.png", dpi=200)
print("saved fig9")
