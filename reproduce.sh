#!/usr/bin/env bash
# Rebuild everything: download the data, run Tests 1-15, redraw the figures.
#
#   ./reproduce.sh            # all three stages (several hours; Test 10 alone is ~80 min)
#   ./reproduce.sh download   # fetch third-party data into data/ (~250 MB)
#   ./reproduce.sh tests      # run the tests; writes results/*.json, *.csv, *.log
#   ./reproduce.sh figures    # redraw results/figures/*.png from results/ and data/
#
# Set PYTHON to choose the interpreter (default: python3).
set -euo pipefail

cd "$(dirname "$0")"
PY="${PYTHON:-python3}"
STAGE="${1:-all}"

# Single-threaded linear algebra: the tests parallelize across processes themselves,
# and nested BLAS threads make an 8 GB machine thrash.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

run() {
  local script="$1"; shift
  local t0=$SECONDS
  echo "==> $script $*"
  "$PY" -I "src/$script" . "$@"
  echo "    done in $((SECONDS - t0)) s"
}

download() {
  run download.py           # Ken French, CBOE VIX, S&P 500 member list, Yahoo stock panel
  run download_intl.py      # 15 stock indices (Yahoo)
  run download_ohlc.py      # highs and lows for the indices, SPY and the stock panel (Yahoo)
}

tests() {
  mkdir -p results/cache results/figures
  run test1_predictability.py           # ~5 min
  run test2_synchrony.py
  run test3_avalanches.py 4
  run test3_avalanches.py 3
  run test3_avalanches.py 5
  run test4_balance.py
  run test5_options.py                  # also fetches SPY prices and the SqueezeMetrics DIX/GEX file
  run test6a_timescales_market.py       # ~10 min
  run test6b_timescales_stocks.py       # ~10 min
  run test6c_closure.py
  run test6d_sector_null.py
  run test6e_pca_null.py
  run test7_contagion.py                # ~10 min

  # Earnings dates (SEC EDGAR) are committed; rebuild only if missing.
  if [[ ! -f data/earnings/earnings_8k_202.csv ]]; then
    "$PY" -I -c "import numpy as np, pandas as pd; d = np.load('results/cache/z_2unit.npz', allow_pickle=True); pd.DataFrame({'symbol': d['symbols']}).to_csv('results/cache/panel_symbols.csv', index=False)"
    run download_earnings.py results/cache/panel_symbols.csv
  fi
  run test8_earnings.py

  # FOMC dates are committed; rebuild only if the saved Fed calendar pages are present (see DATA.md).
  if [[ -d data/macro/raw ]]; then run parse_fomc.py; fi
  run test9_scheduled_news.py
  run test9b_refractory.py
  run test9c_refractory_rolling.py
  run test10_refractory_where.py        # ~80 min
  run test10b_modelfree.py
  run test11_gamma.py
  run test12_brain_model.py 20
  run test12c_model_v2.py 20
  run test12b_oos.py
  run test12d_rolling_oos.py
  run test13_bursts.py 10
  run test13b_model_v3.py 20
  run test14_intl.py
  run test14b_era_vs_stress.py
  run test15_score_driven.py 40         # ~2 min
  run test15b_range.py                  # ~10 min
}

figures() {
  mkdir -p results/figures
  # figures.py draws Figs 1-4 and the shared style; the later scripts import it, so they redraw 1-4 too.
  for f in figures.py figures_timescales.py figures_contagion.py figures_earnings.py figures_refractory.py \
           figures_where.py figures_gamma.py figures_brain_model.py figures_bursts.py figures_intl.py \
           figures_hardening.py figures_hero.py; do
    run "$f"
  done
}

case "$STAGE" in
  download) download ;;
  tests)    tests ;;
  figures)  figures ;;
  all)      download; tests; figures ;;
  *) echo "usage: $0 [download|tests|figures|all]" >&2; exit 2 ;;
esac
