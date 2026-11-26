# Data sources

Everything is daily. Most of the data belongs to third parties whose terms don't allow redistribution, so it is **not in this repository**. `./reproduce.sh download` fetches it into `data/`. A few small files that are public-domain or openly published, and that pin down the exact sample, are included.

The results in this repo were produced with data pulled on 21–25 November 2026.

## Downloaded by the scripts (not included)

| Data | Source | Span used | Fetched by |
|---|---|---|---|
| US market return (Mkt-RF + RF) and 49 value-weighted industry portfolios | [Ken French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html), built from CRSP | Jul 1926 – Sep 2026 | `src/download.py` |
| VIX close | [Cboe](https://www.cboe.com/tradable_products/vix/vix_historical_data/) | Jan 1990 – Nov 2026 | `src/download.py` |
| Adjusted closes and volume for current S&P 500 members, and the index | Yahoo Finance chart API | Jan 1990 – Nov 2026 | `src/download.py` |
| Daily closes for 15 stock indices | Yahoo Finance chart API | 1960 – Nov 2026, varies by index | `src/download_intl.py` |
| Open, high, low, close for the 15 indices, SPY and the 237 panel stocks | Yahoo Finance chart API | as above | `src/download_ohlc.py` |
| SPY open and close | Yahoo Finance chart API | Jan 1993 – Nov 2026 | `src/test5_options.py` |
| Dealer-gamma (GEX) and dark-pool (DIX) estimates | [SqueezeMetrics](https://squeezemetrics.com/monitor/dix) | May 2011 – Nov 2026 | `src/test5_options.py` |

Notes:
- The Yahoo endpoint is unofficial. It can change or rate-limit without notice; the downloaders retry and pause between requests.
- Yahoo timestamps are converted to each exchange's local time before taking the date, so Asian and Australian sessions land on the right day.
- Usable index highs and lows start later than closes (S&P 500: 1962; others between 1984 and 2007). The start dates are listed in `src/test15b_range.py` (`H_L_START`).

## Included in the repository

| File | What | Source | Why it's included |
|---|---|---|---|
| `data/meta/constituents.csv` | S&P 500 members (503 tickers, with sector and SEC CIK) | [datasets/s-and-p-500-companies](https://github.com/datasets/s-and-p-500-companies), compiled from Wikipedia; see that repository for its license | Defines the stock panel. The list changes over time, so a fresh download would give a different panel. |
| `data/earnings/earnings_8k_202.csv` | 22,216 earnings-release filings (8-K Item 2.02) for the 237 panel stocks, Aug 2004 – Nov 2026 | [SEC EDGAR](https://www.sec.gov/search-filings) submissions API, via `src/download_earnings.py` | US government public data; slow to rebuild (one request per company, throttled) |
| `data/macro/fomc_dates.csv` | 249 FOMC meetings and unscheduled actions, 2000–2026 | Federal Reserve [meeting calendars](https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm) and historical pages, parsed by `src/parse_fomc.py` | Public; the parser needs pages saved by hand |
| `data/macro/cpi_release_dates.txt` | CPI release dates, 1999–2026 | BLS [CPI release archive](https://www.bls.gov/bls/news-release/cpi.htm) | Public; compiled by hand |
| `data/macro/jobs_report_release_dates.txt` | Employment Situation release dates, 1999–2026 | BLS [Employment Situation archive](https://www.bls.gov/bls/news-release/empsit.htm) | Public; compiled by hand |

To rebuild the FOMC dates, save `fomccalendars.htm` and `fomchistorical2000.htm` … `fomchistorical2020.htm` from federalreserve.gov into `data/macro/raw/`; `reproduce.sh` then reruns the parser.

## Known data issues

- **Survivorship bias.** The stock panel is today's S&P 500 members with near-complete history since 1990 (237 stocks). Firms that failed or left the index are missing.
- **Price indices.** The international indices are price indices (no dividends). Some emerging-market indices show stale-price effects.
- **Stale or missing highs and lows** early in some index histories are dropped (see `H_L_START`).
