"""Fetch 5-minute OHLCV history from Binance public klines into pipeline CSVs.

Dependency-free (stdlib urllib) and paginated, so it pulls deep history rather
than just the last 1000 candles. Output columns are open, high, low, close
(volume kept for later relative-volume experiments), oldest first, which is
exactly what mlm.data.load_ohlcv_csv consumes.

Note: api.binance.com is geo-restricted in some regions. If it fails, set
--base to a reachable mirror (e.g. https://data-api.binance.vision) which serves
the same klines endpoint, or point the pipeline at any CSV with these columns.
"""
import argparse
import csv
import json
import time
import urllib.request

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def fetch_klines(symbol, interval, start_ms, end_ms, base, limit=1000):
    rows = []
    step = INTERVAL_MS[interval] * limit
    t = start_ms
    while t < end_ms:
        url = (f"{base}/api/v3/klines?symbol={symbol}&interval={interval}"
               f"&startTime={t}&endTime={end_ms}&limit={limit}")
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.load(r)
        if not batch:
            break
        rows.extend(batch)
        t = batch[-1][0] + INTERVAL_MS[interval]
        time.sleep(0.2)   # stay well under the public rate limit
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for k in rows:
            w.writerow([k[0], k[1], k[2], k[3], k[4], k[5]])


def fetch_to_csv(symbol, interval="5m", days=3300,
                 base="https://api.binance.com"):
    """Fetch one symbol to '{symbol}_{interval}.csv', skipping if it exists.

    Cache-aware so a notebook kernel restart does not re-download. Returns the
    output path. `days` past the listing date simply fills to the symbol start.
    """
    import os
    out = f"{symbol.lower()}_{interval}.csv"
    if os.path.exists(out):
        return out
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    rows = fetch_klines(symbol, interval, start_ms, end_ms, base)
    save_csv(rows, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    p.add_argument("--interval", default="5m", choices=list(INTERVAL_MS))
    p.add_argument("--days", type=int, default=30, help="history length in days")
    p.add_argument("--base", default="https://api.binance.com")
    args = p.parse_args()

    end_ms = int(time.time() * 1000)
    for sym in args.symbols:
        out = fetch_to_csv(sym, args.interval, args.days, args.base)
        print(f"{sym}: -> {out}")


if __name__ == "__main__":
    main()
