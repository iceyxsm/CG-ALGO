"""Fetch 5-minute OHLCV history from Binance public klines into pipeline CSVs.

Dependency-free (stdlib urllib) and paginated, so it pulls deep history rather
than just the last 1000 candles. Output columns are open, high, low, close
(volume kept for later relative-volume experiments), oldest first, which is
exactly what mlm.data.load_ohlcv_csv consumes.

Note: api.binance.com returns HTTP 451 from some regions (including the US, and
therefore Colab). The default base here is data-api.binance.vision, a geo-neutral
mirror that serves the identical klines endpoint, so it works on hosted notebooks
out of the box. Pass --base https://api.binance.com if you prefer the main API.
"""
import argparse
import csv
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}
DEFAULT_BASE = "https://data-api.binance.vision"


def _get_page(symbol, interval, start, end, base, limit, retries=4):
    """Fetch one page, retrying with backoff on rate-limit/transient errors."""
    url = (f"{base}/api/v3/klines?symbol={symbol}&interval={interval}"
           f"&startTime={start}&endTime={end}&limit={limit}")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(0.5 * (2 ** attempt))   # exponential backoff


def fetch_klines(symbol, interval, start_ms, end_ms, base, limit=1000,
                 progress=False, workers=20):
    """Fetch klines in parallel.

    Page start times are arithmetic (limit * interval apart), so every page's
    window is known up front without waiting for the previous response. That
    lets the pages download concurrently instead of strictly sequentially,
    turning a multi-minute pull into a sub-minute one. Workers are capped to
    stay within Binance's public rate limit, and each page retries with backoff.
    """
    span = INTERVAL_MS[interval] * limit
    starts = list(range(start_ms, end_ms, span))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(
            lambda s: _get_page(symbol, interval, s, min(s + span, end_ms),
                                base, limit), starts))
    rows = []
    for batch in results:
        if batch:
            rows.extend(batch)
    rows.sort(key=lambda k: k[0])
    # De-duplicate on open time in case page windows overlap at the seams.
    dedup = []
    seen = None
    for k in rows:
        if k[0] != seen:
            dedup.append(k)
            seen = k[0]
    if progress:
        print(f"  {symbol}: {len(dedup):,} candles done", flush=True)
    return dedup


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for k in rows:
            w.writerow([k[0], k[1], k[2], k[3], k[4], k[5]])


def fetch_to_csv(symbol, interval="5m", days=3300, base=DEFAULT_BASE):
    """Fetch one symbol to '{symbol}_{interval}.csv', skipping if it exists.

    Cache-aware so a notebook kernel restart does not re-download. Returns the
    output path. `days` past the listing date simply fills to the symbol start.
    """
    import os
    out = f"{symbol.lower()}_{interval}.csv"
    if os.path.exists(out):
        print(f"  {symbol}: cached -> {out}")
        return out
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    rows = fetch_klines(symbol, interval, start_ms, end_ms, base, progress=True)
    save_csv(rows, out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    p.add_argument("--interval", default="5m", choices=list(INTERVAL_MS))
    p.add_argument("--days", type=int, default=30, help="history length in days")
    p.add_argument("--base", default=DEFAULT_BASE)
    args = p.parse_args()

    end_ms = int(time.time() * 1000)
    for sym in args.symbols:
        out = fetch_to_csv(sym, args.interval, args.days, args.base)
        print(f"{sym}: -> {out}")


if __name__ == "__main__":
    main()
