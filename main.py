import math, re, asyncio
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import yfinance as yf
import httpx
from bs4 import BeautifulSoup

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="VinStock Returns API", version="1.0.0")

# Allow GPT to call this from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

BENCHMARKS = ["SPY", "QQQ"]

# ---------- Helpers ----------
def _annualized_return(series: pd.Series, years: float) -> Optional[float]:
    if series is None or series.empty or years <= 0: return None
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    if start <= 0: return None
    return (end / start) ** (1.0 / years) - 1.0

def _simple_return(series: pd.Series) -> Optional[float]:
    if series is None or series.empty: return None
    start = float(series.iloc[0]); end = float(series.iloc[-1])
    if start <= 0: return None
    return (end / start) - 1.0

def _nearest_slice(px: pd.Series, years: int) -> pd.Series:
    if px is None or px.empty: return px
    last = px.index[-1]
    cutoff = last - pd.DateOffset(years=years)
    return px[px.index >= cutoff]

def _one_month_slice(px: pd.Series) -> pd.Series:
    return px.iloc[-22:] if px is not None and not px.empty else px

def _ytd_slice(px: pd.Series) -> pd.Series:
    if px is None or px.empty: return px
    last = px.index[-1]
    jan1 = pd.Timestamp(year=last.year, month=1, day=1, tz=px.index.tz)
    return px[px.index >= jan1]

def _pct_change_today(px: pd.Series) -> Optional[float]:
    if px is None or len(px) < 2: return None
    prev, last = float(px.iloc[-2]), float(px.iloc[-1])
    if prev == 0: return None
    return (last / prev) * 100.0 - 100.0

def _fmt_pct(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return float(x)

async def fetch_prices(tickers: List[str], years_back: int = 6) -> Dict[str, pd.Series]:
    start = datetime.now(timezone.utc) - timedelta(days=365*years_back + 15)
    data = yf.download(
        tickers=" ".join(tickers),
        start=start.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by='ticker',
        threads=True,
    )
    out: Dict[str, pd.Series] = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = data[(t, 'Close')].dropna().sort_index()
                out[t] = s
            except Exception:
                out[t] = pd.Series(dtype=float)
    else:
        out[tickers[0]] = data['Close'].dropna().sort_index() if 'Close' in data else pd.Series(dtype=float)
    return out

@lru_cache(maxsize=512)
def _rating_from_counts(buy: int, hold: int, sell: int) -> str:
    if max(buy, hold, sell) == 0: return "—"
    if buy >= hold and buy >= sell: return "Buy"
    if sell > buy and sell >= hold: return "Sell"
    return "Hold"

async def try_consensus_rating(ticker: str, timeout_s: float = 6.0) -> str:
    async with httpx.AsyncClient(timeout=timeout_s, headers={"User-Agent": "Mozilla/5.0"}) as client:
        buy = hold = sell = 0
        try:
            r = await client.get(f"https://stockanalysis.com/stocks/{ticker.lower()}/ratings/")
            if r.status_code == 200:
                txt = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
                b = re.search(r"\bBuy\s+(\d+)\b", txt, re.I)
                h = re.search(r"\bHold\s+(\d+)\b", txt, re.I)
                s = re.search(r"\bSell\s+(\d+)\b", txt, re.I)
                buy += int(b.group(1)) if b else 0
                hold += int(h.group(1)) if h else 0
                sell += int(s.group(1)) if s else 0
        except Exception:
            pass
        return _rating_from_counts(buy, hold, sell)

def compute_windows(px: pd.Series) -> Dict[str, Optional[float]]:
    out = {}
    for y in [5,4,3,2,1]:
        out[f"{y}y"] = _fmt_pct(_annualized_return(_nearest_slice(px, y), y))
    out["ytd"] = _fmt_pct(_simple_return(_ytd_slice(px)))
    out["1mo"] = _fmt_pct(_simple_return(_one_month_slice(px)))
    out["today_pl_pct"] = _fmt_pct(_pct_change_today(px))
    return out

def add_benchmark_deltas(row: Dict[str, Any], spy: Dict[str, Any], qqq: Dict[str, Any]) -> Dict[str, Any]:
    def delta(v, b): 
        return None if v is None or b is None else (v - b) * 100.0
    enriched = {}
    for key in ["5y","4y","3y","2y","1y","ytd","1mo"]:
        v = row.get(key)
        d_spy = delta(v, spy.get(key))
        d_qqq = delta(v, qqq.get(key))
        enriched[key] = {
            "value_pct": None if v is None else v * 100.0,
            "d_spy_pp": d_spy,
            "d_qqq_pp": d_qqq
        }
    enriched["today_pl_pct"] = row.get("today_pl_pct")
    return enriched

@app.get("/returns")
async def get_returns(tickers: List[str] = Query(..., description="Tickers like &tickers=AAPL&tickers=MSFT")):
    tickers = [t.upper().strip() for t in tickers if t and t.strip()]
    all_needed = list(dict.fromkeys(tickers + BENCHMARKS))
    prices = await fetch_prices(all_needed, years_back=6)

    bench_results = {b: compute_windows(prices.get(b, pd.Series(dtype=float))) for b in BENCHMARKS}
    out: Dict[str, Any] = {}
    tasks = []
    for t in tickers:
        px = prices.get(t, pd.Series(dtype=float))
        row = compute_windows(px)
        row_enriched = add_benchmark_deltas(row, bench_results["SPY"], bench_results["QQQ"])
        out[t] = {"metrics": row_enriched, "rating": "—"}
        tasks.append(try_consensus_rating(t))
    ratings = await asyncio.gather(*tasks, return_exceptions=True)
    for t, r in zip(tickers, ratings):
        out[t]["rating"] = r if isinstance(r, str) else "—"
    out["_benchmarks"] = bench_results
    return JSONResponse(out)
