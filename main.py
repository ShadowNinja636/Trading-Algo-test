import asyncio
import csv
import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

app = FastAPI(title="SignalDesk")

# CORS middleware configured to accept any port on localhost or 127.0.0.1
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Standard browser User-Agent prevents 403 Forbidden errors from Yahoo/Nifty
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}
CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv"


class Settings(BaseModel):
    short_sma: int = Field(6, ge=2, le=100)
    long_sma: int = Field(30, ge=3, le=250)
    lookback_days: int = Field(252, ge=31, le=1000)
    max_stocks: int = Field(100, ge=1, le=100)

    @model_validator(mode="after")
    def check_sma_periods(self) -> "Settings":
        if self.short_sma >= self.long_sma:
            raise ValueError("Short SMA must be smaller than Long SMA")
        return self


def calculate_average(data: List[float]) -> float:
    return sum(data) / len(data) if data else 0.0


async def fetch_symbols(client: httpx.AsyncClient, limit: int) -> List[Dict[str, str]]:
    """Fetch official Nifty 100 constituent stock list."""
    try:
        res = await client.get(CSV_URL, headers=HEADERS)
        res.raise_for_status()
        reader = csv.DictReader(io.StringIO(res.text))

        symbols = []
        for row in reader:
            symbol = row.get("Symbol", "").strip()
            company = row.get("Company Name", "").strip()
            if symbol:
                symbols.append({"ticker": symbol, "company": company})
            if len(symbols) >= limit:
                break
        return symbols
    except Exception as e:
        raise HTTPException(
            status_code=502, detail=f"Failed to fetch stock constituents: {str(e)}"
        )


async def process_stock(
    client: httpx.AsyncClient,
    stock: Dict[str, str],
    settings: Settings,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Any]]:
    """Fetch daily candle data and detect moving-average crossovers."""
    async with semaphore:
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{stock['ticker']}.NS"
            res = await client.get(url, params={"range": "5y", "interval": "1d"}, headers=HEADERS)
            res.raise_for_status()

            chart_result = res.json()["chart"]["result"][0]
            timestamps = chart_result.get("timestamp", [])
            closes = chart_result["indicators"]["quote"][0].get("close", [])

            # Filter out missing null prices
            prices = [
                (datetime.fromtimestamp(ts, timezone.utc), price)
                for ts, price in zip(timestamps, closes)
                if price is not None
            ]

            if len(prices) < settings.long_sma:
                return None

            found = None
            start_index = max(settings.long_sma, len(prices) - settings.lookback_days)

            for i in range(start_index, len(prices)):
                # Current day short & long SMA values
                curr_short = calculate_average([v for _, v in prices[i - settings.short_sma + 1 : i + 1]])
                curr_long = calculate_average([v for _, v in prices[i - settings.long_sma + 1 : i + 1]])

                # Previous day short & long SMA values
                prev_short = calculate_average([v for _, v in prices[i - settings.short_sma : i]])
                prev_long = calculate_average([v for _, v in prices[i - settings.long_sma : i]])

                if prev_short <= prev_long and curr_short > curr_long:
                    found = ("BULLISH", prices[i][0], prices[i][1], curr_short, curr_long)
                elif prev_short >= prev_long and curr_short < curr_long:
                    found = ("BEARISH", prices[i][0], prices[i][1], curr_short, curr_long)

            if found:
                crossover_type, dt, close, short_val, long_val = found
                return {
                    **stock,
                    "crossover_type": crossover_type,
                    "crossover_date": dt.date().isoformat(),
                    "_dt": dt,
                    "close": close,
                    "short_sma": short_val,
                    "long_sma": long_val,
                }
        except Exception:
            return None
        return None


@app.get("/api/user")
async def get_user_profile():
    return {
        "user_name": "Local research workspace",
        "user_id": "Not connected",
        "products": ["Signal research", "Paper trading ready"],
        "exchanges": ["NSE daily data"],
    }


@app.post("/api/signals")
async def generate_signals(settings: Settings):
    try:
        async with httpx.AsyncClient(timeout=25, follow_redirects=True) as client:
            symbols = await fetch_symbols(client, settings.max_stocks)
            semaphore = asyncio.Semaphore(5)  # Limit concurrent requests to prevent rate limiting
            
            tasks = [
                process_stock(client, stock, settings, semaphore)
                for stock in symbols
            ]
            results = await asyncio.gather(*tasks)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Filter out empty results and sort by newest crossover date first
    valid_rows = [row for row in results if row is not None]
    sorted_rows = sorted(valid_rows, key=lambda x: x["_dt"], reverse=True)[: settings.max_stocks]

    # Format JSON payload response (strip internal datetime sorting helper)
    return [
        {k: v for k, v in dict(rank=idx + 1, **row).items() if k != "_dt"}
        for idx, row in enumerate(sorted_rows)
    ]