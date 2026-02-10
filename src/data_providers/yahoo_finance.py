import yfinance as yf
from datetime import datetime
from src.common.evidence import Evidence, Field, ok, flag, now_iso

def yahoo_quote_url(ticker: str) -> str:
    return f"https://finance.yahoo.com/quote/{ticker.upper().strip()}"

def get_price_history_asof(ticker: str, as_of: str) -> Field:
    try:
        hist = yf.Ticker(ticker).history(period="max")
        hist = hist[hist.index.tz_localize(None) <= datetime.fromisoformat(as_of)]
        if hist.empty:
            return flag(f"No price data on/before {as_of} for {ticker}")
        last = hist.iloc[-1]
        return ok(
            {"close": float(last["Close"]), "date": str(hist.index[-1].date())},
            Evidence(
                source="Yahoo Finance",
                url=yahoo_quote_url(ticker),
                retrieved_at=now_iso(),
                as_of=as_of,
            ),
        )
    except Exception as e:
        return flag(f"Yahoo price fetch failed: {e}")
