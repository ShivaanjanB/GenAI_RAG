from typing import Dict, Any
from src.common.evidence import flag
from src.data_providers.sec_edgar import SECClient, SECFetchConfig, extract_financial_snapshot_us
from src.data_providers.yahoo_finance import get_price_history_asof, get_market_cap_asof, yahoo_quote_url
from src.data_providers.news_feed import get_news_gdelt
from src.valuation.engine import compute_ev

def build_public_comp_dashboard(ticker: str, as_of: str, sec_user_agent: str) -> Dict[str, Any]:
    sec = SECClient(SECFetchConfig(user_agent=sec_user_agent, as_of=as_of))
    fin = extract_financial_snapshot_us(sec, ticker)
    price = get_price_history_asof(ticker, as_of)
    mcap = get_market_cap_asof(ticker, as_of)
    ev = compute_ev(mcap, fin.get("cash", flag("missing cash")), fin.get("debt", flag("missing debt")), as_of, yahoo_quote_url(ticker))
    news = get_news_gdelt(ticker, as_of, days=90)
    # compute multiples, build return dict, etc.
    return {
        "ticker": ticker.upper(),
        "as_of": as_of,
        "market_data": {"price": price.__dict__, "market_cap": mcap.__dict__, "enterprise_value": ev.__dict__},
        "financials": {k: v.__dict__ for k, v in fin.items()},
        "news": news.__dict__,
    }
