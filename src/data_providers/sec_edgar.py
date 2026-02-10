# Simplified excerpt; see the full example for error handling and unit parsing.
import json, time, requests
from dataclasses import dataclass
from src.common.evidence import Evidence, Field, ok, flag, now_iso

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_COMPANYFACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

@dataclass
class SECFetchConfig:
    user_agent: str
    as_of: str
    sleep_seconds: float = 0.2

class SECClient:
    def __init__(self, cfg: SECFetchConfig):
        self.cfg = cfg
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent})

    def cik_from_ticker(self, ticker: str) -> Field:
        data = self.session.get(SEC_TICKERS_URL).json()
        for row in data.values():
            if row["ticker"].upper() == ticker.upper():
                cik_str = str(row["cik_str"]).zfill(10)
                return ok(
                    cik_str,
                    Evidence(
                        source="SEC company_tickers.json",
                        url=SEC_TICKERS_URL,
                        retrieved_at=now_iso(),
                        as_of=self.cfg.as_of,
                        note=f"ticker={ticker}",
                    ),
                )
        return flag(f"SEC CIK not found for ticker={ticker}")
# ... plus helper functions to extract revenue, cash, etc. from companyfacts.
