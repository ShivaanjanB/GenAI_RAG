from src.common.evidence import Evidence, Field, ok, flag, now_iso

def compute_ev(market_cap: Field, cash: Field, debt: Field, as_of: str, url: str) -> Field:
    if market_cap.flagged or cash.flagged or debt.flagged:
        return flag("Missing market cap, cash or debt; cannot compute EV")
    try:
        ev = float(market_cap.value) + float(debt.value) - float(cash.value)
        return ok(
            ev,
            Evidence(
                source="Derived",
                url=url,
                retrieved_at=now_iso(),
                as_of=as_of,
                note="EV = MarketCap + Debt - Cash",
            ),
        )
    except Exception:
        return flag("EV computation failed")
