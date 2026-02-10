import requests
from datetime import datetime, timedelta
from src.common.evidence import Evidence, Field, ok, flag, now_iso

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def get_news_gdelt(company: str, as_of: str, days: int = 90) -> Field:
    try:
        end = datetime.fromisoformat(as_of)
        start = end - timedelta(days=days)
        params = {
            "query": f'"{company}"',
            "mode": "ArtList",
            "format": "json",
            "maxrecords": 20,
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
            "sort": "datedesc",
        }
        r = requests.get(GDELT_DOC_URL, params=params)
        r.raise_for_status()
        articles = [
            {
                "title": a.get("title"),
                "url": a.get("url"),
                "date": a.get("seendate"),
                "domain": a.get("domain"),
            }
            for a in r.json().get("articles", [])
        ]
        return ok(
            articles,
            Evidence(
                source="GDELT",
                url=r.url,
                retrieved_at=now_iso(),
                as_of=as_of,
            ),
        )
    except Exception as e:
        return flag(f"GDELT fetch failed: {e}")
