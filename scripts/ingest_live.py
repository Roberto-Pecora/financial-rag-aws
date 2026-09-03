from __future__ import annotations

import os

from dotenv import load_dotenv

from frag.rag.controller import RagController
from frag.sources.loader import load_live_sources

load_dotenv()

if __name__ == "__main__":
    controller = RagController()

    docs = load_live_sources(
        headers={"User-Agent": os.getenv("SEC_USER_AGENT")}
        if os.getenv("SEC_USER_AGENT")
        else None,
        fred_series_ids=["DGS10"],
        fred_api_key=os.getenv("FRED_API_KEY"),
    )

    indexed = controller.ingest(docs)
    print(
        {
            "documents_loaded": len(docs),
            "documents_indexed": indexed,
            "total_indexed": controller.store.count(),
        }
    )
