from __future__ import annotations

import argparse
import os

from frag.rag.controller import RagController
from frag.sources.loader import load_live_sources, load_sample_sources
from frag.utils.config import configure_runtime


def main():
    configure_runtime()

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--with-fred", action="store_true")
    parser.add_argument("--fred-series", nargs="*", default=["DGS10"])
    args = parser.parse_args()

    if args.sample:
        docs = load_sample_sources()
    else:
        docs = load_live_sources(
            headers={"User-Agent": os.getenv("SEC_USER_AGENT")}
            if os.getenv("SEC_USER_AGENT")
            else None,
            fred_series_ids=args.fred_series if args.with_fred else [],
            fred_api_key=os.getenv("FRED_API_KEY"),
        )

    controller = RagController()
    indexed = controller.ingest(docs)
    print(
        {
            "documents_loaded": len(docs),
            "documents_indexed": indexed,
            "total_indexed": controller.count(),
        }
    )


if __name__ == "__main__":
    main()
