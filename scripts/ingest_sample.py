from frag.rag.controller import RagController
from frag.sources.loader import load_sample_sources

c = RagController()
c.ingest(load_sample_sources())
print(c.query("What changed in revenue and liquidity?", top_k=3))
