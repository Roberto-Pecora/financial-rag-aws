"""Knowledge graph: extraction (gazetteer + fake LLM), graph ops, graphRAG search."""

from __future__ import annotations

from frag.kg import extract as ex
from frag.kg import graph_rag as gr
from frag.kg.graph import PropertyGraph
from frag.kg.schema import Entity, Relation, entity_id

_GAZ = {
    "Issuer": {"Acme Corp": ["Acme"]},
    "Sponsor": {"Blackstone": []},
}


class _FakeLLM:
    def __init__(self, raw):
        self._raw = raw

    def generate(self, prompt):
        return self._raw


# -- extraction ------------------------------------------------------------


def test_gazetteer_extracts_known_entities():
    ents = ex.GazetteerExtractor(_GAZ).extract("Acme announced a deal; Blackstone is the sponsor.")
    names = {e.name for e in ents}
    assert names == {"Acme Corp", "Blackstone"}


def test_llm_extractor_parses_entities_and_relations():
    raw = (
        '{"entities": [{"type":"Issuer","name":"Acme Corp"},'
        '{"type":"Instrument","name":"2031 Notes"}],'
        '"relations": [{"source":"Acme Corp","type":"issues","target":"2031 Notes"}]}'
    )
    ents, rels = ex.LLMExtractor(llm=_FakeLLM(raw)).extract("...")
    assert {e.name for e in ents} == {"Acme Corp", "2031 Notes"}
    assert rels[0].type == "issues"


def test_hybrid_extract_resolves_relation_endpoints_to_ids():
    raw = (
        '{"entities": [{"type":"Instrument","name":"2031 Notes"}],'
        '"relations": [{"source":"Acme Corp","type":"issues","target":"2031 Notes"}]}'
    )
    ents, rels = ex.hybrid_extract("Acme issued the 2031 Notes.", gazetteer=_GAZ, llm=_FakeLLM(raw))
    ids = {e.id for e in ents}
    assert entity_id("Issuer", "Acme Corp") in ids  # from gazetteer
    assert rels[0].source == entity_id("Issuer", "Acme Corp")  # relation linked to gazetteer id
    assert rels[0].target == entity_id("Instrument", "2031 Notes")


def test_llm_extractor_handles_bad_json():
    assert ex.LLMExtractor(llm=_FakeLLM("nope")).extract("x") == ([], [])


def test_hybrid_extract_stamps_provenance():
    """doc_id flows onto entities (source_docs) and relations (source_doc)."""
    raw = '{"entities": [], "relations": [{"source":"Acme","type":"issues","target":"Notes"}]}'
    ents, rels = ex.hybrid_extract(
        "Acme is the sponsor.", gazetteer=_GAZ, llm=_FakeLLM(raw), doc_id="doc-1"
    )
    acme = next(e for e in ents if e.name == "Acme Corp")
    assert acme.attrs["source_docs"] == ["doc-1"]
    assert rels[0].attrs["source_doc"] == "doc-1"


# -- graph -----------------------------------------------------------------


def _sample_graph():
    pg = PropertyGraph()
    a = Entity(entity_id("Issuer", "Acme Corp"), "Issuer", "Acme Corp")
    n = Entity(entity_id("Instrument", "2031 Notes"), "Instrument", "2031 Notes")
    c = Entity(entity_id("Covenant", "Restricted Payments"), "Covenant", "Restricted Payments")
    for e in (a, n, c):
        pg.add_entity(e)
    pg.add_relation(Relation(a.id, "issues", n.id))
    pg.add_relation(Relation(n.id, "has_covenant", c.id))
    return pg, a, n, c


def test_graph_add_resolve_neighbours_subgraph():
    pg, a, n, c = _sample_graph()
    assert pg.resolve("acme corp") == a.id
    assert pg.resolve("Acme Crop") == a.id  # fuzzy
    assert pg.resolve("nonexistent entity xyz") is None
    assert n.id in pg.neighbours(a.id, depth=1)
    assert c.id in pg.neighbours(a.id, depth=2)  # two hops
    text = pg.subgraph_text(a.id, depth=1)
    assert "Acme Corp issues 2031 Notes" in text


def test_add_entity_unions_provenance_across_docs():
    """The same covenant seen in two contracts accumulates both source docs."""
    pg = PropertyGraph()
    cid = entity_id("Covenant", "Change of Control")
    pg.add_entity(Entity(cid, "Covenant", "Change of Control", {"source_docs": ["c1"]}))
    pg.add_entity(Entity(cid, "Covenant", "Change of Control", {"source_docs": ["c2"]}))
    assert pg.provenance(cid) == ["c1", "c2"]


def test_graph_persistence_roundtrip(tmp_path):
    pg, *_ = _sample_graph()
    p = str(tmp_path / "g.json")
    pg.save(p)
    loaded = PropertyGraph.load(p)
    assert loaded.g.number_of_nodes() == 3
    assert loaded.g.number_of_edges() == 2


# -- graphRAG --------------------------------------------------------------


def test_graph_rag_search_returns_store_shape():
    pg, *_ = _sample_graph()
    retriever = gr.GraphRAGRetriever(pg, depth=1)
    hits = retriever.search("what covenant is on the 2031 Notes", top_k=5)
    assert hits
    assert {"text", "metadata", "score"} <= set(hits[0])
    assert hits[0]["metadata"]["source"] == "graph"
    assert "has_covenant" in hits[0]["text"]


def test_graph_rag_hit_carries_provenance():
    """A graphRAG hit cites the source docs of the resolved entity."""
    pg = PropertyGraph()
    did = entity_id("Instrument", "Loan Agreement")
    cid = entity_id("Covenant", "Change of Control")
    pg.add_entity(Entity(did, "Instrument", "Loan Agreement", {"source_docs": ["c1"]}))
    pg.add_entity(Entity(cid, "Covenant", "Change of Control", {"source_docs": ["c1", "c2"]}))
    pg.add_relation(Relation(did, "has_covenant", cid))
    hits = gr.GraphRAGRetriever(pg).search("change of control", top_k=3)
    assert hits and hits[0]["metadata"]["source_docs"] == ["c1", "c2"]


def test_graph_rag_empty_query():
    pg, *_ = _sample_graph()
    assert gr.GraphRAGRetriever(pg).search("") == []
