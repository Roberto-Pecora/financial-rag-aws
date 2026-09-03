from frag.rag.store_qdrant import QdrantStore


def test_search_uses_query_points_api(monkeypatch):
    class DummyClient:
        def __init__(self):
            self.calls = []

        def collection_exists(self, name):
            return True

        def get_collection(self, name):
            return type("Info", (), {"points_count": 1})()

        def create_collection(self, *args, **kwargs):
            return None

        def create_payload_index(self, *args, **kwargs):
            return None

        def query_points(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return []

    dummy_client = DummyClient()
    monkeypatch.setattr("frag.rag.store_qdrant.QdrantClient", lambda *args, **kwargs: dummy_client)
    monkeypatch.setattr(
        "frag.rag.store_qdrant.SentenceTransformer",
        lambda *args, **kwargs: type(
            "Embedder",
            (),
            {
                "encode": lambda self, text, convert_to_numpy=True: [0.1, 0.2],
                "get_sentence_embedding_dimension": lambda self: 2,
            },
        )(),
    )

    # Pin dense mode: this test targets the vector query_points path and must
    # not pick up a RETRIEVAL_MODE=hybrid from an ambient .env.
    store = QdrantStore(url="http://example.com", api_key="test", retrieval_mode="dense")
    store.search("hello", top_k=3)
    assert dummy_client.calls, "expected query_points to be called"
