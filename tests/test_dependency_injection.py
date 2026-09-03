from frag.rag.actor import Actor
from frag.rag.controller import RagController
from frag.rag.critic import Critic


class InMemoryStore:
    def __init__(self):
        self.documents = []

    def ingest(self, docs):
        self.documents.extend(docs)
        return len(docs)

    def search(self, query, top_k=8, metadata_filter=None):
        return [
            {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": 1.0,
            }
            for doc in self.documents
            if query.lower() in doc["text"].lower()
        ][:top_k]

    def count(self):
        return len(self.documents)


class FakeLLMClient:
    def generate(self, prompt):
        return '{"answer":"stub answer","citations":["doc-1"]}'


def test_controller_accepts_injected_store():
    store = InMemoryStore()
    controller = RagController(store=store)

    indexed = controller.ingest(
        [{"id": "doc-1", "text": "example", "metadata": {"source_type": "test"}}]
    )

    assert indexed == 1
    assert controller.count() == 1


def test_actor_and_critic_accept_injected_llm_client():
    actor = Actor(llm_client=FakeLLMClient())
    critic = Critic(llm_client=FakeLLMClient())

    actor_response = actor.act("What changed?", [{"doc_id": "doc-1", "text": "example"}])
    critic_response = critic.critique(
        "What changed?",
        [{"doc_id": "doc-1", "text": "example"}],
        "stub answer",
        ["doc-1"],
    )

    assert actor_response["answer"] == "stub answer"
    assert actor_response["citations"] == ["doc-1"]
    assert critic_response["score"] >= 0.0
