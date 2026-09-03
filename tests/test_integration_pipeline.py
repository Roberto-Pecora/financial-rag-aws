from frag.rag.actor import Actor
from frag.rag.controller import RagController
from frag.rag.critic import Critic


class FakeStore:
    def __init__(self):
        self.documents = []

    def ingest(self, docs):
        self.documents.extend(docs)
        return len(docs)

    def search(self, query, top_k=8, metadata_filter=None):
        """Returns {text, metadata, score} — matching QdrantStore.search() shape.

        Matches on any individual word in the query so realistic query strings
        like 'What changed in revenue?' still hit docs containing 'revenue'.
        """
        terms = [t.strip("?,.!").lower() for t in query.split()]
        return [
            {
                "text": doc["text"],
                "metadata": doc.get("metadata", {}),
                "score": 1.0,
            }
            for doc in self.documents
            if any(term in doc["text"].lower() for term in terms)
        ][:top_k]

    def count(self):
        return len(self.documents)


class FakeActorLLM:
    def generate(self, prompt):
        return '{"answer":"Revenue increased.","citations":["doc-1"]}'


class FakeCriticLLM:
    def generate(self, prompt):
        return (
            '{"overall_score":0.9,"faithfulness_score":0.95,'
            '"completeness_score":0.88,"citation_score":0.87,"issues":[]}'
        )


SAMPLE_DOC = {
    "id": "doc-1",
    "text": "Revenue increased and liquidity remained strong.",
    "metadata": {"doc_id": "doc-1", "source_type": "test"},
}


def test_actor_uses_store_shape_contexts():
    """Actor.act() must not KeyError when given {text, metadata, score} contexts."""
    store = FakeStore()
    store.ingest([SAMPLE_DOC])
    contexts = store.search("Revenue", top_k=3)

    actor = Actor(llm_client=FakeActorLLM())
    result = actor.act("What changed in revenue?", contexts)

    assert result["answer"] == "Revenue increased."
    assert result["citations"] == ["doc-1"]


def test_critic_uses_store_shape_contexts():
    """Critic.critique() must not KeyError when given {text, metadata, score} contexts."""
    store = FakeStore()
    store.ingest([SAMPLE_DOC])
    contexts = store.search("Revenue", top_k=3)

    actor = Actor(llm_client=FakeActorLLM())
    critic = Critic(llm_client=FakeCriticLLM())
    actor_out = actor.act("What changed in revenue?", contexts)
    result = critic.critique(
        "What changed in revenue?", contexts, actor_out["answer"], actor_out["citations"]
    )

    assert result["score"] == 0.9
    assert "faithfulness=0.95" in result["notes"]


def test_answer_with_critique_accepted():
    """RagController.answer_with_critique() returns accepted when score >= min_score."""
    store = FakeStore()
    store.ingest([SAMPLE_DOC])

    controller = RagController(store=store)
    controller.actor = Actor(llm_client=FakeActorLLM())
    controller.critic = Critic(llm_client=FakeCriticLLM())
    controller.min_score = 0.8

    result = controller.answer_with_critique("What changed in revenue?", top_k=3)

    assert result["status"] == "accepted"
    assert result["answer"] == "Revenue increased."
    assert result["citations"] == ["doc-1"]
    assert result["critic_score"] == 0.9
    assert "faithfulness" in result["critic_notes"]


def test_answer_with_critique_abstained():
    """RagController.answer_with_critique() returns abstained when score < min_score."""
    store = FakeStore()
    store.ingest([SAMPLE_DOC])

    controller = RagController(store=store)
    controller.actor = Actor(llm_client=FakeActorLLM())
    controller.critic = Critic(llm_client=FakeCriticLLM())
    controller.min_score = 0.99  # force abstain

    result = controller.answer_with_critique("What changed in revenue?", top_k=3)

    assert result["status"] == "abstained"


def test_answer_with_critique_no_results():
    """Returns abstained with score 0.0 when store returns no contexts."""
    store = FakeStore()  # empty

    controller = RagController(store=store)
    controller.actor = Actor(llm_client=FakeActorLLM())
    controller.critic = Critic(llm_client=FakeCriticLLM())

    result = controller.answer_with_critique("What changed in revenue?", top_k=3)

    assert result["answer"] == "Insufficient evidence retrieved."
    assert result["critic_score"] == 0.0


def test_actor_only_path_accepts_without_critic():
    """Default path (CRITIC=off): actor answers, no critic call, critic_score is None."""
    store = FakeStore()
    store.ingest([SAMPLE_DOC])

    controller = RagController(store=store)  # critic disabled by default in tests
    controller.actor = Actor(llm_client=FakeActorLLM())

    assert controller.critic is None
    result = controller.answer_with_critique("What changed in revenue?", top_k=3)

    assert result["status"] == "accepted"
    assert result["answer"] == "Revenue increased."
    assert result["citations"] == ["doc-1"]
    assert result["critic_score"] is None
    assert "critic disabled" in result["critic_notes"]


def test_actor_only_path_abstains_on_no_evidence():
    """Default path abstains when the store returns nothing, without a critic."""
    store = FakeStore()  # empty

    controller = RagController(store=store)
    controller.actor = Actor(llm_client=FakeActorLLM())

    assert controller.critic is None
    result = controller.answer_with_critique("What changed in revenue?", top_k=3)

    assert result["status"] == "abstained"
    assert result["answer"] == "Insufficient evidence retrieved."
    assert result["critic_score"] is None
