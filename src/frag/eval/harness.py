import pandas as pd

from frag.eval.metrics import mrr_at_k, ndcg_at_k, precision_at_k, recall_at_k
from frag.eval.relevance import chunk_is_relevant, extract_facts


def evaluate(predictions, golden, k_values=(1, 3, 5, 10)):
    """Score retrieval predictions against the golden set.

    Relevance is content-based: a retrieved chunk counts as a hit if its text
    contains the numeric facts from the golden row's reference_answer. This is
    independent of chunk boundaries / doc_ids, so runs over differently-chunked
    collections stay comparable. The IR metrics themselves are unchanged - they
    are fed synthetic position ids (the rank of each retrieved chunk) with the
    relevant positions as the gold set.
    """
    gmap = {g["query"]: g for g in golden}
    rows = []
    scored = []
    for p in predictions:
        g = gmap[p["query"]]
        facts = extract_facts(g.get("reference_answer", ""))
        retrieved = p.get("retrieved_docs", [])

        # A retrieved position is "gold" if its chunk text contains all facts.
        pred_ids = [str(i) for i in range(len(retrieved))]
        gold_ids = [
            str(i) for i, d in enumerate(retrieved) if chunk_is_relevant(d.get("text", ""), facts)
        ]

        row = {
            "query": p["query"],
            "critic_score": p.get("critic_score", 0.0),
        }
        # Queries whose reference_answer has no numeric facts can't be
        # content-scored; exclude them from the retrieval-metric averages
        # rather than counting them as automatic zeros.
        row["scorable"] = int(bool(facts))
        for k in k_values:
            if facts:
                row[f"recall@{k}"] = recall_at_k(pred_ids, gold_ids, k)
                row[f"precision@{k}"] = precision_at_k(pred_ids, gold_ids, k)
                row[f"mrr@{k}"] = mrr_at_k(pred_ids, gold_ids, k)
                row[f"ndcg@{k}"] = ndcg_at_k(pred_ids, gold_ids, k)
        rows.append(row)
        if facts:
            scored.append(row)

    # Summary averages over scorable queries only, so non-numeric queries
    # don't drag the retrieval metrics down.
    summary_df = pd.DataFrame(scored) if scored else pd.DataFrame(rows)
    summary = summary_df.mean(numeric_only=True).to_dict()
    summary["n_scored"] = len(scored)
    summary["n_total"] = len(rows)
    return {"rows": rows, "summary": summary}
