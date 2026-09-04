from __future__ import annotations

import argparse
import ast
import json
import os
import time

import mlflow
import pandas as pd
from dotenv import load_dotenv

from frag.eval.harness import evaluate
from frag.rag import prompts
from frag.rag.controller import RagController

load_dotenv()

GOLDEN_PATH = "data/golden_curated.csv"
RESULTS_PATH = "data/eval_results.json"
PREDICTIONS_PATH = "data/eval_predictions.jsonl"
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "frag-retrieval-eval"


def _parse_literal(value):
    return ast.literal_eval(value) if isinstance(value, str) and value.strip() else None


def load_golden(path: str) -> list[dict]:
    df = pd.read_csv(path)
    golden = df.to_dict(orient="records")
    for row in golden:
        row["gold_doc_ids"] = _parse_literal(row["gold_doc_ids"]) or []
        row["metadata_filter"] = _parse_literal(row.get("metadata_filter"))
    return golden


def run_predictions(
    controller: RagController, golden: list[dict], max_retries: int = 1, retry_delay: float = 5.0
) -> tuple[list[dict], list[str]]:
    predictions = []
    skipped = []
    for g in golden:
        for attempt in range(max_retries + 1):
            try:
                result = controller.answer_with_critique(
                    g["query"], top_k=10, metadata_filter=g.get("metadata_filter")
                )
                break
            except Exception as e:
                if attempt < max_retries:
                    print(f"  retrying {g['query']!r} after error: {e}")
                    time.sleep(retry_delay)
                else:
                    print(f"  SKIPPING {g['query']!r} after {max_retries + 1} attempts: {e}")
                    skipped.append(g["query"])
                    result = None
        if result is None:
            continue
        retrieved_docs = [
            {"doc_id": r["metadata"].get("doc_id", ""), "text": r.get("text", "")}
            for r in result["results"]
        ]
        predictions.append(
            {
                "query": g["query"],
                "retrieved_docs": retrieved_docs,
                "actor_answer": result["answer"],
                "critic_score": result["critic_score"],
            }
        )
    return predictions, skipped


def _collection_point_count(collection_name: str) -> int | None:
    """Lightweight point-count lookup via Qdrant's REST API directly, so
    --finish doesn't need to load a full RagController (embedding model)
    just to read collection metadata."""
    import requests

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    api_key = os.getenv("QDRANT_API_KEY")
    headers = {"api-key": api_key} if api_key else {}
    try:
        r = requests.get(f"{url}/collections/{collection_name}", headers=headers, timeout=10)
        r.raise_for_status()
        return r.json()["result"]["points_count"]
    except Exception:
        return None


def cmd_batch(args: argparse.Namespace) -> None:
    """Run predictions for golden[start:end] and append them to PREDICTIONS_PATH.

    Runs as its own process invocation so memory (embedding model, Qdrant
    client, Ollama connections) is fully released when it exits, keeping
    peak memory low on memory-constrained machines when scripted as
    several small batches instead of one long-running process.
    """
    golden = load_golden(GOLDEN_PATH)[args.start : args.end]
    if not golden:
        print(f"No golden rows in slice [{args.start}:{args.end}], nothing to do.")
        return

    controller = RagController()
    predictions, skipped = run_predictions(controller, golden)

    with open(PREDICTIONS_PATH, "a") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    print(
        f"Batch [{args.start}:{args.end}]: {len(predictions)} predictions, {len(skipped)} skipped"
    )
    if skipped:
        for q in skipped:
            print(f"  SKIPPED: {q}")
    print(f"Appended to {PREDICTIONS_PATH}")


def cmd_finish(args: argparse.Namespace) -> None:
    """Read accumulated predictions from PREDICTIONS_PATH, score them against
    the golden set, and log a single MLflow run."""
    golden = load_golden(GOLDEN_PATH)

    predictions = []
    with open(PREDICTIONS_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    scored_queries = {p["query"] for p in predictions}
    all_queries = {g["query"] for g in golden}
    missing = all_queries - scored_queries

    report = evaluate(predictions, golden)

    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("Summary (mean across queries):")
    for metric, value in report["summary"].items():
        print(f"  {metric:16s} {value:.3f}")
    if missing:
        print(f"\nMissing {len(missing)}/{len(golden)} queries (not found in {PREDICTIONS_PATH}):")
        for q in sorted(missing):
            print(f"  - {q}")
    print(f"\nWrote {RESULTS_PATH}")

    collection_name = os.getenv("QDRANT_COLLECTION", "sec_filings")
    collection_points = _collection_point_count(collection_name)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_param("golden_path", GOLDEN_PATH)
        mlflow.log_param("golden_size", len(golden))
        mlflow.log_param(
            "embedding_model",
            os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        )
        mlflow.log_param("collection_name", collection_name)
        if collection_points is not None:
            mlflow.log_param("collection_points", collection_points)
        mlflow.log_param("retrieval_mode", os.getenv("RETRIEVAL_MODE", "dense"))
        mlflow.log_param("actor_model", os.getenv("ACTOR_MODEL", ""))
        mlflow.log_param("critic_model", os.getenv("CRITIC_MODEL", ""))
        # Prompt versions in use, so runs are comparable across prompt A/B tests.
        mlflow.log_param(
            "actor_prompt_version", prompts.resolve_version("actor", "ACTOR_PROMPT_VERSION")
        )
        mlflow.log_param(
            "critic_prompt_version", prompts.resolve_version("critic", "CRITIC_PROMPT_VERSION")
        )
        mlflow.log_param("chunk_size", os.getenv("CHUNK_SIZE", "600"))
        mlflow.log_param("chunk_overlap", os.getenv("CHUNK_OVERLAP", "100"))
        mlflow.log_param("chunk_strategy", os.getenv("CHUNK_STRATEGY", "window"))
        mlflow.log_param("scoring", "content")
        mlflow.log_param("missing_queries", len(missing))
        if args.note:
            mlflow.set_tag("note", args.note)
        for metric, value in report["summary"].items():
            mlflow.log_metric(metric.replace("@", "_at_"), value)
        mlflow.log_artifact(RESULTS_PATH)

    print(f"Logged to MLflow experiment '{MLFLOW_EXPERIMENT}' (run: {args.run_name or 'unnamed'})")
    print("View with: mlflow ui --backend-store-uri sqlite:///mlflow.db")


def cmd_all(args: argparse.Namespace) -> None:
    """Run the full golden set in a single process (original behavior)."""
    golden = load_golden(GOLDEN_PATH)
    controller = RagController()
    predictions, skipped = run_predictions(controller, golden)

    with open(PREDICTIONS_PATH, "w") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")

    if skipped:
        print(f"Skipped {len(skipped)}/{len(golden)} queries after repeated failures:")
        for q in skipped:
            print(f"  - {q}")

    cmd_finish(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default=None,
        help="MLflow run name, e.g. baseline / hybrid-search / bge-small",
    )
    parser.add_argument(
        "--note", default=None, help="Free-text note describing what changed for this run"
    )
    parser.add_argument(
        "--start", type=int, default=None, help="Batch mode: golden row start index (inclusive)"
    )
    parser.add_argument(
        "--end", type=int, default=None, help="Batch mode: golden row end index (exclusive)"
    )
    parser.add_argument(
        "--finish",
        action="store_true",
        help="Score accumulated predictions and log the MLflow run, without running any queries",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="With --start/--end: clear any existing predictions file before this batch",
    )
    args = parser.parse_args()

    if args.finish:
        cmd_finish(args)
    elif args.start is not None or args.end is not None:
        if args.fresh and os.path.exists(PREDICTIONS_PATH):
            os.remove(PREDICTIONS_PATH)
        cmd_batch(args)
    else:
        cmd_all(args)
