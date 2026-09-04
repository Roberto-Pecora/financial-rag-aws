# Financial RAG on AWS

A retrieval-first financial research assistant for SEC filings, earnings reports, messy PDFs, tables, and scanned documents.

The system combines corpus-trained dense retrieval, BM25, Reciprocal Rank Fusion (RRF), and cross-encoder reranking. It answers only from retrieved evidence, cites supporting chunks, and abstains when the evidence is insufficient.

It is designed for research on published financial data, not investment advice.

> The goal is not to use the best model on a public leaderboard. It is to measure and improve retrieval on the documents that matter.

```text
PDF / SEC HTML / scans
        │
        ▼
Text extraction · table linearisation · OCR
        │
        ▼
Canonical chunks with stable content-hash IDs
        │
        ▼
S3 corpus + OpenSearch index
        │
        ▼
Dense k-NN + BM25 → RRF → cross-encoder reranker
        │
        ▼
Answer with citations, or abstain
```

## Why retrieval-first?

Embedding benchmarks do not guarantee performance on a specific financial corpus. In the predecessor prototype, `bge-small` underperformed `all-MiniLM-L6-v2` on dense financial tables, while hybrid dense-plus-BM25 retrieval delivered the largest improvement.

This project treats retrieval as a measurable engineering problem:

- Ingest the messy documents analysts actually use
- Create training pairs from the corpus itself
- Fine-tune the embedding model and reranker
- Evaluate every change against a fixed golden set
- Separate retrieval metrics from LLM answer quality

## Features

| Area | What it does |
|---|---|
| Messy ingestion | Extracts text PDFs with `pymupdf`, linearises tables with `pdfplumber`, runs Textract OCR for scans, and normalises SEC HTML into one chunk schema |
| Idempotent batch jobs | Uses content-hash IDs, so re-ingesting unchanged documents is a no-op and documents can be processed independently |
| Trained retrieval | Fine-tunes `bge-small` with `MultipleNegativesRankingLoss`, using synthetic queries, positive pairs, and BM25-mined hard negatives |
| Hybrid search | Runs dense k-NN and BM25 in one OpenSearch index, then combines results with RRF |
| Reranking | Applies a corpus-trained cross-encoder to the fused candidate list |
| graphRAG | Extracts issuer, instrument, and covenant relationships into a property graph for structured credit questions |
| Grounded answers | The actor must cite retrieved evidence or abstain; an optional critic can apply a further risk gate |
| Deterministic evaluation | Measures `recall@k`, `precision@k`, `MRR@k`, and `nDCG@k` from content-based relevance labels |
| Cost control | Keeps persistent AWS services small, gates Textract, and provisions or tears down infrastructure with Terraform |

## Architecture

The application, not the LLM, controls retrieval, reranking, citations, and answer gating.

| Component | Responsibility |
|---|---|
| `frag.sources` | PDFs, tables, OCR, and SEC HTML to canonical chunks |
| `frag.jobs.ingest_manifest` | Idempotent manifest-driven batch ingestion |
| `frag.aws.s3_store` | Corpus, model artifacts, and evaluation data in S3 |
| `frag.rag.store_opensearch` | Dense k-NN, BM25, and RRF hybrid retrieval |
| `frag.rag.reranker` | Cross-encoder reranking |
| `frag.kg` | Entity and covenant extraction, property graph, and graphRAG |
| `frag.train` | Pair mining and retrieval-model training |
| `frag.rag.{actor,critic,controller}` | Grounded answer generation, optional gate, and orchestration |
| `frag.eval` | Exact in-memory evaluation and ablation runs |
| `infra/` | Terraform for S3, OpenSearch, and least-privilege IAM |

All retrieval backends implement the same contract:

```python
search(query) -> [{"text": ..., "metadata": ..., "score": ...}]
```

This allows the application to switch between OpenSearch, a local store, or graphRAG without changing downstream answer-generation logic.

## Quick start

### Install

```bash
uv sync --frozen --extra dev --extra eval
cp .env.example .env
```

Install optional dependencies when needed:

```bash
uv sync --frozen --extra data --extra kg --extra efficient
```

### Provision AWS

```bash
make infra-plan
make infra-up
cd infra && terraform output
```

The AWS data plane consists of:

- S3 for the corpus, training artifacts, and evaluation outputs
- One small OpenSearch node for hybrid retrieval
- AWS Textract only when OCR is required

Add the Terraform outputs to `.env`, then tear down the infrastructure when finished:

```bash
make infra-down
```

### Build and index a corpus

```bash
python scripts/build_corpus.py \
  --out data/corpus.jsonl \
  --s3-key corpus/corpus.jsonl
```

```bash
STORE_BACKEND=opensearch python -c "
from frag.rag.controller import RagController
import json

controller = RagController()
records = [json.loads(line) for line in open('data/corpus.jsonl')]
print(controller.ingest(records))
"
```

### Query the API

```bash
export OPENROUTER_API_KEY=...
uvicorn frag.api.main:app --port 8000
```

```bash
curl -s localhost:8000/v1/query \
  -d '{"query":"What changed in revenue and liquidity?","top_k":5}'
```

## Train on the corpus

No manually labelled relevance dataset is required. The training pipeline generates synthetic questions and mines BM25 hard negatives: lexically similar chunks that are useful retrieval-stage distractors.

```bash
python scripts/mine_pairs.py \
  --corpus data/corpus.jsonl \
  --out data/pairs.jsonl

python scripts/finetune_embedding.py \
  --pairs data/pairs.jsonl \
  --out artifacts/bge-ft

python scripts/train_reranker.py \
  --pairs data/pairs.jsonl \
  --out artifacts/reranker
```

Training can run locally or on a Colab GPU. Model artifacts can be stored in S3.

## graphRAG

Some financial questions require structured relationships rather than only a relevant paragraph. For example, restricted-payments capacity may require a path such as:

```text
Issuer → Instrument → Covenant → Exception → Capacity
```

The graph combines:

- Deterministic entity extraction for stable entities such as issuers, sponsors, and tickers
- Schema-guided LLM extraction for covenants, instruments, and financial relationships

Its retriever returns the same `{text, metadata, score}` contract as standard chunk retrieval.

```bash
python scripts/build_graph.py \
  --corpus data/corpus.jsonl \
  --out data/graph.json \
  --gazetteer data/gazetteer.json
```

## Evaluation

Evaluation is deterministic and retrieval-focused.

A chunk counts as relevant when it contains the facts asserted in the golden answer, rather than when it matches a particular chunk ID. This makes the evaluation robust to changes in chunking strategy.

The exact in-memory evaluator reports:

- `recall@k`
- `precision@k`
- `MRR@k`
- `nDCG@k`

Production OpenSearch uses HNSW, where `ef_search` controls the recall-versus-latency trade-off. Evaluation uses brute-force exact search, so reported retrieval accuracy excludes ANN approximation loss.

### Ablations and benchmarks

The ablation matrix compares base and fine-tuned embeddings, both with and without reranking. Each configuration reports mean ± standard deviation over repeated runs, p50/p95 retrieval latency, and estimated cost in MLflow.

```bash
python scripts/run_ablation.py \
  --corpus data/corpus.jsonl \
  --base BAAI/bge-small-en-v1.5 \
  --finetuned artifacts/bge-ft \
  --reranker artifacts/reranker \
  --repeats 5
```

```bash
python scripts/run_index_ablation.py \
  --corpus data/corpus.jsonl \
  --bit-width 4
```

```bash
python scripts/bench_end_to_end.py \
  --golden data/golden_seed.csv \
  --repeats 3
```

Quantised 2-bit and 4-bit retrieval is treated as an optional first-stage recall mechanism, followed by full-precision rescoring. Its memory, latency, and recall impact are measured on the corpus rather than assumed.

## Starting baseline

The repository includes the evaluation harness and ablation scripts, but it does not claim final fine-tuning or reranking results before training has been run.

The predecessor project reported the following starting point on a similar corpus:

| Configuration | Recall@10 | nDCG@10 |
|---|---:|---:|
| Dense, base `bge-small` | 0.39 | 0.15 |
| Dense, `all-MiniLM-L6-v2` | 0.52 | 0.24 |
| Hybrid dense + BM25 with RRF | 0.68 | 0.34 |

Hybrid retrieval was the decisive initial improvement. Corpus-specific fine-tuning and cross-encoder reranking are the next levers, measured through the ablation runner.

## Development

```bash
make test
make check
```

Tests are hermetic: stores, LLMs, and AWS clients are stubbed, so the suite runs without AWS, a GPU, or API keys.

## Scope and limitations

- The critic is an optional weak LLM-judge signal, not a retrieval metric.
- The system is a low-cost, extensible research platform rather than a production-volume document-processing service.
- OpenSearch and Textract are billable paths. Review Terraform plans and run `make infra-down` after use.

## License

MIT
