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

Embedding benchmarks do not guarantee performance on a specific financial corpus. Measured on FinanceBench, `bge-base` beat both the smaller `bge-small` and the larger `bge-large`: a bigger model did not help, and the mid-sized model won. In the predecessor prototype, `bge-small` underperformed `all-MiniLM-L6-v2` on dense financial tables, while hybrid dense-plus-BM25 retrieval delivered the largest improvement.

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
| Trained retrieval | Fine-tunes `bge-base` with `MultipleNegativesRankingLoss`, using synthetic queries, positive pairs, and BM25-mined hard negatives |
| Hybrid search | Runs dense k-NN and BM25 in one OpenSearch index, then combines results with RRF |
| Reranking | Applies a corpus-trained cross-encoder to the fused candidate list |
| Entity-aware retrieval | Derives a company filter from the query, so a question about one issuer does not drift onto similarly-worded passages about another |
| graphRAG | Extracts issuer, instrument, and covenant relationships into a property graph for structured credit questions |
| Corrective retrieval | Grades retrieved passages against the query and, if too few are relevant, rewrites the query and re-retrieves (CRAG-style, off by default) |
| Agentic tool-loop | Routes multi-hop questions to a tool-using loop over retrieval, graph lookup, and calculation, under turn and tool-call budgets |
| Grounded answers | The actor must cite retrieved evidence or abstain; an optional critic can apply a further risk gate |
| Answer grounding | Maps each fact in the answer back to the passage that supports it, flagging any claim no retrieved passage grounds |
| Injection defence | Wraps retrieved passages as untrusted data and screens input and output, so a poisoned document cannot redirect the model or fabricate a source |
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
| `frag.rag.grader` | Corrective retrieval: passage grading and query rewrite |
| `frag.agent.{loop,tools}` | Tool-using loop over retrieval, graph lookup, and calculation, with budgets and a reasoning trace |
| `frag.agent.{router,orchestrator}` | Routes a question to deterministic RAG or the tool-loop, rewrites follow-ups, and critic-gates the agent |
| `frag.agent.guardrails` | Input and output screens for injection defence |
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

## Corrective retrieval

When a query retrieves weak evidence, one more retrieval round is cheaper than a wrong answer. An LLM grades each retrieved passage against the query, and if too few are relevant, the query is rewritten (expanding abbreviations and adding the specific financial term a filing is likely to use) and re-retrieved. The grader fails open, so a parse error keeps evidence rather than starving the actor. It is off by default (`CORRECTIVE`) and adds an LLM call per passage when on.

## Agentic path

Most questions are a single lookup and go straight to deterministic RAG. A router sends multi-hop questions — comparisons, thresholds, questions that need retrieval and the graph together — to a tool-using loop instead. The loop calls retrieval, graph lookup, and a whitelisted calculator over provider-native function-calling, under turn and tool-call budgets, and records a reasoning trace. Its final answer passes through the same critic risk gate as the RAG path. Routing is deterministic by default; an LLM classifier sits behind `ROUTER_LLM`.

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
  --base BAAI/bge-base-en-v1.5 \
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

### Embedding model size

Base embeddings were compared on FinanceBench (Colab GPU) to test the assumption that a larger model retrieves better.

| Model | Params | recall@1 | recall@5 | recall@10 |
|---|---:|---:|---:|---:|
| `bge-small` | 33M | 0.050 | 0.107 | 0.227 |
| **`bge-base`** | 109M | **0.084** | **0.175** | **0.245** |
| `bge-large` | 335M | 0.060 | 0.109 | 0.218 |

The mid-sized model wins: `bge-large` does not beat `bge-base` on this corpus. The live system uses `bge-base` as its base embedding.

## Starting baseline

The repository includes the evaluation harness and ablation scripts, but it does not claim final fine-tuning or reranking results before training has been run.

The predecessor project reported the following starting point on a similar corpus:

| Configuration | Recall@10 | nDCG@10 |
|---|---:|---:|
| Dense, base `bge-small` | 0.39 | 0.15 |
| Dense, `all-MiniLM-L6-v2` | 0.52 | 0.24 |
| Hybrid dense + BM25 with RRF | 0.68 | 0.34 |

Hybrid retrieval was the decisive initial improvement. Corpus-specific fine-tuning and cross-encoder reranking are the next levers, measured through the ablation runner.

## Prompt-injection defence

A RAG system over ingested documents is exposed to indirect prompt injection: the untrusted text is the retrieved passage itself. The defence is layered and mostly deterministic, so most of it runs without an API key.

- Each retrieved passage is wrapped as untrusted data, with the delimiters sanitised so a document cannot forge or escape the markers. The actor and critic are told to treat the wrapped text as data, never as instructions.
- An input screen blocks override and jailbreak attempts before any LLM call.
- An output screen requires the answer's citations to be a subset of the retrieved labels, so a fabricated source downgrades the answer to an abstention.

The defence is measured, not assumed: `scripts/eval_security.py` runs an authored set of poisoned-document and jailbreak cases and reports a defence rate, with the deterministic cases passing with no key.

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
