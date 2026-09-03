# Financial RAG on AWS

A financial retrieval-augmented research assistant over **messy source documents**
— SEC filings, earnings PDFs, scanned reports — built around **trained retrieval**
rather than an off-the-shelf embedding. Hybrid dense + BM25 search runs on a
single free-tier **AWS OpenSearch** node; the embedding model and a cross-encoder
reranker are **finetuned on the corpus itself**, and every configuration is scored
by a deterministic, content-based evaluation harness.

It is a rebuild of the [actor-critic financial RAG](https://github.com/Hydaspex/actor-critic-financial-rag)
prototype, keeping that project's eval harness and answer gate but closing its
three gaps: no messy/PDF ingestion, no *trained* IR models, and no cloud.

```
PDF / SEC HTML / scan        manifest (idempotent)        AWS data plane
        │                          │                           │
   text · tables · OCR ──► canonical chunk records ──► S3 corpus + OpenSearch
        │                                                      │  dense k-NN + BM25
   pymupdf / pdfplumber / Textract                             ▼
                                                        RRF fusion ──► cross-encoder rerank
                                                                              │
                                                                actor (cite or abstain)
                                                                      · optional critic gate
```

The system researches published financial data and must abstain when the
retrieved evidence is insufficient. It is not investment advice.

## Why this exists

An embedding that tops a public retrieval leaderboard does not transfer
unconditionally to a specific corpus — the predecessor project found a "stronger"
model (`bge-small`) *underperforming* a weaker one on dense financial tables. The
lesson is that retrieval quality on your own documents is an engineering result
you earn, not a model you download. This project treats retrieval as the primary
problem: it ingests the messy documents real desks actually hold, mines training
data from the corpus, finetunes both retrieval-stage models, and measures each
change against a fixed golden set instead of trusting a leaderboard.

## What it demonstrates

- **Messy ingestion:** text-layer PDFs (`pymupdf`), financial tables linearised
  as `label: value` pairs (`pdfplumber`), and scanned pages via **AWS Textract**,
  all normalised into one canonical chunk schema with a per-path tag.
- **Trained retrieval:** `bge-small` finetuned with `MultipleNegativesRankingLoss`,
  and a cross-encoder reranker, both trained on **synthetic queries + mined hard
  negatives** — no human labels.
- **Hybrid search on managed AWS:** dense k-NN and BM25 in one OpenSearch index,
  fused with Reciprocal Rank Fusion, provisioned and torn down by Terraform.
- **Idempotent, batchable ingestion:** content-hash chunk ids make re-ingest a
  no-op; stateless per-document functions parallelise without coordination.
- **Data-centric evaluation:** content-based relevance (a chunk is a hit if it
  contains the reference answer's facts), an ablation matrix over
  {base vs finetuned} × {rerank off vs on}, tracked in MLflow.
- **Cost discipline:** the whole AWS footprint is free-tier, Textract is spend-gated,
  and teardown is one command.

## Architecture

The model never queries AWS directly; the application owns retrieval, reranking,
and the answer gate.

| Component | Responsibility |
|---|---|
| `frag.sources` (pdf_text / pdf_tables / ocr_textract) | Messy documents → canonical `{id, text, metadata}` records |
| `frag.jobs.ingest_manifest` | Idempotent, stateless batch ingestion → JSONL |
| `frag.aws.s3_store` | S3 data lake I/O (corpus, artifacts, eval) |
| `frag.rag.store_opensearch` | Managed hybrid store; dense k-NN + BM25 fused with RRF |
| `frag.rag.reranker` | Cross-encoder reranking stage (RERANK=on) |
| `frag.train` | Pair mining, embedding finetune, reranker training |
| `frag.rag.{actor,critic,controller}` | Grounded answer, optional risk gate, orchestration |
| `frag.eval` | Content-based harness, in-memory eval, ablation matrix |
| `infra/` | Terraform: S3 + OpenSearch + least-privilege IAM |

The retrieval `search()` contract (`{text, metadata, score}`) and the LLM
`generate()` contract are held fixed across backends, so moving from Qdrant to
OpenSearch and from a local model to OpenRouter changed no downstream code.

## Quick start

### 1. Install

```bash
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[dev,eval]"
cp .env.example .env   # fill in keys as needed
```

### 2. Stand up the AWS data plane (free-tier)

```bash
make infra-plan        # review the diff first
make infra-up          # S3 + one t3.small.search OpenSearch node (~12 min)
cd infra && terraform output   # -> S3_BUCKET, OPENSEARCH_ENDPOINT into .env
```

OpenSearch bills by the hour; `make infra-down` when finished.

### 3. Build a corpus and index it

```bash
python scripts/build_corpus.py --out data/corpus.jsonl --s3-key corpus/corpus.jsonl
STORE_BACKEND=opensearch python -c "from frag.rag.controller import RagController; import json; \
  c=RagController(); print(c.ingest([json.loads(l) for l in open('data/corpus.jsonl')]))"
```

### 4. Ask a question

```bash
export OPENROUTER_API_KEY=...        # actor/critic run through OpenRouter
uvicorn frag.api.main:app --port 8000
curl -s localhost:8000/v1/query -d '{"query":"What changed in revenue and liquidity?","top_k":5}'
```

## Training the IR models

Both trained models learn from the corpus, no labels. Run the mining locally,
the fits on a Colab GPU (`notebooks/colab_train.ipynb`), and push the artifacts
to S3.

```bash
python scripts/mine_pairs.py --corpus data/corpus.jsonl --out data/pairs.jsonl   # OpenRouter
python scripts/finetune_embedding.py --pairs data/pairs.jsonl --out artifacts/bge-ft
python scripts/train_reranker.py    --pairs data/pairs.jsonl --out artifacts/reranker
```

Hard negatives are mined with **BM25 on purpose** — lexically close distractors
are exactly what a dense model ranks just below the answer, so mining them targets
the failure the finetune is meant to fix.

## Evaluation

Relevance is **content-based**: a retrieved chunk counts as a hit if its text
contains the numeric facts asserted in the golden row's reference answer, so the
golden set is independent of chunk boundaries and survives re-chunking. Metrics
are deterministic (`recall@k`, `precision@k`, `MRR@k`, `nDCG@k`); the actor/critic
are not in the retrieval-metric loop.

The ablation matrix sweeps the two trained models against their baselines:

```bash
python scripts/run_ablation.py --corpus data/corpus.jsonl \
    --base BAAI/bge-small-en-v1.5 --finetuned artifacts/bge-ft --reranker artifacts/reranker
```

### Results

The harness, ablation runner, and MLflow tracking are in place; the headline
numbers are produced by the command above once the models are trained, and are
not reproduced here as invented figures. The starting point they build on is the
predecessor project's documented baseline on the same style of corpus:

| Configuration | recall@10 | nDCG@10 |
|---|--:|--:|
| dense, base `bge-small` | 0.39 | 0.15 |
| dense, `all-MiniLM-L6-v2` | 0.52 | 0.24 |
| **hybrid (dense + BM25, RRF)** | **0.68** | **0.34** |

*(From the [actor-critic prototype](https://github.com/Hydaspex/actor-critic-financial-rag);
hybrid was the decisive lever there. This project's finetune and reranker are the
next two levers, measured by the ablation runner above.)*

The design keeps two honest caveats visible rather than tuning them away: the
critic score is a weak LLM-judge signal and is excluded from the retrieval
metrics; and "scale" here means the pipeline is operated on free-tier infra and
architected to extend, not run at production volume.

## AWS and cost

Data plane on AWS, compute plane off it: S3 (corpus, artifacts, eval), a single
free-tier OpenSearch node (hybrid search), and Textract (OCR) — while embedding
finetuning, reranker training, and model serving run on Colab / locally. This
keeps the whole footprint inside AWS Free Tier. See [`infra/README.md`](infra/README.md)
for the teardown runbook and guardrails.

## Development

```bash
make test        # 81 tests, hermetic — no AWS, no GPU, no API key
make check       # ruff lint + format
```

The tests stub the store, the LLM, and AWS clients, so the full suite runs
offline. Every billable path (Textract, OpenSearch) is gated or injectable.

## License

MIT.
