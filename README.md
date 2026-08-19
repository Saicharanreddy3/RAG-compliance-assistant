# RAG-Powered Compliance Document Assistant

Grounded question answering over compliance documents. Every answer carries
source citations back to the exact clause it came from, and the system abstains
rather than guess when the documents do not contain the answer.

The point of the project is not that it does retrieval-augmented generation —
that part is well-trodden. The point is the parts that decide whether a RAG
system is trustworthy enough to put in front of a compliance team:

- **Citations are validated, not trusted.** Markers emitted by the model are
  resolved back to chunks that were actually retrieved. Markers that point
  nowhere are stripped, and an answer with no surviving citation is returned
  with `grounded: false` instead of being passed off as sourced.
- **Abstention is a first-class behaviour.** When the best retrieved chunk
  scores below a threshold, the system refuses before it ever calls the model.
  Three cases in the regression set exist purely to check that it refuses.
- **A frozen regression set gates every change.** Prompt, chunking, and
  retrieval changes are scored against a fixed question/answer file in CI, with
  hard thresholds on retrieval hit rate, MRR, and citation precision.
- **It runs with no credentials.** Embeddings, the LLM, and the vector store are
  each behind an adapter, and the defaults are offline. `git clone && make ingest && make eval`
  works on a clean machine with no AWS account and no API keys.

```bash
git clone https://github.com/Saicharanreddy3/rag-compliance-assistant
cd rag-compliance-assistant
pip install -r requirements-dev.txt
make ingest
make ask Q="How long must system audit logs be retained?"
```

```
System audit logs, including authentication events, privileged access events, and configuration
changes, must be retained for a minimum of eighteen (18) months. [1]

Sources:
  [1] Data Retention and Disposal Policy — 3.2 Audit Logs  (score 0.4829)

(2.17 ms, answered)
```

---

## Contents

- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API](#api)
- [Evaluation](#evaluation)
- [Design decisions](#design-decisions)
- [Retrieval tuning](#retrieval-tuning)
- [Known limitations](#known-limitations)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project layout](#project-layout)

---

## Architecture

```
Documents (.md/.txt/.pdf/.json)
      │
      ▼
  loader.py ──────► chunker.py ──────► section-aware chunks
                    (headings kept as        │
                     citation metadata)      │
                                             ▼
                                      embeddings.py
                                    (hash | bedrock | openai)
                                             │
                                             ▼
                                     vector_store.py
                                    (local JSON | pinecone)
                                             │
  question ──► retriever.py ◄────────────────┘
               dense search + IDF-weighted lexical rerank
                    │
                    ▼
              grounding check ──► abstain if best score < threshold
                    │
                    ▼
              pipeline.py ──► llm.py (echo | bedrock | openai)
                    │
                    ▼
            citation validation ──► Answer{answer, citations[], grounded}
                    │
                    ▼
              FastAPI /ask
```

Three seams are provider-agnostic — embeddings, the vector store, and the LLM.
Each is a `Protocol` with a `build_*` factory driven by environment variables, so
switching from the offline defaults to Bedrock + Pinecone + GPT-4o is
configuration, not a code change.

---

## Quick start

### Offline (default — no credentials)

```bash
pip install -r requirements-dev.txt

make ingest                                     # load and index data/documents
make ask Q="What is the minimum password length?"
make eval                                       # run the regression suite
make test                                       # 60 unit and contract tests
make serve                                      # http://localhost:8000/docs
```

### With real providers

```bash
cp .env.example .env
```

```bash
EMBEDDING_PROVIDER=bedrock
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
VECTOR_BACKEND=pinecone
PINECONE_API_KEY=...
OPENAI_API_KEY=...
```

```bash
pip install boto3 openai pinecone
make ingest && make eval
```

### Docker

```bash
docker build -t rag-compliance-assistant .
docker run -p 8000:8000 rag-compliance-assistant
```

The image builds the index at build time, so the container starts ready to
serve and `/health` reports a non-zero vector count immediately.

---

## Configuration

Everything is environment-driven; see [`.env.example`](.env.example) for the
full list. The ones that change behaviour most:

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_PROVIDER` | `hash` | `hash` (offline), `bedrock`, `openai` |
| `LLM_PROVIDER` | `echo` | `echo` (offline), `bedrock`, `openai` |
| `VECTOR_BACKEND` | `faiss` | Local JSON-backed store, or `pinecone` |
| `RETRIEVAL_TOP_K` | `5` | Chunks passed to the model |
| `RETRIEVAL_CANDIDATE_K` | `20` | Candidates fetched before reranking |
| `RETRIEVAL_MIN_SCORE` | `0.15` | Chunks below this never reach the model |
| `GROUNDING_THRESHOLD` | `0.2` | Below this, the system abstains |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `900` / `150` | Chunking geometry |

---

## API

### `POST /ask`

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long must audit logs be retained?"}'
```

```json
{
  "question": "How long must audit logs be retained?",
  "answer": "System audit logs must be retained for a minimum of eighteen (18) months. [1]",
  "grounded": true,
  "citations": [
    {
      "marker": 1,
      "chunk_id": "data_retention_policy::0004",
      "doc_id": "data_retention_policy",
      "label": "Data Retention and Disposal Policy — 3.2 Audit Logs",
      "quote": "System audit logs, including authentication events...",
      "score": 0.4829
    }
  ],
  "retrieved": [
    {
      "chunk_id": "data_retention_policy::0004",
      "doc_id": "data_retention_policy",
      "section": "3.2 Audit Logs",
      "score": 0.4829,
      "preview": "System audit logs, including authentication events..."
    }
  ],
  "metadata": {
    "latency_ms": 2.17,
    "reason": "answered",
    "prompt_version": "2024-11-08.v3",
    "model": "echo-extractive",
    "retrieved_count": 3,
    "top_score": 0.4829
  }
}
```

`retrieved` is returned alongside `citations` deliberately: it lets a reviewer
see what the model was shown, not just what it chose to cite. That distinction
is most of what makes a wrong answer debuggable.

### Other endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Vector count, active providers, prompt version |
| `POST /ingest` | Index a directory (`{"directory": "...", "reset": true}`) |
| `GET /docs` | OpenAPI UI |

Error contract: `409` if the index is empty, `404` for a missing ingest
directory, `422` for a malformed question.

---

## Evaluation

[`evals/regression_set.json`](evals/regression_set.json) holds 17 frozen cases —
14 factual lookups across three policy documents and 3 abstention cases whose
answers are deliberately absent from the corpus.

```bash
make eval
```

```
====================================================================
Regression set 2024-11-08.v2 | prompt 2024-11-08.v3
echo/echo-extractive | embeddings=hash | backend=faiss
====================================================================
  Cases              16/17 passed (94.1%)
  Retrieval hit rate 100.0%
  MRR                0.964
  Keyword recall     92.9%
  Citation precision 100.0%
  Abstention rate    100.0%
  Latency p50/p95    2.8 / 2.9 ms

All thresholds met.
```

### Metrics

| Metric | Meaning | CI gate |
|---|---|---|
| `pass_rate` | Cases with no failed assertion | ≥ 0.90 |
| `retrieval_hit_rate` | Expected document appears in top-k | ≥ 0.95 |
| `mrr` | Mean reciprocal rank of the expected document | ≥ 0.75 |
| `citation_precision` | Citations resolving to actually-retrieved chunks | ≥ 0.95 |
| `abstention_rate` | Unanswerable questions correctly refused | reported |
| `keyword_recall` | Answers containing the expected fact | reported |

Every metric is deterministic — no LLM judge anywhere in the gate. That is a
deliberate constraint: with a judge in the loop, a score movement is ambiguous
between "the system changed" and "the judge drifted." Here a movement always
means the system changed. An LLM judge is a reasonable addition for fluency
scoring, but it belongs beside these gates, not inside them.

The scoring code has its own tests ([`tests/test_evals.py`](tests/test_evals.py)),
including structural validation of the dataset itself — a silently broken metric
is worse than no metric.

### Changing the regression set

Add cases freely. When editing or removing an existing case, bump
`dataset_version` so historical scores stay interpretable; otherwise the same
version number refers to two different question sets and score history becomes
meaningless. `PROMPT_VERSION` in
[`prompts.py`](src/rag_assistant/generation/prompts.py) serves the same purpose
for prompt changes, and both are stamped into every report.

---

## Design decisions

### Citation validation

Models cite confidently and sometimes wrongly. `_resolve_citations` maps every
`[n]` marker back to the chunk at that position, drops markers pointing past the
end of the retrieved set, and collapses duplicates:

```python
result = pipeline.answer("How long must audit logs be retained?")
# LLM returned: "...eighteen months [1] and reviewed yearly [99]."
result.answer     # "...eighteen months [1] and reviewed yearly."
result.grounded   # True — [1] resolved
```

An answer whose markers all fail to resolve comes back with `grounded: false`
and an empty citation list, so a caller can surface it rather than display it as
sourced.

### Abstention before generation

The grounding check runs on retrieval scores, before the model is called. If
nothing clears the threshold, the model never sees the question. This is cheaper
than generating and post-filtering, and it removes the chance that a fluent
model talks its way past a weak retrieval result.

### Section-aware chunking

Compliance documents are densely sectioned, and the heading is usually the most
useful citation metadata available. The chunker tracks the nearest preceding
heading — markdown (`## 3.2 Audit Logs`), numbered (`4.2 Data Retention`), and
keyword (`ARTICLE 12`) forms — and attaches it to every chunk, which is what
produces `Data Retention and Disposal Policy — 3.2 Audit Logs` rather than a
bare filename.

Chunking is deterministic: identical input yields identical chunk ids. This is
what makes the eval reproducible across runs.

### Provider adapters

Isolating provider SDKs behind `Embedder`, `VectorStore`, and `LLMClient`
protocols means a vendor change is a config change. It also means the test suite
and eval harness run against deterministic offline implementations, so CI needs
no secrets and never bills anyone.

---

## Retrieval tuning

Retrieval is a 50/50 blend of dense cosine similarity and an IDF-weighted
lexical score. Both halves and both weights came out of specific regression
failures, and the history is worth recording:

| Change | Pass rate |
|---|---|
| Initial (dense 0.75 / lexical 0.25, no stemming) | 11.8% |
| Fixed `doc_id` slug mismatch | 82.3% |
| Rebalanced to 50/50, added plural collapsing | 88.2% |
| Added IDF weighting + contextual embeddings | **94.1%** |

Three of those were real bugs, and the eval is what surfaced them:

1. **`doc_id` slug mismatch.** The loader slugified `access_control_standard`
   into `access-control-standard`, so every eval case reported a retrieval miss
   while the system was in fact answering correctly. Retrieval hit rate read
   0% while the answers on screen were right — a good reminder that a metric
   disagreeing violently with observed behaviour usually indicts the metric
   first.

2. **No plural collapsing.** A query for `password` could not match text saying
   `passwords`. Compliance text alternates between singular definitions and
   plural obligations constantly, so this was costing real matches.

3. **Boilerplate chunks outranking specific ones.** Unweighted token overlap
   scores `minimum` and `password` identically, so the "Purpose and Scope"
   preamble — which contains every generic policy word — beat the clause that
   actually answered the question. Weighting matched terms by inverse document
   frequency suppresses this. It is BM25's central idea without the full scoring
   function or an extra dependency.

Chunks are also embedded as `"<title> - <section>\n\n<text>"` rather than bare
text, so the discriminating words in a heading contribute to the match.

Backends that cannot cheaply enumerate their contents (Pinecone) return an empty
IDF table, and the rerank degrades to unweighted overlap rather than failing.

---

## Known limitations

**One regression case fails on the default offline configuration**, and it is
left failing on purpose.

`password-length` retrieves the correct chunk at rank 2, but the `echo` stub —
which picks a sentence by term overlap rather than reading — takes rank 1. The
hashed bag-of-words embedder simply cannot tell that "password length
requirement" is closer to an authentication clause than to a document's purpose
statement. A trained embedding model ranks it first, and the case passes with
`EMBEDDING_PROVIDER=bedrock` or `openai`.

It would be easy to tune the case green. It is more useful as a standing marker
of where the offline configuration's ceiling actually is, and the 90% gate is
set with it in mind.

Other constraints worth stating plainly:

- The local vector store does a brute-force scan. Fine for hundreds or low
  thousands of chunks; use Pinecone beyond that.
- `HashingEmbedder` has no semantic understanding — it will not match
  paraphrases that share no vocabulary. It exists for determinism and offline
  CI, not quality.
- There is no reranker model, query expansion, or hybrid BM25 index. All three
  are sensible next steps.
- Retrieval is single-hop; questions requiring synthesis across several
  documents are not handled well.
- No authentication on the API. Add it before exposing this anywhere real.

---

## Testing

```bash
make test    # 60 tests
make lint    # ruff, clean
make ci      # lint + test + ingest + eval, exactly what CI runs
```

| File | Covers |
|---|---|
| `tests/test_ingestion.py` | Heading detection, chunk determinism, size bounds, loader edge cases |
| `tests/test_retrieval.py` | Ranking, IDF weighting, stemming, store persistence, idempotent upsert |
| `tests/test_pipeline.py` | Citation resolution, hallucinated-marker stripping, abstention paths, grounding flags |
| `tests/test_api.py` | Response contract, status codes, validation, OpenAPI generation |
| `tests/test_evals.py` | The scoring functions themselves, plus dataset structural validation |

Two tests are regression locks on bugs found during development:
`test_embedding_of_empty_text_does_not_crash` (signed hash collisions cancelling
to zero, hitting `log(0)`) and
`test_chunker_respects_max_size_with_tolerance_for_overlap` (text with no
sentence terminators — tables, bullet lists — never splitting and producing
oversized chunks).

CI runs on Python 3.10, 3.11, and 3.12, then builds the Docker image and smoke
tests a live container end to end.

---

## Deployment

The FastAPI app is transport-agnostic. It runs under uvicorn locally, as a
container on EKS, or in Lambda behind API Gateway via Mangum:

```python
from mangum import Mangum
from rag_assistant.api.main import app

handler = Mangum(app)
```

The container runs as a non-root user and ships a `HEALTHCHECK` against
`/health`, which reports vector count and active providers — enough for a
readiness probe to distinguish "process up" from "actually able to answer."

---

## Project layout

```
rag-compliance-assistant/
├── src/rag_assistant/
│   ├── config.py              Environment-driven settings
│   ├── models.py              Document, Chunk, ScoredChunk, Citation, Answer
│   ├── cli.py                 ingest / ask / eval
│   ├── ingestion/
│   │   ├── loader.py          .md, .txt, .pdf, .json
│   │   └── chunker.py         Section-aware, deterministic
│   ├── retrieval/
│   │   ├── embeddings.py      hash | bedrock | openai
│   │   ├── vector_store.py    local | pinecone
│   │   └── retriever.py       Dense + IDF-weighted lexical rerank
│   ├── generation/
│   │   ├── prompts.py         Versioned templates
│   │   ├── llm.py             echo | bedrock | openai
│   │   └── pipeline.py        Orchestration + citation validation
│   └── api/
│       ├── main.py            FastAPI app
│       └── schemas.py         Request/response contract
├── evals/
│   ├── regression_set.json    17 frozen cases
│   ├── metrics.py             Deterministic scoring
│   └── run_eval.py            Runner + threshold gate
├── tests/                     60 tests
├── data/documents/            3 sample compliance policies
├── .github/workflows/ci.yml   Lint, test, eval gate, image smoke test
├── Dockerfile
└── Makefile
```

---

## License

MIT

