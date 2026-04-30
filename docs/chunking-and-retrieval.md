# MSCI Knowledge Graph RAG — Chunking, Graph Store & Retrieval

This document complements [parsing.md](parsing.md) (which covers HTML inspection, section detection, and the parsing pipeline). Here we summarize **how far the project has gotten**, the **chunking and tradeoffs** (with pointers to the parser), **Phase 3a (Neo4j + embeddings + structural graph)**, **how the retriever is tested**, and what is **deferred** (e.g. LLM-based entity graph).

---

## 1. Progress at a glance

| Phase | What | Status |
|-------|------|--------|
| 1 | Download 10-K / 10-Q HTML from SEC; `inspect.ipynb` for structure | Done (`download.py`, `data/raw/`) |
| 2 | Parse, clean, chunk to JSONL | Done (`parse.py`, `data/processed/*.jsonl`) — **details in [parsing.md](parsing.md)** |
| 3a | Embed chunks (Azure), load **Chunk** + **Filing** + **Section** in Neo4j, vector index, `NEXT` / `PART_OF` / `CONTAINS` / `HAS_FIRST_CHUNK` | Done (`store.py`, `retrieve.py`) |
| 3b | LLM entity/relationship extraction (`LLMGraphTransformer` → extra nodes) | **Deferred** — not required to validate chunk retrieval; useful later to compare answer quality or graph-augmented retrieval |
| 4 | End-user RAG query pipeline (full answer generation) | Not started |

**Chunk counts (reference):** ~593 `Chunk` nodes across both JSONL files (e.g. ~416 10-K, ~177 10-Q) — see [parsing.md#output](parsing.md#output) for exact numbers and section distribution.

---

## 2. Chunking: steps and tradeoffs (summary)

The **full** step list lives in [parsing.md#parsing-pipeline-parsey](parsing.md#parsing-pipeline-parsey) (DOM walk, table flattening, cleaning, metadata). The **decisions that matter for retrieval** are below.

| Decision | Choice | Rationale / tradeoff |
|----------|--------|----------------------|
| **Splitter** | `RecursiveCharacterTextSplitter` | Prefers breaks at paragraph / line / sentence before a hard cut; fewer mid-sentence splits than a fixed window. |
| **Chunk size** | 2000 characters | More context per vector than 1000; fewer total chunks and embedding cost; most tables fit in one chunk (large tables are still long — see 10% overlap and Item 7/8 in benchmarks). |
| **Overlap** | 200 characters (~10%) | Reduces the chance that a fact spanning two chunk boundaries appears only in fragments. Slight duplication across adjacent chunks. |
| **Separators** | `["\n\n", "\n", ". ", " "]` | Document-style hierarchy before falling back to spaces. |
| **Text vs table** | Same stream; `chunk_type` = `text` or `table` in metadata | One retrieval surface; `chunk_type` enables filters (e.g. financial tables) without a separate “table store.” |
| **Scope** | All sections, including all Items | Section labels in metadata; no hard “only Item 1–7” — matches full 10-K/10-Q use. |
| **What we did not do (yet)** | `LLMGraphTransformer` / APOC-heavy import | **Intentionally deferred** — adds cost, APOC dependency for LangChain’s default import path, and tuning (noise vs. recall). Revisit when you want to measure **accuracy with vs. without** an entity layer. **Optional** script: `graph_extract.py` (requires Neo4j APOC if using LangChain’s `Neo4jGraph.add_graph_documents` as written). |

**Output:** one JSON line per chunk in `data/processed/`, with `page_content` + metadata (`section`, `page_number`, `source`, `filing_type`, `period`, `chunk_type`, `chunk_index`, `filename`).

---

## 3. Phase 3a — graph store and embeddings (`store.py`)

**Goal:** Make chunks searchable in Neo4j and preserve **document order and hierarchy** for traversal, not only vector similarity.

**Process (high level):**
1. Load all JSONL records as LangChain `Document` objects.
2. **Embeddings:** Azure OpenAI (e.g. `text-embedding-3-small`) → property `embedding` on each **Chunk** node; vector index name e.g. `chunk_embeddings` on label `Chunk`, text in `page_content`.
3. **Structure:** `Filing` (by `filing_type` + `period`), `Section` (by section name + filing + period), then:
   - `Chunk`–`PART_OF`–>`Section`, `Filing`–`CONTAINS`–>`Section`, `Section`–`HAS_FIRST_CHUNK`–>`Chunk`, **Chunk**–`NEXT`–**Chunk** within the same section (order by `chunk_index`).

**Why this matters for retrieval**
- **Semantic search:** vector similarity on `page_content` (and optional **metadata pre-filters** on Chunk properties).
- **Graph:** walk `NEXT` for “read on,” or start from a `Section` / `Filing` for scoped queries.
- **No new chunk nodes in 3b** if you add entities later: entities would attach to existing chunks, not replace them.

**Infra note:** local Neo4j (e.g. Docker) with Bolt URL in `.env`; `store.py` is written for Neo4j 5+ Cypher (ordering in `collect` was adjusted for version compatibility in an earlier fix).

### 3.1 Form 13F investors graph (`load_13f_graph.py`)

**Data:** [../data/processed/form13f_msci_investors_by_manager.csv](../data/processed/form13f_msci_investors_by_manager.csv) (one row per 13F filer for MSCI; built by [../form13f_msci_extract.py](../form13f_msci_extract.py)).

**Model (Neo4j):**
- **`Company`** — one issuer node for MSCI (`ticker`, `cusip`, `name` from the 10-K Preamble in the graph when present, else `"MSCI Inc."`).
- **`Filing`** — reuses the existing 10-K **FY2025** node from `store.py`.
- **`[:FILED]`** — `(:Company)-[:FILED]->(:Filing)` (issuer **filed** that report).
- **`Manager`** — 13F filers (`cik`, `name`, `address`); not the stock issuer.
- **`Address`** — one per distinct `key` (full SEC line, or a per-CIK placeholder if missing); `raw`, optional US 2-letter `state` (parsed from the line when it matches `..., ST, 12345`).
- **`[:LOCATED_AT]`** — `(:Manager)-[:LOCATED_AT]->(:Address)`.
- **`[:OWNS_STOCK_IN]`** — `(:Manager)-[:OWNS_STOCK_IN {value, shares, as_of, issuer_cusip, source}]->(:Company)` with `source` = `sec_form_13f`.
- **Full-text** index `managerNameFulltext` on `Manager.name` (Neo4j 5).

**When to run:** after `store.py` has created the **Filing** `10-K` / `FY2025` (and ideally Preamble `Chunk` nodes for the canonical name). Re-running the loader is idempotent: `MERGE` / `SET` refresh from the current CSV.

**Command:** `uv run python src/load_13f_graph.py` (uses `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` from `.env`).

---

## 4. Retrieval quality — how it is calculated (`retrieve.py`)

We **do not** claim classical IR precision/recall with gold relevance labels. Instead we use a **practical two-part score** on a small benchmark.

### 4.1 Benchmark

- **10 questions** (`Q01`–`Q10`) with an **expected section** (string that should appear in the chunk’s `section` field).
- **Top k = 5** chunks per question.
- Modes (per question, see `BENCHMARK` in `retrieve.py`):
  - **Filtered semantic:** `Neo4jVector.similarity_search` with a **metadata filter** (e.g. `filing_type`, `section`, `chunk_type`) — reduces **semantic drift** to wrong sections.
  - **Cypher only:** e.g. **Q05** (ticker / exchange) via `page_content` `CONTAINS` in **Preamble** where keyword search is more reliable than a single short embedding query.
  - **Semantic without filter** only where a filter was not required by design (see script for the mix).

### 4.2 Metrics

1. **Section hit @ k**  
   For each question, **at least one** of the top-`k` chunks has `section` that **contains** the expected section string (case-insensitive). Reported as **hit rate over 10** (e.g. 10/10).

2. **LLM-as-judge (1–3)**  
   For **each** of the `k` chunks, the judge LLM (e.g. Azure `gpt-4.1`, temperature 0) scores **relevance to the question**:
   - 1 = not relevant, 2 = partial, 3 = highly relevant.  
   The script prints **per-chunk** scores and the **mean over chunks** (e.g. **avg ≈ 2.17/3** on a good run). This is a **relevance** signal, not factual correctness.

3. **Persistence**  
   Results (including per-question details) are written to **`../data/processed/benchmark_results.json`** for follow-up (no re-run of retrieval needed to re-read numbers).

**Limitations (intentional honesty)**
- Section hit is a **recall of “right section,”** not token-level or answer-level F1.
- Averages depend on phrasing; **Q03** (revenue table) can be noisy if `chunk_type=table` pulls multiple financial tables—tightening filter (e.g. `section` to Item 7) is a possible improvement.
- **Q08** (competition risk) can score lower on the judge if chunks are “generic risk” but still in 1A — a future **entity layer** is one way to help, if you return to Phase 3b.

---

## 5. Deferring LLM graph transform / Phase 3b

**Decision:** For now, **no production dependency** on `LLMGraphTransformer` (and no requirement to add APOC for LangChain’s graph import).

**Why this is fine**
- Phase 3a already validates **structural + vector + filtered + cypher** retrieval.
- You can add entity extraction **later** and A/B the **same benchmark** (or human eval) to quantify benefit vs. cost and noise.

**If you return to it**
- `graph_extract.py` is a **optional** sample (e.g. Item 1 + Item 1A only); LangChain’s default Neo4j write path **expects APOC**, or you replace it with a custom Cypher import.

---

## 5.1 Retrieval strategy comparison (`rag_retrieval_compare.py`)

This script runs **the same questions** through three ways of building **LLM context** and scores each run with a judge model (two separate **0–10** scores: **context relevance** and **answer quality**). It does **not** use APOC; window expansion is plain Cypher on `[:NEXT]`.

| Mode | What is retrieved | Role |
|------|-------------------|------|
| **vector_topk** | Top **5** similar `Chunk` texts (vector index), each with **citation** (`page_number`, `chunk_index`, `source_url` from metadata) | Baseline: broad semantic coverage. |
| **graph_window** | Vector **top-1** anchor, then **prev + center + next** along `[:NEXT]`, with structural header (Filing/Section/Chunk) and per-block citations (page + `source_url` from `Filing`) | Semantic anchor + local graph continuity. |
| **hybrid_topk_windows** | Vector top-5, **each** hit expanded to a `NEXT` window, deduplicated, concatenated; `retrieval_meta` lists all window anchors with page + source | Stronger recall + neighborhoods; can be long. |

**Scoring (LLM judge, 0–10 each, per mode per question):**

- **Context relevance** — does the supplied context contain enough **grounded** information to answer the question?
- **Answer quality** — is the answer **correct** and **supported by the context** (penalize hallucination)?

**How to run:** interactive (`uv run python src/rag_retrieval_compare.py`) — you choose how many questions, type each question, optional JSON filter (same idea as `retrieve.py`). Results are printed and saved to `data/processed/rag_retrieval_compare_results.json` (from project root) with a **`retrieval_meta`** object per run (e.g. `page_number`, `chunk_index`, `source_url` for each hit or window).

**Caveats:** the judge is another LLM (not a human); scores are **relative** within a session. The answer model is instructed to cite `page` and `source_url` from the context lines.

---

## 6. Commands (reference)

```bash
# Parse + chunk (produces JSONL)
uv run python src/parse.py

# Build Neo4j: embeddings + Filing / Section + relationships
uv run python src/store.py

# Optional: 13F investors (Company / Manager / OWNS_STOCK_IN) from processed CSV
uv run python src/load_13f_graph.py

# Retrieval benchmark (Azure embeddings + LLM judge)
uv run python src/retrieve.py

# Compare vector vs graph window vs hybrid + 0-10 scores (interactive)
uv run python src/rag_retrieval_compare.py
```

---

## 7. Cross-references

| Topic | Where |
|-------|--------|
| iXBRL, table flattening, section regex, full chunk metadata | [parsing.md](parsing.md) |
| Exact benchmark questions, filters, Cypher for Q05 | [../src/retrieve.py](../src/retrieve.py) (`BENCHMARK`) |
| Index name, node labels, graph wiring Cypher | [../src/store.py](../src/store.py) |
| Form 13F Company / Manager / `OWNS_STOCK_IN`, fulltext on managers | [../load_13f_graph.py](../load_13f_graph.py) |
| Optional Phase 3b sample (deferred) | [../src/graph_extract.py](../src/graph_extract.py) |
| Vector vs graph-window vs hybrid + 0-10 judge scores | [../src/rag_retrieval_compare.py](../src/rag_retrieval_compare.py) |

---

## 8. One-line summary

**We built a full chunk pipeline and a Neo4j-backed retriever (vector + graph structure), measured it with 10 questions using section hit rate and an LLM relevance score, and parked LLM-based entity graph extraction for a later comparison — not for lack of value, but to keep the baseline clear and the stack simpler until you need the extra layer.**
