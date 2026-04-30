# graph-rag — MSCI filings, Neo4j, Form 13F

Use **MSCI** SEC **10-K / 10-Q** HTML as the source: parse and **chunk** to JSONL, **embed** with **Azure OpenAI**, load a **structural + vector** graph in **Neo4j**, and optionally add **Form 13F** holders of **MSCI** (managers, addresses, ownership).

---

## Repository layout

```
graph_rag/
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── PARSING.md                (stub → docs/parsing.md)
├── CHUNKING_RETRIEVAL.md     (stub → docs/chunking-and-retrieval.md)
│
├── src/                      # Python modules (see paths.py for data/ root)
│   ├── paths.py
│   ├── download.py, parse.py, store.py, retrieve.py
│   ├── load_13f_graph.py, form13f_msci_extract.py
│   ├── rag_retrieval_compare.py, graph_extract.py, main.py
│
├── notebooks/                # Jupyter (run with project root as cwd)
│   ├── neo4j_explore.ipynb, retriever.ipynb, inspect.ipynb, infotable_test.ipynb
│
├── docs/
│   ├── parsing.md
│   └── chunking-and-retrieval.md
│
├── data/
│   ├── raw/                  # EDGAR .htm
│   └── processed/            # JSONL, 13F CSV, benchmark JSON
```

---

## Quick start

1. **Install**

   ```bash
   cd graph_rag
   uv sync
   cp .env.example .env
   ```

2. **Fill `.env`**

   - **Neo4j:** `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` (optional: `NEO4J_DATABASE`)
   - **Azure OpenAI:** `AZURE_OPENAI_*`, `AZURE_EMBEDDING_DEPLOYMENT` (for `store.py` / retrieval)
   - **Chat / judge:** `AZURE_DEPLOYMENT_NAME` (for `retrieve.py`, `rag_retrieval_compare.py`)

   See [`.env.example`](.env.example) for the full list.

3. **Run the pipeline (reference)** — from the project root, so `data/` paths resolve.

   | Step | Command |
   |------|---------|
   | Download filings | `uv run python src/download.py` |
   | Parse + chunk | `uv run python src/parse.py` |
   | Build graph + embeddings | `uv run python src/store.py` |
   | Form 13F → CSV (optional) | `uv run python src/form13f_msci_extract.py <path-to-13F-tsv-folder>` |
   | Load 13F investors (after CSV exists) | `uv run python src/load_13f_graph.py` |
   | Benchmark retrieval | `uv run python src/retrieve.py` |

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/parsing.md](docs/parsing.md) | iXBRL handling, section detection, chunking, JSONL output |
| [docs/chunking-and-retrieval.md](docs/chunking-and-retrieval.md) | Neo4j model, vector index, retrieval metrics, 13F graph, commands |

Stub files at the repo root ([PARSING.md](PARSING.md), [CHUNKING_RETRIEVAL.md](CHUNKING_RETRIEVAL.md)) redirect to `docs/` so old links still work.

---

## What gets built in Neo4j (short)

- **Filing / Section / Chunk** with `NEXT`, `PART_OF`, `CONTAINS`, `HAS_FIRST_CHUNK`, and vector index `chunk_embeddings` on `Chunk.page_content`.
- **13F (after `load_13f_graph.py`):** `Company` (MSCI) → `Filing` (`FILED`); `Manager` → `Company` (`OWNS_STOCK_IN`); `Manager` → `Address` (`LOCATED_AT`); full-text on `Manager.name` (`managerNameFulltext`).

---

## Notebooks

| Notebook | Use |
|----------|-----|
| `notebooks/neo4j_explore.ipynb` | Connect to Neo4j, list labels, sample Cypher (no Azure) |
| `notebooks/retriever.ipynb` | Vector search via LangChain (needs working embeddings) |
| `notebooks/inspect.ipynb` | Raw HTML structure before parsing |
| `notebooks/infotable_test.ipynb` | Form 13F TSV + MSCI CUSIP smoke test |

---

## License

Add a `LICENSE` file if you open-source or redistribute the project.
