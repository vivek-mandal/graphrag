"""
Phase 3b -- Entity & Relationship Extraction (optional, revisit later)

Runs LLMGraphTransformer on a sample of 10-K chunks (Item 1 + Item 1A) and
writes to Neo4j using LangChain's Neo4jGraph.add_graph_documents().

**Neo4j must have the APOC plugin** -- LangChain's importer uses
apoc.create / apoc.merge / apoc.meta. Enable APOC in Docker, e.g.:
  NEO4J_PLUGINS='["apoc"]'
  and allow the procedures in neo4j.conf (see Neo4j APOC install docs).

Pydantic may log serializer warnings from langchain_experimental; they are
cosmetic and safe to ignore.

If you are not using APOC yet, skip this script and improve in a later pass
(e.g. APOC-enabled DB or a custom Cypher writer).
"""

import json
import os
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import AzureChatOpenAI

from paths import PROCESSED_DIR

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_10K     = PROCESSED_DIR / "msci_10k_fy2025_chunks.jsonl"

SAMPLE_SECTIONS = {
    "Item 1 - Business",
    "Item 1A - Risk Factors",
}

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

AZURE_ENDPOINT    = os.environ["AZURE_OPENAI_ENDPOINT"].strip()
AZURE_API_KEY     = os.environ["AZURE_OPENAI_API_KEY"].strip()
AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"].strip()
AZURE_DEPLOYMENT  = os.environ["AZURE_DEPLOYMENT_NAME"].strip()

# ---------------------------------------------------------------------------
# Load sample chunks
# ---------------------------------------------------------------------------

def load_sample_chunks() -> list[Document]:
    docs = []
    with open(JSONL_10K, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["section"] not in SAMPLE_SECTIONS:
                continue
            doc = Document(
                page_content=rec["page_content"],
                metadata={
                    "id":          f"{rec['filename']}::chunk_{rec['chunk_index']}",
                    "chunk_index": rec["chunk_index"],
                    "section":     rec["section"],
                    "filing_type": rec["filing_type"],
                    "period":      rec["period"],
                    "chunk_type":  rec["chunk_type"],
                },
            )
            docs.append(doc)

    by_section: dict[str, int] = {}
    for d in docs:
        s = d.metadata["section"]
        by_section[s] = by_section.get(s, 0) + 1

    print(f"Loaded {len(docs)} sample chunks:")
    for section, count in sorted(by_section.items()):
        print(f"  {count:3d}  {section}")
    return docs

# ---------------------------------------------------------------------------
# LLMGraphTransformer
# ---------------------------------------------------------------------------

def extract_graph_documents(docs: list[Document]):
    llm = AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        temperature=0,
    )

    transformer = LLMGraphTransformer(llm=llm)

    print(f"\nRunning LLMGraphTransformer on {len(docs)} chunks...")
    print("(One LLM call per chunk.)\n")

    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    batch_size = 10
    all_graph_docs = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        graph_docs = transformer.convert_to_graph_documents(batch)
        all_graph_docs.extend(graph_docs)

        done = min(i + batch_size, len(docs))
        ent = sum(len(gd.nodes) for gd in all_graph_docs)
        rel = sum(len(gd.relationships) for gd in all_graph_docs)
        print(f"  [{done:3d}/{len(docs)}]  entities: {ent}  relationships: {rel}")

    te = sum(len(gd.nodes) for gd in all_graph_docs)
    tr = sum(len(gd.relationships) for gd in all_graph_docs)
    print(f"\nExtraction done: {te} entities, {tr} relationships across {len(docs)} chunks")
    return all_graph_docs

# ---------------------------------------------------------------------------
# Write to Neo4j (requires APOC)
# ---------------------------------------------------------------------------

def write_to_neo4j(graph_docs) -> Neo4jGraph:
    print("\nWriting entities to Neo4j (requires APOC)...")

    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )

    graph.add_graph_documents(
        graph_docs,
        baseEntityLabel=True,
        include_source=True,
    )

    print("Done writing to Neo4j.")
    return graph

# ---------------------------------------------------------------------------
# Statistics (schema matches LangChain: Document -[:MENTIONS]-> __Entity__)
# ---------------------------------------------------------------------------

def print_stats(graph: Neo4jGraph):
    print("\n" + "=" * 60)
    print("EXTRACTION STATISTICS")
    print("=" * 60)

    print("\nEntity types (by label on __Entity__):")
    result = graph.query("""
        MATCH (e:__Entity__)
        UNWIND labels(e) AS lbl
        WHERE lbl <> '__Entity__'
        RETURN lbl AS entity_type, count(*) AS n
        ORDER BY n DESC
    """)
    if result:
        for row in result:
            bar = "#" * min(row["n"], 40)
            print(f"  {row['n']:4d}  {bar}  {row['entity_type']}")
    else:
        print("  (none)")

    print("\nLLM relationship types (excluding structural):")
    result = graph.query("""
        MATCH ()-[r]->()
        WHERE NOT type(r) IN ['MENTIONS','PART_OF','NEXT','CONTAINS','HAS_FIRST_CHUNK']
        RETURN type(r) AS rel_type, count(*) AS n
        ORDER BY n DESC
        LIMIT 20
    """)
    if result:
        for row in result:
            bar = "#" * min(row["n"], 40)
            print(f"  {row['n']:4d}  {bar}  {row['rel_type']}")
    else:
        print("  (none)")

    print("\nTop entities by how many source documents mention them (MENTIONS):")
    result = graph.query("""
        MATCH (d:Document)-[:MENTIONS]->(e:__Entity__)
        UNWIND labels(e) AS lbl
        WITH e, lbl, count(d) AS mentions
        WHERE lbl <> '__Entity__'
        RETURN lbl AS type, e.id AS entity, mentions
        ORDER BY mentions DESC
        LIMIT 15
    """)
    if result:
        for row in result:
            print(f"  {row['mentions']:4d}x  [{row['type']}]  {row['entity']}")
    else:
        print("  (none)")

    print("\nSample Document -> entity (first 5):")
    result = graph.query("""
        MATCH (d:Document)-[:MENTIONS]->(e:__Entity__)
        RETURN d.id AS doc_id, e.id AS entity
        LIMIT 5
    """)
    if result:
        for row in result:
            print(f"  {str(row['doc_id'])!r:45s}  MENTIONS  {row['entity']}")
    else:
        print("  (none)")

    print("\nNeo4j Browser:")
    print("  MATCH (d:Document)-[:MENTIONS]->(e:__Entity__) RETURN d, e LIMIT 50")
    print("  MATCH p=(a:__Entity__)-[r]->(b:__Entity__) RETURN p LIMIT 30")

# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MSCI Graph RAG -- Phase 3b: Entity Extraction (sample, optional)")
    print("=" * 60)

    docs       = load_sample_chunks()
    graph_docs = extract_graph_documents(docs)
    graph      = write_to_neo4j(graph_docs)
    print_stats(graph)

    print("\nPhase 3b run complete (when APOC is available).")
    print("http://localhost:7474 -- Neo4j Browser")

if __name__ == "__main__":
    main()
