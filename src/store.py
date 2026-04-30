"""
Phase 3a — Build the graph store in Neo4j AuraDB.

Steps:
  1. Load all 593 chunks from JSONL files as LangChain Documents
  2. Embed with AzureOpenAI text-embedding-3-small and create Chunk nodes + vector index
  3. Create Filing nodes (one per filing)
  4. Create Section nodes (one per distinct section per filing)
  5. Wire relationships:
       PART_OF        : Chunk → Section
       CONTAINS       : Filing → Section
       NEXT           : Chunk → Chunk (sequential, within same section only)
       HAS_FIRST_CHUNK: Section → first Chunk in document order
  6. Print graph statistics
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings
from neo4j import GraphDatabase

from paths import PROCESSED_DIR

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
JSONL_FILES = [
    PROCESSED_DIR / "msci_10k_fy2025_chunks.jsonl",
    PROCESSED_DIR / "msci_10q_q1_2026_chunks.jsonl",
]

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

AZURE_ENDPOINT   = os.environ["AZURE_OPENAI_ENDPOINT"].strip()
AZURE_API_KEY    = os.environ["AZURE_OPENAI_API_KEY"].strip()
AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"].strip()
AZURE_EMBEDDING_DEPLOYMENT = os.environ["AZURE_EMBEDDING_DEPLOYMENT"].strip()

INDEX_NAME = "chunk_embeddings"
NODE_LABEL = "Chunk"

# ---------------------------------------------------------------------------
# Step 1 — Load JSONL chunks as LangChain Documents
# ---------------------------------------------------------------------------

def load_documents() -> list[Document]:
    docs = []
    for path in JSONL_FILES:
        with open(path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                doc = Document(
                    page_content=rec["page_content"],
                    metadata={
                        "chunk_index":  rec["chunk_index"],
                        "source":       rec["source"],
                        "filename":     rec["filename"],
                        "filing_type":  rec["filing_type"],
                        "period":       rec["period"],
                        "page_number":  rec["page_number"],
                        "section":      rec["section"],
                        "chunk_type":   rec["chunk_type"],
                    },
                )
                docs.append(doc)
    print(f"Loaded {len(docs)} documents from {len(JSONL_FILES)} files")
    return docs

# ---------------------------------------------------------------------------
# Step 2 — Embed + create Chunk nodes with vector index
# ---------------------------------------------------------------------------

def embed_and_store(docs: list[Document]) -> Neo4jVector:
    print(f"\nEmbedding {len(docs)} chunks with {AZURE_EMBEDDING_DEPLOYMENT}...")
    print("(This makes one batched API call — ~$0.006 total)")

    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )

    vector_store = Neo4jVector.from_documents(
        documents=docs,
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=INDEX_NAME,
        node_label=NODE_LABEL,
        text_node_property="page_content",
        embedding_node_property="embedding",
    )

    print(f"Created {len(docs)} Chunk nodes with vector index '{INDEX_NAME}'")
    return vector_store

# ---------------------------------------------------------------------------
# Step 3-5 — Wire Filing, Section nodes and all relationships via Cypher
# ---------------------------------------------------------------------------

def build_graph_structure(docs: list[Document]):
    print("\nBuilding graph structure (Filing, Section nodes + relationships)...")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:

        # -- Filing nodes (one per unique filing_type + period combo) --
        filings = {}
        for doc in docs:
            m = doc.metadata
            key = (m["filing_type"], m["period"])
            if key not in filings:
                filings[key] = {
                    "filing_type": m["filing_type"],
                    "period":      m["period"],
                    "source_url":  m["source"],
                    "filename":    m["filename"],
                }

        for f in filings.values():
            session.run("""
                MERGE (f:Filing {filing_type: $filing_type, period: $period})
                SET f.source_url = $source_url, f.filename = $filename
            """, **f)
        print(f"  Created {len(filings)} Filing nodes")

        # -- Section nodes --
        session.run("""
            MATCH (c:Chunk)
            WITH DISTINCT c.section AS name, c.filing_type AS ft, c.period AS period
            MERGE (s:Section {name: name, filing_type: ft, period: period})
        """)
        section_count = session.run(
            "MATCH (s:Section) RETURN count(s) AS n"
        ).single()["n"]
        print(f"  Created {section_count} Section nodes")

        # -- PART_OF: Chunk → Section --
        session.run("""
            MATCH (c:Chunk), (s:Section)
            WHERE c.section = s.name
              AND c.filing_type = s.filing_type
              AND c.period = s.period
            MERGE (c)-[:PART_OF]->(s)
        """)
        part_of_count = session.run(
            "MATCH ()-[r:PART_OF]->() RETURN count(r) AS n"
        ).single()["n"]
        print(f"  Created {part_of_count} PART_OF relationships")

        # -- CONTAINS: Filing → Section --
        session.run("""
            MATCH (f:Filing), (s:Section)
            WHERE f.filing_type = s.filing_type
              AND f.period = s.period
            MERGE (f)-[:CONTAINS]->(s)
        """)
        contains_count = session.run(
            "MATCH ()-[r:CONTAINS]->() RETURN count(r) AS n"
        ).single()["n"]
        print(f"  Created {contains_count} CONTAINS relationships")

        # -- NEXT: sequential Chunks within same section --
        # Sort by chunk_index within each (section, filing_type, period) group
        # Note: collect() with ORDER BY inside requires Neo4j 5.13+, use WITH + ORDER BY instead
        session.run("""
            MATCH (c:Chunk)
            WITH c.section AS sec, c.filing_type AS ft, c.period AS p, c
            ORDER BY c.chunk_index ASC
            WITH sec, ft, p, collect(c) AS chunks
            UNWIND range(0, size(chunks) - 2) AS i
            WITH chunks[i] AS curr, chunks[i+1] AS nxt
            MERGE (curr)-[:NEXT]->(nxt)
        """)
        next_count = session.run(
            "MATCH ()-[r:NEXT]->() RETURN count(r) AS n"
        ).single()["n"]
        print(f"  Created {next_count} NEXT relationships")

        # -- HAS_FIRST_CHUNK: Section → first Chunk in document order --
        session.run("""
            MATCH (s:Section)<-[:PART_OF]-(c:Chunk)
            WITH s, c
            ORDER BY c.chunk_index ASC
            WITH s, collect(c)[0] AS first
            MERGE (s)-[:HAS_FIRST_CHUNK]->(first)
        """)
        first_count = session.run(
            "MATCH ()-[r:HAS_FIRST_CHUNK]->() RETURN count(r) AS n"
        ).single()["n"]
        print(f"  Created {first_count} HAS_FIRST_CHUNK relationships")

    driver.close()

# ---------------------------------------------------------------------------
# Step 6 — Print graph statistics
# ---------------------------------------------------------------------------

def print_graph_stats():
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    with driver.session() as session:

        # Node counts by label
        print("\nNode counts:")
        result = session.run("""
            CALL apoc.meta.stats()
            YIELD labels
            RETURN labels
        """)
        # Fallback if APOC not available
        for label in ["Chunk", "Section", "Filing"]:
            n = session.run(
                f"MATCH (n:{label}) RETURN count(n) AS cnt"
            ).single()["cnt"]
            print(f"  {label:<12}: {n}")

        # Relationship counts by type
        print("\nRelationship counts:")
        for rel in ["NEXT", "PART_OF", "CONTAINS", "HAS_FIRST_CHUNK"]:
            n = session.run(
                f"MATCH ()-[r:{rel}]->() RETURN count(r) AS cnt"
            ).single()["cnt"]
            print(f"  {rel:<20}: {n}")

        # NEXT chain lengths per section (10-K only)
        print("\nNEXT chain lengths per section (10-K FY2025):")
        result = session.run("""
            MATCH (s:Section {filing_type: "10-K"})<-[:PART_OF]-(c:Chunk)
            WITH s.name AS section, count(c) AS chunk_count
            ORDER BY chunk_count DESC
            RETURN section, chunk_count
        """)
        for rec in result:
            bar = "#" * min(rec["chunk_count"], 50)
            print(f"  {rec['chunk_count']:3d}  {bar}  {rec['section'][:60]}")

        # Sections in 10-Q
        print("\nNEXT chain lengths per section (10-Q Q1 2026):")
        result = session.run("""
            MATCH (s:Section {filing_type: "10-Q"})<-[:PART_OF]-(c:Chunk)
            WITH s.name AS section, count(c) AS chunk_count
            ORDER BY chunk_count DESC
            RETURN section, chunk_count
        """)
        for rec in result:
            bar = "#" * min(rec["chunk_count"], 50)
            print(f"  {rec['chunk_count']:3d}  {bar}  {rec['section'][:60]}")

    driver.close()
    print("\nDone.")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MSCI Knowledge Graph RAG — Graph Store Builder")
    print("=" * 60)

    docs = load_documents()
    embed_and_store(docs)
    build_graph_structure(docs)
    print_graph_stats()

if __name__ == "__main__":
    main()
