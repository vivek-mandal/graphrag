"""
Phase 3a — Retrieval Quality Benchmark

Tests 3 retrieval modes against 10 benchmark questions:
  Mode A: Pure semantic search (vector similarity)
  Mode B: Metadata-filtered semantic search (section/filing_type pre-filter)
  Mode C: Pure Cypher (structured property queries)

Evaluation:
  - Section-hit@k : did any top-k chunk come from the expected section?
  - LLM score     : GPT-4.1 rates each retrieved chunk 1-3 for relevance
  - Prints a summary table at the end
"""

import os
import json
from dotenv import load_dotenv
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import HumanMessage
from neo4j import GraphDatabase

from paths import PROCESSED_DIR

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["AZURE_EMBEDDING_DEPLOYMENT"].strip(),
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
    api_key=os.environ["AZURE_OPENAI_API_KEY"].strip(),
    api_version=os.environ["AZURE_OPENAI_API_VERSION"].strip(),
)

llm = AzureChatOpenAI(
    azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"].strip(),
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].strip(),
    api_key=os.environ["AZURE_OPENAI_API_KEY"].strip(),
    api_version=os.environ["AZURE_OPENAI_API_VERSION"].strip(),
    temperature=0,
)

K = 5  # top-k chunks to retrieve per question

# ---------------------------------------------------------------------------
# Benchmark questions
# ---------------------------------------------------------------------------

BENCHMARK = [
    {
        "id": "Q01",
        "question": "What are MSCI's main business segments?",
        "expected_section": "Item 1 - Business",
        "mode": "filtered",
        "filter": {"filing_type": "10-K", "section": "Item 1 - Business"},
    },
    {
        "id": "Q02",
        "question": "What risk factors does MSCI highlight for its Index business?",
        "expected_section": "Item 1A - Risk Factors",
        "mode": "filtered",
        "filter": {"section": "Item 1A - Risk Factors"},
    },
    {
        "id": "Q03",
        "question": "How did MSCI's operating revenue change from 2024 to 2025?",
        "expected_section": "Item 7",
        "mode": "filtered",
        "filter": {"filing_type": "10-K", "chunk_type": "table"},
    },
    {
        "id": "Q04",
        "question": "What cybersecurity risks does MSCI face?",
        "expected_section": "Item 1C - Cybersecurity",
        "mode": "filtered",
        "filter": {"section": "Item 1C - Cybersecurity"},
    },
    {
        "id": "Q05",
        "question": "What is MSCI's ticker symbol and which stock exchange is it listed on?",
        "expected_section": "Preamble",
        "mode": "cypher",
        "filter": None,
        "cypher": """
            MATCH (c:Chunk)
            WHERE c.section = 'Preamble' AND c.filing_type = '10-K'
              AND (c.page_content CONTAINS 'New York Stock Exchange'
                   OR c.page_content CONTAINS 'Trading Symbol'
                   OR c.page_content CONTAINS 'MSCI')
            RETURN c.page_content AS page_content, c.section AS section,
                   c.chunk_index AS chunk_index, c.page_number AS page_number,
                   c.filing_type AS filing_type, c.chunk_type AS chunk_type
            ORDER BY c.chunk_index ASC
            LIMIT 5
        """,
    },
    {
        "id": "Q06",
        "question": "What are MSCI's ESG and sustainability products and offerings?",
        "expected_section": "Item 1 - Business",
        "mode": "filtered",
        "filter": {"filing_type": "10-K", "section": "Item 1 - Business"},
    },
    {
        "id": "Q07",
        "question": "How did MSCI's Index segment revenue perform in Q1 2026 compared to Q1 2025?",
        "expected_section": "Item 1 - Financial Statements",
        "mode": "filtered",
        "filter": {"filing_type": "10-Q", "chunk_type": "table"},
    },
    {
        "id": "Q08",
        "question": "What are the key risks related to competition and competitive threats that MSCI faces from competitors in the index and analytics market?",
        "expected_section": "Item 1A - Risk Factors",
        "mode": "filtered",
        "filter": {"section": "Item 1A - Risk Factors"},
    },
    {
        "id": "Q09",
        "question": "What is MSCI's total outstanding debt as of December 2025?",
        "expected_section": "Item 8 - Financial Statements",
        "mode": "filtered",
        "filter": {"filing_type": "10-K", "section": "Item 8 - Financial Statements and Supplementary Data", "chunk_type": "table"},
    },
    {
        "id": "Q10",
        "question": "What properties does MSCI lease and in which cities are they located?",
        "expected_section": "Item 2 - Properties",
        "mode": "filtered",
        "filter": {"section": "Item 2 - Properties"},
    },
]

# ---------------------------------------------------------------------------
# Vector store connection
# ---------------------------------------------------------------------------

def get_vector_store() -> Neo4jVector:
    return Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="chunk_embeddings",
        node_label="Chunk",
        text_node_property="page_content",
        embedding_node_property="embedding",
    )

# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def retrieve_semantic(vs: Neo4jVector, question: str, filter_dict: dict | None) -> list[dict]:
    """Mode A (no filter) or Mode B (with metadata filter)."""
    kwargs = {"k": K}
    if filter_dict:
        kwargs["filter"] = filter_dict
    results = vs.similarity_search(question, **kwargs)
    return [
        {
            "page_content": r.page_content,
            "section":      r.metadata.get("section", ""),
            "chunk_index":  r.metadata.get("chunk_index", -1),
            "page_number":  r.metadata.get("page_number", -1),
            "filing_type":  r.metadata.get("filing_type", ""),
            "chunk_type":   r.metadata.get("chunk_type", ""),
        }
        for r in results
    ]


def retrieve_cypher(cypher: str) -> list[dict]:
    """Mode C — pure Cypher query."""
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    results = []
    with driver.session() as session:
        for rec in session.run(cypher):
            results.append({
                "page_content": rec.get("page_content", ""),
                "section":      rec.get("section", ""),
                "chunk_index":  rec.get("chunk_index", -1),
                "page_number":  rec.get("page_number", -1),
                "filing_type":  rec.get("filing_type", ""),
                "chunk_type":   rec.get("chunk_type", ""),
            })
    driver.close()
    return results[:K]

# ---------------------------------------------------------------------------
# LLM-as-judge scoring
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating retrieval quality for a financial document Q&A system.

Question: {question}

Retrieved chunk:
\"\"\"
{chunk}
\"\"\"

Rate how relevant this chunk is to answering the question above.
Respond with ONLY a single integer: 1, 2, or 3.
  1 = Not relevant (chunk does not help answer the question)
  2 = Partially relevant (chunk contains some related information)
  3 = Highly relevant (chunk directly helps answer the question)
"""

def score_chunk(question: str, chunk_text: str) -> int:
    prompt = JUDGE_PROMPT.format(question=question, chunk=chunk_text[:800])
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        score = int(response.content.strip()[0])
        return score if score in (1, 2, 3) else 1
    except Exception:
        return 1

# ---------------------------------------------------------------------------
# Section hit check
# ---------------------------------------------------------------------------

def section_hit(chunks: list[dict], expected_section: str) -> bool:
    """True if any retrieved chunk's section contains the expected section string."""
    for c in chunks:
        if expected_section.lower() in c["section"].lower():
            return True
    return False

# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark():
    print("=" * 70)
    print("MSCI GraphRAG — Retrieval Quality Benchmark")
    print(f"k={K} chunks retrieved per question")
    print("=" * 70)

    vs = get_vector_store()

    results = []

    for q in BENCHMARK:
        print(f"\n{'-'*70}")
        print(f"[{q['id']}] {q['question']}")
        print(f"  Expected section : {q['expected_section']}")
        print(f"  Mode             : {q['mode'].upper()}")

        # Retrieve
        if q["mode"] == "cypher":
            chunks = retrieve_cypher(q["cypher"])
        else:
            chunks = retrieve_semantic(vs, q["question"], q.get("filter"))

        if not chunks:
            print("  WARNING: No chunks retrieved!")
            results.append({**q, "hit": False, "avg_score": 0.0, "chunks": []})
            continue

        # Section hit
        hit = section_hit(chunks, q["expected_section"])

        # LLM scores
        scores = []
        for i, c in enumerate(chunks):
            score = score_chunk(q["question"], c["page_content"])
            scores.append(score)
            marker = "[HIT]" if q["expected_section"].lower() in c["section"].lower() else "     "
            print(f"  {marker} #{i+1} score={score}/3  section={c['section'][:50]}  page={c['page_number']}")
            print(f"         {c['page_content'][:120].replace(chr(10), ' ')}...")

        avg_score = sum(scores) / len(scores)
        print(f"  Section hit@{K}: {'YES' if hit else 'NO'}")
        print(f"  Avg LLM score : {avg_score:.2f}/3")

        results.append({
            **q,
            "hit": hit,
            "avg_score": avg_score,
            "scores": scores,
            "chunks": chunks,
        })

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'ID':<5} {'Mode':<10} {'Hit@5':<7} {'AvgScore':<10} {'Question'}")
    print("-" * 70)

    total_hits = 0
    total_score = 0.0
    for r in results:
        hit_str   = "YES" if r["hit"] else "NO"
        score_str = f"{r['avg_score']:.2f}/3"
        q_short   = r["question"][:45]
        print(f"{r['id']:<5} {r['mode']:<10} {hit_str:<7} {score_str:<10} {q_short}")
        total_hits  += int(r["hit"])
        total_score += r["avg_score"]

    n = len(results)
    print("-" * 70)
    print(f"{'TOTAL':<5} {'':10} {total_hits}/{n} hits   {total_score/n:.2f}/3 avg score")
    print()

    # Save results to JSON for further analysis
    out = []
    for r in results:
        out.append({
            "id":              r["id"],
            "question":        r["question"],
            "expected_section": r["expected_section"],
            "mode":            r["mode"],
            "hit":             r["hit"],
            "avg_score":       r["avg_score"],
            "scores":          r.get("scores", []),
            "retrieved_sections": [c["section"] for c in r.get("chunks", [])],
        })

    out_path = str(PROCESSED_DIR / "benchmark_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    run_benchmark()
