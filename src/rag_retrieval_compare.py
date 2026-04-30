"""
Compare retrieval: vector (top-k) vs graph-window (vector-anchored) vs hybrid (vector top-k windows).

For each question, prints retrieved context size, model answer, and two LLM judge scores (0-10):
  - context_relevance: does the context contain enough grounded info to answer?
  - answer_quality: is the answer correct and supported by the context?

Requires: Neo4j with chunk_embeddings index, Chunk + NEXT (from store.py), Azure .env (same as retrieve.py).

Re-instantiate Neo4jVector if you change retrieval_query (this script builds a fresh one for hybrid mode).
"""

import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

from paths import PROCESSED_DIR
from langchain_community.vectorstores import Neo4jVector
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from neo4j import GraphDatabase

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"].strip()
AZURE_API_KEY = os.environ["AZURE_OPENAI_API_KEY"].strip()
AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"].strip()
AZURE_DEPLOYMENT = os.environ["AZURE_DEPLOYMENT_NAME"].strip()
AZURE_EMBED_DEPLOY = os.environ["AZURE_EMBEDDING_DEPLOYMENT"].strip()

INDEX_NAME = "chunk_embeddings"
NODE_LABEL = "Chunk"
TEXT_PROP = "page_content"
EMBED_PROP = "embedding"

VECTOR_K = 5  # mode A: how many similar chunks to concatenate
WINDOW_HOPS = 1  # +/- hops on NEXT for window expansion

# Cypher: expand prev / self / next around the vector hit (no APOC)

RETRIEVAL_QUERY_WINDOW = f"""
WITH node, score
OPTIONAL MATCH (prev:Chunk)-[:NEXT*1..{WINDOW_HOPS}]->(node)
OPTIONAL MATCH (node)-[:NEXT*1..{WINDOW_HOPS}]->(next:Chunk)
WITH node, score,
  coalesce(prev.page_content, "") AS p,
  coalesce(node.page_content, "") AS t,
  coalesce(next.page_content, "") AS n
RETURN
  p + "\n\n" + t + "\n\n" + n AS text,
  score,
  node {{.*, `page_content`: Null, `embedding`: Null, id: Null}} AS metadata
"""

# Window expansion around a specific center chunk (plus structural relationships).
WINDOW_AROUND_CENTER = f"""
MATCH (center:Chunk)
WHERE center.filing_type = $filing_type
  AND center.period = $period
  AND center.section = $section
  AND center.chunk_index = $chunk_index
OPTIONAL MATCH (prev:Chunk)-[:NEXT*1..{WINDOW_HOPS}]->(center)
OPTIONAL MATCH (center)-[:NEXT*1..{WINDOW_HOPS}]->(next:Chunk)
MATCH (s:Section {{name: $section, filing_type: $filing_type, period: $period}})
MATCH (f:Filing {{filing_type: $filing_type, period: $period}})
OPTIONAL MATCH (s)-[:HAS_FIRST_CHUNK]->(first:Chunk)
RETURN
  center.page_content AS center_text,
  coalesce(prev.page_content, '') AS prev_text,
  coalesce(next.page_content, '') AS next_text,
  coalesce(prev.chunk_index, -1) AS prev_chunk_index,
  coalesce(prev.page_number, -1) AS prev_page_number,
  center.chunk_index AS center_chunk_index,
  coalesce(center.page_number, -1) AS center_page_number,
  coalesce(next.chunk_index, -1) AS next_chunk_index,
  coalesce(next.page_number, -1) AS next_page_number,
  s.name AS section_name,
  f.filing_type AS filing_type,
  f.period AS period,
  coalesce(f.source_url, '') AS source_url,
  coalesce(first.chunk_index, -1) AS first_chunk_index,
  coalesce(first.page_number, -1) AS first_page_number
"""

# ---------------------------------------------------------------------------
# Stopwords for keyword extraction (graph anchor)
# ---------------------------------------------------------------------------

# (Keyword anchoring was removed: graph mode is vector-anchored now.)


def get_embeddings():
    return AzureOpenAIEmbeddings(
        azure_deployment=AZURE_EMBED_DEPLOY,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
    )


def get_llm(temperature: float = 0.0) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_deployment=AZURE_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        temperature=temperature,
    )


def vector_store_default(embeddings) -> Neo4jVector:
    return Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=INDEX_NAME,
        node_label=NODE_LABEL,
        text_node_property=TEXT_PROP,
        embedding_node_property=EMBED_PROP,
    )


def vector_store_window(embeddings) -> Neo4jVector:
    return Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=INDEX_NAME,
        node_label=NODE_LABEL,
        text_node_property=TEXT_PROP,
        embedding_node_property=EMBED_PROP,
        retrieval_query=RETRIEVAL_QUERY_WINDOW,
    )


# ---------------------------------------------------------------------------
# Retrieval — three modes
# ---------------------------------------------------------------------------

def _format_structural_header(
    filing_type: str,
    period: str,
    section: str,
    center_chunk_index: int,
    center_page_number: int | None,
    source_url: str,
    first_chunk_index: int | None,
    first_page_number: int | None,
) -> str:
    pg = center_page_number if center_page_number is not None else -1
    fci = first_chunk_index if first_chunk_index is not None else -1
    fpg = first_page_number if first_page_number is not None else -1
    return (
        "[STRUCTURE]\n"
        f"Filing: Filing(filing_type={filing_type!r}, period={period!r})\n"
        f"Section: Section(name={section!r})\n"
        f"Edges: Filing-[:CONTAINS]->Section, Chunk-[:PART_OF]->Section, "
        "prev-[:NEXT]->center-[:NEXT]->next\n"
        f"HAS_FIRST_CHUNK: Section-[:HAS_FIRST_CHUNK]->Chunk(chunk_index={fci}, page_number={fpg})\n"
        f"CenterChunk: chunk_index={center_chunk_index}, page_number={pg}\n"
        f"SourceURL: {source_url}\n"
        "[/STRUCTURE]\n"
    )


def _window_text(
    filing_type: str,
    period: str,
    section: str,
    source_url: str,
    prev_text: str,
    prev_chunk_index: int,
    prev_page_number: int,
    center_text: str,
    center_chunk_index: int,
    center_page_number: int,
    next_text: str,
    next_chunk_index: int,
    next_page_number: int,
) -> str:
    cite = (
        f"filing={filing_type} period={period} section={section!r} "
        f"source_url={source_url!r} "
    )
    parts = []
    if prev_text.strip():
        parts.append(
            "[PREV]\n"
            f"Citation: {cite}"
            f"chunk_index={prev_chunk_index} page_number={prev_page_number}\n\n"
            + prev_text.strip()
            + "\n[/PREV]"
        )
    parts.append(
        "[CENTER]\n"
        f"Citation: {cite}"
        f"chunk_index={center_chunk_index} page_number={center_page_number}\n\n"
        + center_text.strip()
        + "\n[/CENTER]"
    )
    if next_text.strip():
        parts.append(
            "[NEXT]\n"
            f"Citation: {cite}"
            f"chunk_index={next_chunk_index} page_number={next_page_number}\n\n"
            + next_text.strip()
            + "\n[/NEXT]"
        )
    return "\n\n".join(parts)


def retrieve_window_for_chunk(driver, meta: dict) -> tuple[str, str]:
    """
    Expand around a known chunk using NEXT and attach full structural header.
    Expects chunk metadata from Neo4jVector results (includes filing_type, period, section, chunk_index, etc.).
    """
    filing_type = meta.get("filing_type", "")
    period = meta.get("period", "")
    section = meta.get("section", "")
    chunk_index = int(meta.get("chunk_index", -1))
    page_number = meta.get("page_number", None)

    if not (filing_type and period and section and chunk_index >= 0):
        return "", "missing metadata for window expansion"

    with driver.session() as session:
        rec = session.run(
            WINDOW_AROUND_CENTER,
            filing_type=filing_type,
            period=period,
            section=section,
            chunk_index=chunk_index,
        ).single()

    if not rec:
        return "", "window query returned no rows"

    sec_name = rec.get("section_name", section)
    ft = rec.get("filing_type", filing_type)
    pr = rec.get("period", period)

    header = _format_structural_header(
        filing_type=ft,
        period=pr,
        section=sec_name,
        center_chunk_index=chunk_index,
        center_page_number=int(rec.get("center_page_number", page_number if page_number is not None else -1)),
        source_url=rec.get("source_url", ""),
        first_chunk_index=rec.get("first_chunk_index", -1),
        first_page_number=rec.get("first_page_number", -1),
    )
    su = (rec.get("source_url", "") or "").strip()
    window = _window_text(
        filing_type=ft,
        period=pr,
        section=sec_name,
        source_url=su,
        prev_text=rec.get("prev_text", "") or "",
        prev_chunk_index=int(rec.get("prev_chunk_index", -1)),
        prev_page_number=int(rec.get("prev_page_number", -1)),
        center_text=rec.get("center_text", "") or "",
        center_chunk_index=int(rec.get("center_chunk_index", chunk_index)),
        center_page_number=int(rec.get("center_page_number", -1)),
        next_text=rec.get("next_text", "") or "",
        next_chunk_index=int(rec.get("next_chunk_index", -1)),
        next_page_number=int(rec.get("next_page_number", -1)),
    )
    return header + "\n" + window, f"window +/-{WINDOW_HOPS} around chunk_index={chunk_index}"


def retrieve_vector(
    vs: Neo4jVector, question: str, flt: dict | None
) -> tuple[str, dict[str, Any]]:
    docs = vs.similarity_search(question, k=VECTOR_K, filter=flt)
    if not docs:
        return "", {"hits": []}
    parts = []
    hits = []
    for d in docs:
        if not (d.page_content or "").strip():
            continue
        m = d.metadata or {}
        ft = m.get("filing_type", "")
        pr = m.get("period", "")
        sec = m.get("section", "")
        idx = m.get("chunk_index", -1)
        pg = m.get("page_number", -1)
        src = m.get("source", m.get("source_url", ""))
        hits.append(
            {
                "filing_type": ft,
                "period": pr,
                "section": sec,
                "chunk_index": idx,
                "page_number": pg,
                "source_url": src,
            }
        )
        parts.append(
            "[CHUNK]\n"
            f"Citation: filing={ft} period={pr} section={sec!r} chunk_index={idx} page_number={pg} source_url={src}\n\n"
            + d.page_content.strip()
            + "\n[/CHUNK]"
        )
    return "\n\n---\n\n".join(parts), {"hits": hits}


def retrieve_graph_window(
    driver, vs_window: Neo4jVector, question: str, flt: dict | None
) -> tuple[str, dict[str, Any]]:
    """
    Vector-anchored: pick top-1 by vector similarity, then return a NEXT window.
    Uses Neo4jVector's retrieval_query to do windowing in Neo4j.
    """
    docs = vs_window.similarity_search(question, k=1, filter=flt)
    if not docs:
        return "", {"anchor": None}
    m = docs[0].metadata or {}
    # Use our explicit window fetch so we can include prev/next citations + filing source_url.
    ctx, note = retrieve_window_for_chunk(driver, m)
    if not ctx.strip():
        src = m.get("source", m.get("source_url", ""))
        return (docs[0].page_content or ""), {
            "anchor": {
                "filing_type": m.get("filing_type", ""),
                "period": m.get("period", ""),
                "section": m.get("section", ""),
                "chunk_index": m.get("chunk_index", -1),
                "page_number": m.get("page_number", -1),
                "source_url": src,
            },
            "note": "fallback to Neo4jVector text (window query empty)",
        }
    src = m.get("source", m.get("source_url", ""))
    anchor_meta = {
        "filing_type": m.get("filing_type", ""),
        "period": m.get("period", ""),
        "section": m.get("section", ""),
        "chunk_index": m.get("chunk_index", -1),
        "page_number": m.get("page_number", -1),
        "source_url": src,
        "window_note": note,
    }
    return ctx, {"anchor": anchor_meta}


def retrieve_hybrid_topk_windows(
    driver, vs_plain: Neo4jVector, question: str, flt: dict | None
) -> tuple[str, dict[str, Any]]:
    """
    Hybrid option 2: take vector top-k, expand each with a NEXT window using Cypher,
    dedupe by (filing_type, period, section, chunk_index), and concatenate.
    """
    docs = vs_plain.similarity_search(question, k=VECTOR_K, filter=flt)
    if not docs:
        return "", {"note": "no vector hits", "windows": []}

    seen: set[tuple[str, str, str, int]] = set()
    contexts: list[str] = []
    metas: list[dict[str, Any]] = []

    for d in docs:
        m = d.metadata or {}
        key = (
            str(m.get("filing_type", "")),
            str(m.get("period", "")),
            str(m.get("section", "")),
            int(m.get("chunk_index", -1)),
        )
        if key in seen:
            continue
        seen.add(key)
        ctx, note = retrieve_window_for_chunk(driver, m)
        if ctx.strip():
            contexts.append(ctx)
            metas.append(
                {
                    "filing_type": m.get("filing_type", ""),
                    "period": m.get("period", ""),
                    "section": m.get("section", ""),
                    "chunk_index": m.get("chunk_index", -1),
                    "page_number": m.get("page_number", -1),
                    "source_url": m.get("source", m.get("source_url", "")),
                    "window_note": note,
                }
            )

    if not contexts:
        return "", {"note": "window expansion returned empty contexts", "windows": []}
    return "\n\n====\n\n".join(contexts), {
        "note": f"windows={len(contexts)} (dedup from k={VECTOR_K})",
        "windows": metas,
    }


# ---------------------------------------------------------------------------
# Answer + judge
# ---------------------------------------------------------------------------

ANSWER_SYS = (
    "You are a financial research assistant. Answer ONLY using the provided context.\n"
    "If the context does not contain enough information, say you cannot answer from the context.\n"
    "Be concise (2-6 sentences unless the question asks for a list).\n\n"
    "CITATIONS ARE REQUIRED:\n"
    "- Every sentence (or every bullet) MUST end with a citation in parentheses.\n"
    "- Each citation MUST include at least: page_number (as page=...), chunk_index, and source_url from the Citation line.\n"
    "  Example: (filing=10-K period=FY2025 page=41 chunk=123 source_url='https://...')\n"
    "- If multiple chunks support a claim, cite 1-2 of them.\n"
)

JUDGE_SYS = """You score retrieval and answer quality for a Q&A over SEC filing excerpts.
Return ONLY a JSON object with these integer keys (0-10 each):
  "context_relevance": how well the context contains information needed to answer the question
  "answer_quality": how correct and grounded the answer is in the context (penalize hallucination)
  "brief_reason": one short sentence
No markdown, no code fence."""


def answer_from_context(llm: AzureChatOpenAI, question: str, context: str) -> str:
    if not context.strip():
        return "[No context retrieved.]"
    msg = HumanMessage(
        content=f"Context from filings:\n\n{context}\n\n---\n\nQuestion: {question}\n\nAnswer:"
    )
    r = llm.invoke([HumanMessage(content=ANSWER_SYS), msg])
    return (r.content or "").strip()


def judge_scores(
    llm: AzureChatOpenAI, question: str, context: str, answer: str
) -> dict[str, Any]:
    body = f"""Question: {question}

Context (excerpt, may be long):
{context[:12000]}

Answer to evaluate:
{answer}
"""
    r = llm.invoke([HumanMessage(content=JUDGE_SYS), HumanMessage(content=body)])
    raw = (r.content or "").strip()
    # strip ```json
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return {
            "context_relevance": 0,
            "answer_quality": 0,
            "brief_reason": "judge parse failed",
        }


# ---------------------------------------------------------------------------
# Results aggregation
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    mode: str
    context_len: int
    context_note: str
    answer: str
    context_relevance: float
    answer_quality: float
    retrieval_meta: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass
class QuestionResult:
    question: str
    filter_used: dict | None
    runs: list[RunResult] = field(default_factory=list)


def parse_filter_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    return json.loads(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("RAG retrieval compare: vector vs graph-window (vector-anchored) vs hybrid (top-k windows)")
    print("=" * 70)

    embeddings = get_embeddings()
    llm = get_llm(0.0)
    judge = get_llm(0.0)

    vs_plain = vector_store_default(embeddings)
    vs_window = vector_store_window(embeddings)
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

    nq = int(input("How many questions? (e.g. 10): ").strip() or "10")
    all_q: list[QuestionResult] = []

    for i in range(nq):
        print(f"\n--- Question {i + 1} / {nq} ---")
        q = input("Your question: ").strip()
        if not q:
            print("Skip empty question.")
            continue
        fline = input(
            "Optional metadata filter as JSON, e.g. "
            '{"filing_type":"10-K","section":"Item 1A - Risk Factors"}  (Enter for none): '
        )
        try:
            flt = parse_filter_line(fline)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON filter, ignoring: {e}")
            flt = None
        # No keyword input: graph mode is vector-anchored.

        qr = QuestionResult(question=q, filter_used=flt)

        # A — vector
        ctx_v, meta_v = retrieve_vector(vs_plain, q, flt)
        ans_v = answer_from_context(llm, q, ctx_v)
        sc_v = judge_scores(judge, q, ctx_v, ans_v)
        cr_v = float(sc_v.get("context_relevance", 0) or 0)
        aq_v = float(sc_v.get("answer_quality", 0) or 0)
        qr.runs.append(
            RunResult(
                mode="vector_topk",
                context_len=len(ctx_v),
                context_note=f"k={VECTOR_K}",
                answer=ans_v,
                context_relevance=cr_v,
                answer_quality=aq_v,
                retrieval_meta=meta_v,
                reason=str(sc_v.get("brief_reason", "")),
            )
        )

        # B — graph-window (vector-anchored top-1 + NEXT window)
        ctx_g, meta_g = retrieve_graph_window(driver, vs_window, q, flt)
        ans_g = answer_from_context(llm, q, ctx_g)
        sc_g = judge_scores(judge, q, ctx_g, ans_g)
        qr.runs.append(
            RunResult(
                mode="graph_window",
                context_len=len(ctx_g),
                context_note=f"vector top-1 + NEXT window (hops={WINDOW_HOPS})",
                answer=ans_g,
                context_relevance=float(sc_g.get("context_relevance", 0) or 0),
                answer_quality=float(sc_g.get("answer_quality", 0) or 0),
                retrieval_meta=meta_g,
                reason=str(sc_g.get("brief_reason", "")),
            )
        )

        # C — hybrid: vector top-k, expand each with NEXT window, dedupe
        ctx_h, meta_h = retrieve_hybrid_topk_windows(driver, vs_plain, q, flt)
        note_h = str(meta_h.get("note", ""))
        ans_h = answer_from_context(llm, q, ctx_h)
        sc_h = judge_scores(judge, q, ctx_h, ans_h)
        qr.runs.append(
            RunResult(
                mode="hybrid_topk_windows",
                context_len=len(ctx_h),
                context_note=note_h,
                answer=ans_h,
                context_relevance=float(sc_h.get("context_relevance", 0) or 0),
                answer_quality=float(sc_h.get("answer_quality", 0) or 0),
                retrieval_meta=meta_h,
                reason=str(sc_h.get("brief_reason", "")),
            )
        )

        all_q.append(qr)

        for r in qr.runs:
            print(f"\n  [{r.mode}] context_len={r.context_len}  {r.context_note}")
            print(f"  retrieval_meta: {json.dumps(r.retrieval_meta, ensure_ascii=False)[:900]}")
            print(f"  context_relevance={r.context_relevance:.1f}/10  answer_quality={r.answer_quality:.1f}/10  {r.reason}")
            print(textwrap.fill(r.answer, width=88))

    driver.close()

    # summary
    print("\n" + "=" * 70)
    print("SUMMARY (mean scores over questions)")
    print("=" * 70)
    modes = ["vector_topk", "graph_window", "hybrid_topk_windows"]
    for m in modes:
        crs = [x.context_relevance for q in all_q for x in q.runs if x.mode == m]
        aqs = [x.answer_quality for q in all_q for x in q.runs if x.mode == m]
        if not crs:
            continue
        print(
            f"  {m:<24}  context_rel={sum(crs)/len(crs):.2f}/10   answer={sum(aqs)/len(aqs):.2f}/10  (n={len(crs)})"
        )

    out_path = str(PROCESSED_DIR / "rag_retrieval_compare_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "question": q.question,
                    "filter": q.filter_used,
                    "runs": [
                        {
                            "mode": r.mode,
                            "context_len": r.context_len,
                            "context_note": r.context_note,
                            "retrieval_meta": r.retrieval_meta,
                            "context_relevance": r.context_relevance,
                            "answer_quality": r.answer_quality,
                            "brief_reason": r.reason,
                            "answer": r.answer,
                        }
                        for r in q.runs
                    ],
                }
                for q in all_q
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
