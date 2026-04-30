"""
Load Form 13F investor CSV into Neo4j (Company MSCI, Manager filers, OWNS_STOCK_IN).

Input: data/processed/form13f_msci_investors_by_manager.csv
(one row per managerCik; see form13f_msci_extract.py)

- MERGE Company (MSCI) and link to existing Filing 10-K FY2025 via FILED
- MERGE Manager nodes; MERGE OWNS_STOCK_IN to Company with value, shares, as_of
- MERGE Address from managerAddress (key = raw line or a placeholder per cik) with parsed US `state`;
  MERGE (Manager)-[:LOCATED_AT]->(Address) for aggregation by `address.state`
- CREATE FULLTEXT index on Manager.name (idempotent)

**Canonical `Company.name`:** Prefer the registrant line from existing `Chunk` nodes
(`filing_type` 10-K, `period` FY2025, `section` Preamble): EDGAR uses "MSCI INC."; we
normalize to **"MSCI Inc."** to match the plan / display. If no Preamble chunks are in
the graph (e.g. 13F load before `store.py`), the fallback is `MSCI_CANONICAL_NAME`.

Re-run safe: MERGE/SET refresh properties from current CSV.
"""

import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

from paths import PROCESSED_DIR

load_dotenv()

NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# Must match [store.py] / parse metadata for the MSCI 10-K in the graph
FILING_10K_TYPE = "10-K"
FILING_10K_PERIOD = "FY2025"

# Fallback if no 10-K Preamble chunks are loaded yet (aligns with SEC "Exact Name" line)
MSCI_CANONICAL_NAME = "MSCI Inc."
MSCI_TICKER = "MSCI"
MSCI_CUSIP = "55354G100"

_REG_MSCI = re.compile(
    r"(MSCI\s+INC\.)",  # Preamble in msci_10k_fy2025_chunks: "MSCI INC." before "(Exact Name...)"
    re.IGNORECASE,
)

CSV_PATH = PROCESSED_DIR / "form13f_msci_investors_by_manager.csv"
BATCH = 500

# 2-letter US (incl. DC) — validate parser for "..., ST, 12345" style lines
_US_STATE_CODES = frozenset(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT "
    "NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split()
)
# City line before ST and ZIP: ", ST, 12345" (SEC 13F addresses)
_RE_US_STATE = re.compile(r",\s*([A-Z]{2})\s*,\s*\d", re.IGNORECASE)


def parse_us_state_from_address(address: str) -> str | None:
    if not (address and str(address).strip()):
        return None
    t = re.sub(r"\s+", " ", str(address).strip().upper())
    m = _RE_US_STATE.search(t)
    if not m:
        return None
    code = m.group(1).upper()
    return code if code in _US_STATE_CODES else None


def _norm_cik(x) -> str:
    s = str(x).strip()
    if s.isdigit():
        return s.zfill(10)
    return s


def resolve_msci_name_from_graph(session) -> str:
    """
    Use the 10-K Preamble text already in the graph (same source as 10-K title line).
    If absent, return MSCI_CANONICAL_NAME.
    """
    result = session.run(
        """
        MATCH (ch:Chunk {filing_type: $ft, period: $pd, section: "Preamble"})
        RETURN ch.page_content AS t
        ORDER BY ch.chunk_index ASC
        LIMIT 10
        """,
        ft=FILING_10K_TYPE,
        pd=FILING_10K_PERIOD,
    )
    for rec in result:
        t = rec.get("t")
        if not t:
            continue
        m = _REG_MSCI.search(t)
        if m:
            return "MSCI Inc."
    return MSCI_CANONICAL_NAME


def ensure_company_and_filing(session) -> None:
    """MSCI Company node + FILED -> existing 10-K Filing from store.py."""
    filing = session.run(
        """
        MATCH (f:Filing {filing_type: $ft, period: $pd})
        RETURN f.filing_type AS ft
        LIMIT 1
        """,
        ft=FILING_10K_TYPE,
        pd=FILING_10K_PERIOD,
    ).single()
    if not filing:
        print(
            f"ERROR: No Filing node for {FILING_10K_TYPE} / {FILING_10K_PERIOD}. "
            "Run store.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    company_name = resolve_msci_name_from_graph(session)
    session.run(
        """
        MERGE (co:Company {ticker: $ticker})
        SET co.name = $name,
            co.cusip = $cusip
        WITH co
        MATCH (f:Filing {filing_type: $ft, period: $pd})
        MERGE (co)-[:FILED]->(f)
        """,
        ticker=MSCI_TICKER,
        name=company_name,
        cusip=MSCI_CUSIP,
        ft=FILING_10K_TYPE,
        pd=FILING_10K_PERIOD,
    )
    print(
        f"  MERGE Company({MSCI_TICKER}) name={company_name!r} -[:FILED]-> "
        f"Filing({FILING_10K_TYPE}, {FILING_10K_PERIOD})"
    )


def ensure_manager_fulltext_index(session) -> None:
    # Neo4j 5+ syntax; ignore if index already exists
    try:
        session.run(
            """
            CREATE FULLTEXT INDEX managerNameFulltext IF NOT EXISTS
            FOR (m:Manager) ON EACH [m.name]
            """
        )
        print("  FULLTEXT INDEX managerNameFulltext on Manager.name (created or exists)")
    except Exception as e:
        msg = str(e).lower()
        if "equivalent" in msg or "already" in msg:
            print("  FULLTEXT index: already present")
        else:
            print(f"  WARNING: fulltext index: {e}")


def load_managers_from_csv(session, path: Path) -> int:
    df = pd.read_csv(path, dtype={"managerCik": "string"}, low_memory=False)
    # Expected columns
    for col in ("managerCik", "managerName", "managerAddress", "value", "shares", "reportCalendarOrQuarter", "cusip"):
        if col not in df.columns:
            print(f"ERROR: missing column {col!r} in {path}", file=sys.stderr)
            sys.exit(1)

    n = 0
    for start in range(0, len(df), BATCH):
        batch = df.iloc[start : start + BATCH]
        rows = []
        for _, r in batch.iterrows():
            cik = _norm_cik(r["managerCik"])
            addr_text = (str(r["managerAddress"]).strip() if pd.notna(r["managerAddress"]) else "")
            addr_key = addr_text if addr_text else f"__empty__|{cik}"
            state = parse_us_state_from_address(addr_text)
            rows.append(
                {
                    "cik": cik,
                    "name": (str(r["managerName"]).strip() if pd.notna(r["managerName"]) else ""),
                    "address": addr_text,
                    "addr_key": addr_key,
                    "state": state,
                    "value": float(r["value"]) if pd.notna(r["value"]) else 0.0,
                    "shares": float(r["shares"]) if pd.notna(r["shares"]) else 0.0,
                    "as_of": (str(r["reportCalendarOrQuarter"]).strip() if pd.notna(r.get("reportCalendarOrQuarter")) else ""),
                    "cusip": (str(r["cusip"]).strip() if pd.notna(r.get("cusip")) else MSCI_CUSIP),
                }
            )

        session.run(
            """
            UNWIND $rows AS row
            MERGE (m:Manager {cik: row.cik})
            SET m.name = row.name,
                m.address = row.address
            MERGE (a:Address {key: row.addr_key})
            SET a.raw = row.address,
                a.state = row.state
            MERGE (m)-[:LOCATED_AT]->(a)
            WITH m, row
            MATCH (co:Company {ticker: $ticker})
            MERGE (m)-[r:OWNS_STOCK_IN]->(co)
            SET r.value = row.value,
                r.shares = row.shares,
                r.as_of = row.as_of,
                r.issuer_cusip = row.cusip,
                r.source = $source
            """,
            rows=rows,
            ticker=MSCI_TICKER,
            source="sec_form_13f",
        )
        n += len(rows)
        print(f"  loaded batch {start + 1}..{start + len(rows)} / {len(df)}")

    return n


def main():
    if not CSV_PATH.is_file():
        print(f"ERROR: CSV not found: {CSV_PATH.resolve()}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("Load Form 13F investors into Neo4j")
    print("=" * 60)
    print(f"CSV: {CSV_PATH.resolve()}")
    print(f"Canonical name: Preamble 10-K FY2025 in graph, else {MSCI_CANONICAL_NAME!r}")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    with driver.session() as session:
        ensure_company_and_filing(session)
        n = load_managers_from_csv(session, CSV_PATH)
        ensure_manager_fulltext_index(session)

    driver.close()
    print(f"\nDone. Managers processed: {n}")
    print("Example: CALL db.index.fulltext.queryNodes('managerNameFulltext', 'jpmorgan') YIELD node, score RETURN node.name, score LIMIT 5")


if __name__ == "__main__":
    main()
