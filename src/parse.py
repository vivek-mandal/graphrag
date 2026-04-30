"""
Parse and chunk MSCI EDGAR HTML filings into LangChain Documents.

Pipeline per file:
  1. Strip hidden XBRL div (display:none)
  2. Walk DOM in document order, tracking page number and section
  3. Convert tables to pipe-delimited text; keep body text as-is
  4. Chunk with RecursiveCharacterTextSplitter (2000 chars, 200 overlap)
  5. Attach metadata: source, filing_type, period, chunk_index,
                      chunk_type, page_number, section
  6. Save to data/processed/<name>_chunks.jsonl (one JSON object per line)
"""

import json
import os
import re

from bs4 import BeautifulSoup, NavigableString, Tag
from langchain_text_splitters import RecursiveCharacterTextSplitter

from paths import PROCESSED_DIR, RAW_DIR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FILINGS = [
    {
        "filename": "msci_10k_fy2025.htm",
        "filing_type": "10-K",
        "period": "FY2025",
        "source_url": "https://www.sec.gov/Archives/edgar/data/1408198/000140819826000011/msci-20251231.htm",
    },
    {
        "filename": "msci_10q_q1_2026.htm",
        "filing_type": "10-Q",
        "period": "Q1 2026",
        "source_url": "https://www.sec.gov/Archives/edgar/data/1408198/000140819826000034/msci-20260331.htm",
    },
]

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
SEPARATORS = ["\n\n", "\n", ". ", " "]

# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Matches "Item 1.", "Item 1A.", "Item 7A." — with or without trailing period
_ITEM_RE = re.compile(r"^Item\s+\d+[A-Z]?\.?$")


def _extract_section_map(soup: BeautifulSoup) -> dict[Tag, str]:
    """
    Return a mapping of {DOM Tag → section label} for every bold Item span.

    Two HTML patterns in MSCI filings:
      Pattern A: <span bold>Item 1.</span>  next-sibling → "Business"
      Pattern B: <span bold>Item 7.\\xa0\\xa0Management's Discussion...</span>
    """
    section_map: dict[Tag, str] = {}

    for span in soup.find_all("span"):
        style = span.get("style", "")
        if "font-weight:700" not in style:
            continue

        raw = span.get_text(strip=True)

        # Pattern B — title embedded in same span with \xa0 separators
        # e.g. "Item 7.\xa0\xa0Management's Discussion..."
        if re.match(r"^Item\s+\d+[A-Z]?\.", raw) and "\xa0" in raw:
            parts = raw.replace("\xa0", " ").split(None, 2)
            if len(parts) >= 2:
                item_id = (parts[0] + " " + parts[1]).rstrip(".")
                title = " ".join(parts[2:]).strip() if len(parts) > 2 else ""
                label = f"{item_id.strip()} - {title}".strip(" -")
                section_map[span] = label
                continue

        # Pattern A — item number only; title in next sibling
        # Also handles split case: "Item 1A" span + "." span + title span
        if _ITEM_RE.match(raw):
            item_num = raw.rstrip(".")
            # Look ahead up to 3 siblings to find a non-punctuation title
            sibling = span.find_next_sibling()
            title = ""
            for _ in range(3):
                if sibling is None:
                    break
                t = sibling.get_text(strip=True)
                if t and t not in (".", ",", ";"):
                    title = t
                    break
                sibling = sibling.find_next_sibling()
            label = f"{item_num} - {title}".strip(" -")
            section_map[span] = label

    return section_map


# ---------------------------------------------------------------------------
# Table → pipe-delimited text
# ---------------------------------------------------------------------------

def _table_to_text(table: Tag) -> str:
    """Convert an HTML table to pipe-delimited rows."""
    rows = table.find_all("tr")
    if not rows:
        return table.get_text(separator=" ", strip=True)

    lines = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        cell_texts = [c.get_text(separator=" ", strip=True) for c in cells]
        cell_texts = [t for t in cell_texts if t]  # drop empty cells
        if cell_texts:
            lines.append(" | ".join(cell_texts))

    if not lines:
        return ""

    return "[TABLE]\n" + "\n".join(lines) + "\n[/TABLE]"


# ---------------------------------------------------------------------------
# DOM walker — produces ordered blocks with metadata
# ---------------------------------------------------------------------------

def _walk_dom(soup: BeautifulSoup, section_map: dict[Tag, str]):
    """
    Walk the DOM top-down in document order.
    Yield (text_block, page_number, section, chunk_type) tuples.

    - <hr> tags increment the page counter
    - Bold Item spans update current section
    - <table> tags are converted to pipe-delimited text
    - All other visible text is yielded as plain text blocks
    """
    current_page = 1
    current_section = "Preamble"
    visited_tables: set[int] = set()  # avoid double-yielding nested tables

    def _walk(node):
        nonlocal current_page, current_section

        if isinstance(node, NavigableString):
            text = str(node).strip()
            # Filter out pure whitespace, standalone page numbers, and XML declarations
            if (text
                    and not re.fullmatch(r"\d{1,3}", text)
                    and not text.startswith("<!--")
                    and not re.fullmatch(r"[_\-=]{5,}", text)):
                yield (text, current_page, current_section, "text")
            return

        if not isinstance(node, Tag):
            return

        tag = node.name

        # Page break
        if tag == "hr":
            current_page += 1
            return

        # Section header detection
        if tag == "span" and id(node) in {id(k) for k in section_map}:
            current_section = section_map[node]

        # Table — convert whole table, skip children
        if tag == "table":
            tid = id(node)
            if tid not in visited_tables:
                visited_tables.add(tid)
                text = _table_to_text(node)
                if text:
                    yield (text, current_page, current_section, "table")
            return  # don't recurse into table children

        # Recurse
        for child in node.children:
            yield from _walk(child)

    yield from _walk(soup)


# ---------------------------------------------------------------------------
# Main parsing function
# ---------------------------------------------------------------------------

def parse_filing(filing: dict) -> list[dict]:
    src_path = str(RAW_DIR / filing["filename"])
    print(f"\n[PARSE] {filing['filename']}")

    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
        html = f.read()

    # Strip XML processing instruction and HTML comments at the top (Workiva artifacts)
    html = re.sub(r"<\?xml[^?]*\?>", "", html, count=1)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

    soup = BeautifulSoup(html, "html.parser")

    # Step 1 — strip hidden XBRL block
    hidden = soup.find("div", style=lambda s: s and "display:none" in s)
    if hidden:
        hidden.decompose()
        print("  Stripped hidden XBRL block")

    # Step 2 — build section map
    section_map = _extract_section_map(soup)
    print(f"  Detected {len(section_map)} sections")

    # Step 3 — walk DOM, collect blocks
    blocks: list[tuple[str, int, str, str]] = list(_walk_dom(soup, section_map))
    print(f"  Collected {len(blocks)} raw text blocks")

    # Step 4 — group consecutive same-section/page blocks into larger passages
    # This gives the splitter more natural text to work with
    passages: list[tuple[str, int, str, str]] = []
    if blocks:
        buf_text, buf_page, buf_section, buf_type = blocks[0]
        for text, page, section, ctype in blocks[1:]:
            if section == buf_section and page == buf_page and ctype == buf_type:
                buf_text += "\n" + text
            else:
                passages.append((buf_text, buf_page, buf_section, buf_type))
                buf_text, buf_page, buf_section, buf_type = text, page, section, ctype
        passages.append((buf_text, buf_page, buf_section, buf_type))
    print(f"  Merged into {len(passages)} passages")

    # Step 5 — chunk each passage
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )

    records: list[dict] = []
    chunk_index = 0

    for passage_text, page, section, ctype in passages:
        # Clean up encoding artifacts
        passage_text = (
            passage_text
            .replace("\xa0", " ")
            .replace("\u2019", "'")
            .replace("\u2018", "'")
            .replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2014", "-")
            .replace("\u2013", "-")
            .replace("\ufffd", "")   # replacement char from encoding errors
        )
        # Remove underscore separator lines (footnote dividers from financial tables)
        # e.g. "________________" — keep the footnote text that follows, drop the line itself
        passage_text = re.sub(r"^_{5,}\s*$", "", passage_text, flags=re.MULTILINE)
        # Collapse excessive whitespace (but keep single newlines)
        passage_text = re.sub(r" {2,}", " ", passage_text)
        passage_text = re.sub(r"\n{3,}", "\n\n", passage_text)
        passage_text = passage_text.strip()

        if not passage_text:
            continue

        chunks = splitter.split_text(passage_text)
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            records.append({
                "chunk_index": chunk_index,
                "page_content": chunk,
                "source": filing["source_url"],
                "filename": filing["filename"],
                "filing_type": filing["filing_type"],
                "period": filing["period"],
                "page_number": page,
                "section": section,
                "chunk_type": ctype,
            })
            chunk_index += 1

    print(f"  Produced {len(records)} chunks")
    return records


# ---------------------------------------------------------------------------
# Save to JSONL
# ---------------------------------------------------------------------------

def save_jsonl(records: list[dict], filing: dict) -> str:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stem = filing["filename"].replace(".htm", "")
    out_path = str(PROCESSED_DIR / f"{stem}_chunks.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_kb = os.path.getsize(out_path) / 1024
    print(f"  Saved: {out_path}  ({size_kb:.1f} KB, {len(records)} lines)")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("MSCI Knowledge Graph RAG — Parser")
    print(f"chunk_size={CHUNK_SIZE}  overlap={CHUNK_OVERLAP}")
    print("=" * 60)

    for filing in FILINGS:
        records = parse_filing(filing)
        save_jsonl(records, filing)

    print("\nDone.")


if __name__ == "__main__":
    main()
