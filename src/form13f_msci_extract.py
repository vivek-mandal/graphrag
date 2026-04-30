"""
Form 13F bulk extract: MSCI (CUSIP 55354G100) positions joined to filing manager info.

Joins INFOTABLE + COVERPAGE + SUBMISSION on ACCESSION_NUMBER.
VALUE is in dollars for filings on/after 2023-01-03 (per SEC Form 13F metadata).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from paths import PROCESSED_DIR

# Identifies this pipeline / dataset in output rows (not present in raw TSVs).
DEFAULT_FORM13F_SOURCE = "sec_form_13f"

# MSCI Inc. common — filter holdings to this CUSIP (holders of MSCI stock).
DEFAULT_MSCI_CUSIP = "55354G100"

DEFAULT_CHUNK_SIZE = 300_000

_COVER_ADDRESS_COLS = [
    "FILINGMANAGER_STREET1",
    "FILINGMANAGER_STREET2",
    "FILINGMANAGER_CITY",
    "FILINGMANAGER_STATEORCOUNTRY",
    "FILINGMANAGER_ZIPCODE",
]


def _form13f_paths(data_dir: Path) -> dict[str, Path]:
    d = data_dir.resolve()
    return {
        "infotable": d / "INFOTABLE.tsv",
        "coverpage": d / "COVERPAGE.tsv",
        "submission": d / "SUBMISSION.tsv",
    }


def _format_manager_address(cover: pd.DataFrame) -> pd.Series:
    def row_addr(row: pd.Series) -> str:
        parts: list[str] = []
        for c in _COVER_ADDRESS_COLS:
            v = row.get(c)
            if pd.isna(v):
                continue
            s = str(v).strip()
            if s:
                parts.append(s)
        return ", ".join(parts)

    return cover.apply(row_addr, axis=1)


def _normalize_cik(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.replace(r"^0+(\d+)$", r"\1", regex=True)
    s = s.str.zfill(10)
    return s


def read_filtered_infotable(
    infotable_path: Path,
    cusip: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    cusip = cusip.strip()
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        infotable_path,
        sep="\t",
        chunksize=chunk_size,
        low_memory=False,
        dtype={"CUSIP": "string", "ACCESSION_NUMBER": "string", "SSHPRNAMTTYPE": "string"},
    ):
        m = chunk[chunk["CUSIP"] == cusip]
        if not m.empty:
            keep = [
                "ACCESSION_NUMBER",
                "INFOTABLE_SK",
                "NAMEOFISSUER",
                "CUSIP",
                "VALUE",
                "SSHPRNAMT",
                "SSHPRNAMTTYPE",
            ]
            use = [c for c in keep if c in m.columns]
            parts.append(m[use].copy())
    if not parts:
        return pd.DataFrame(
            columns=[
                "ACCESSION_NUMBER",
                "INFOTABLE_SK",
                "NAMEOFISSUER",
                "CUSIP",
                "VALUE",
                "SSHPRNAMT",
                "SSHPRNAMTTYPE",
            ]
        )
    return pd.concat(parts, ignore_index=True)


def load_cover_and_submission(cover_path: Path, sub_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cover = pd.read_csv(
        cover_path,
        sep="\t",
        low_memory=False,
        dtype={"ACCESSION_NUMBER": "string"},
    )
    sub = pd.read_csv(
        sub_path,
        sep="\t",
        low_memory=False,
        dtype={"ACCESSION_NUMBER": "string", "CIK": "string"},
    )
    if cover.duplicated("ACCESSION_NUMBER").any():
        cover = cover.drop_duplicates("ACCESSION_NUMBER", keep="last")
    if sub.duplicated("ACCESSION_NUMBER").any():
        sub = sub.drop_duplicates("ACCESSION_NUMBER", keep="last")
    return cover, sub


def build_msci_investor_table(
    data_dir: str | os.PathLike[str],
    *,
    cusip: str = DEFAULT_MSCI_CUSIP,
    source: str = DEFAULT_FORM13F_SOURCE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    paths = _form13f_paths(data_dir)
    for p in paths.values():
        if not p.is_file():
            raise FileNotFoundError(f"Missing Form 13F file: {p}")

    info = read_filtered_infotable(paths["infotable"], cusip, chunk_size=chunk_size)
    if info.empty:
        out = pd.DataFrame(
            columns=[
                "source",
                "managerCik",
                "managerAddress",
                "managerName",
                "reportCalendarOrQuarter",
                "cusip6",
                "cusip",
                "companyName",
                "value",
                "shares",
            ]
        )
        return out

    cover, sub = load_cover_and_submission(paths["coverpage"], paths["submission"])
    cover = cover.set_index("ACCESSION_NUMBER", verify_integrity=True)
    sub = sub.set_index("ACCESSION_NUMBER", verify_integrity=True)

    addr = _format_manager_address(cover)
    cover = cover.assign(_managerAddress=addr.values)

    j = info.merge(
        sub[["CIK"]].reset_index(),
        on="ACCESSION_NUMBER",
        how="left",
    )
    j = j.merge(
        cover[
            [
                "FILINGMANAGER_NAME",
                "REPORTCALENDARORQUARTER",
                "_managerAddress",
            ]
        ].reset_index(),
        on="ACCESSION_NUMBER",
        how="left",
    )

    shares = j["SSHPRNAMT"].where(j["SSHPRNAMTTYPE"] == "SH")
    cusip_str = j["CUSIP"].astype("string")
    out = pd.DataFrame(
        {
            "source": source,
            "managerCik": _normalize_cik(j["CIK"]),
            "managerAddress": j["_managerAddress"],
            "managerName": j["FILINGMANAGER_NAME"],
            "reportCalendarOrQuarter": j["REPORTCALENDARORQUARTER"],
            "cusip6": cusip_str.str.slice(0, 6),
            "cusip": cusip_str,
            "companyName": j["NAMEOFISSUER"],
            "value": j["VALUE"],
            "shares": shares,
        }
    )
    return out


def _to_report_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")


def _rows_in_filers_latest_report_period(w: pd.DataFrame) -> pd.DataFrame:
    """
    Per managerCik, keep only rows whose report end date is that filer's *latest*
    `reportCalendarOrQuarter` in this data. If every row for a filer has an unparseable
    date, keep all rows (same-period aggregation with messy labels only).
    """
    w = w.copy()
    w["_rpt_dt"] = _to_report_date(w["reportCalendarOrQuarter"].astype("string"))
    w["_rpt_max"] = w.groupby("managerCik", sort=True)["_rpt_dt"].transform("max")
    in_max_period = w["_rpt_dt"].notna() & w["_rpt_dt"].eq(w["_rpt_max"])
    all_unparsed = w["_rpt_dt"].isna() & w["_rpt_max"].isna()
    out = w[in_max_period | all_unparsed]
    return out.drop(columns=["_rpt_dt", "_rpt_max"])


def aggregate_msci_investors_by_manager(line_items: pd.DataFrame) -> pd.DataFrame:
    """
    One row per 13F filer (`managerCik`).

    **Same report date in the extract:** all matching rows are kept and `value` / `shares`
    are summed (e.g. multiple line items, same quarter).

    **Multiple report dates for one filer:** only rows in the *latest* calendar period
    (per `reportCalendarOrQuarter` parsed with ``%d-%b-%Y``) are summed; earlier periods
    in the same extract are dropped, so `value`/`shares` reflect a single as-of.
    `nReportingPeriodsInExtract` is the count of *distinct* report labels before that cut.
    """
    if line_items.empty:
        return line_items

    w0 = line_items.copy()
    n_per_filer = w0.groupby("managerCik", sort=True)[
        "reportCalendarOrQuarter"
    ].nunique().reset_index(name="nReportingPeriodsInExtract")
    w = _rows_in_filers_latest_report_period(w0)
    w["_rpt_dt"] = _to_report_date(w["reportCalendarOrQuarter"].astype("string"))

    def _pick_name(s: pd.Series) -> str:
        m = s.dropna()
        m = m.astype(str).str.strip()
        if m.empty:
            return ""
        mode = m.mode()
        return str(mode.iloc[0]) if len(mode) else m.iloc[0]

    g = w.groupby("managerCik", sort=True)
    out = g.agg(
        source=("source", "first"),
        managerName=("managerName", "first"),
        managerAddress=("managerAddress", "first"),
        cusip6=("cusip6", "first"),
        cusip=("cusip", "first"),
        companyName=("companyName", _pick_name),
        value=("value", "sum"),
        shares=("shares", "sum"),
    )
    out = out.reset_index()
    nlines = w.groupby("managerCik", sort=True).size().reset_index(name="holdingsCount")
    out = out.merge(nlines, on="managerCik", how="left")
    out = out.merge(n_per_filer, on="managerCik", how="left")
    out["nReportingPeriodsInExtract"] = out["nReportingPeriodsInExtract"].astype("int64")
    out["droppedEarlierPeriods"] = out["nReportingPeriodsInExtract"] > 1

    idx = w.groupby("managerCik", sort=True)["_rpt_dt"].idxmax()
    rep = w.loc[idx, ["managerCik", "reportCalendarOrQuarter"]].copy()
    rep = rep.drop_duplicates("managerCik", keep="last")
    out = out.drop(columns=["reportCalendarOrQuarter"], errors="ignore").merge(
        rep, on="managerCik", how="left"
    )
    out = out[
        [
            "source",
            "managerCik",
            "managerAddress",
            "managerName",
            "reportCalendarOrQuarter",
            "cusip6",
            "cusip",
            "companyName",
            "value",
            "shares",
            "holdingsCount",
            "nReportingPeriodsInExtract",
            "droppedEarlierPeriods",
        ]
    ]
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build MSCI 13F investor table (CUSIP 55354G100) from bulk Form 13F TSV folder.",
    )
    p.add_argument(
        "data_dir",
        type=Path,
        help="Path to Form 13F folder containing INFOTABLE.tsv, COVERPAGE.tsv, SUBMISSION.tsv",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Write CSV to this path (default: data/processed/form13f_msci_investors.csv under project root)",
    )
    p.add_argument(
        "--cusip",
        default=DEFAULT_MSCI_CUSIP,
        help=f"Holdings CUSIP to keep (default: {DEFAULT_MSCI_CUSIP})",
    )
    p.add_argument(
        "--source",
        default=DEFAULT_FORM13F_SOURCE,
        help=f"Value for output `source` column (default: {DEFAULT_FORM13F_SOURCE})",
    )
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="INFOTABLE read chunk size")
    p.add_argument(
        "--line-items",
        action="store_true",
        help="Keep one output row per 13F line item (no aggregation). Default: one row per unique manager, shares/value summed.",
    )
    return p.parse_args()


def _default_out_path(*, by_manager: bool) -> Path:
    name = "form13f_msci_investors_by_manager.csv" if by_manager else "form13f_msci_investors.csv"
    return PROCESSED_DIR / name


def main() -> None:
    args = _parse_args()
    by_manager = not args.line_items
    out_path = args.out or _default_out_path(by_manager=by_manager)
    df = build_msci_investor_table(
        args.data_dir,
        cusip=args.cusip,
        source=args.source,
        chunk_size=args.chunk_size,
    )
    if by_manager and not df.empty:
        df = aggregate_msci_investors_by_manager(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}" + (" (1 per managerCik)" if by_manager else " (per line item)"))


if __name__ == "__main__":
    main()
