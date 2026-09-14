"""Build the compact report as PDF and DOCX with embedded charts."""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from src.config import FIGURE_DIR, TABLE_DIR
from src.measures.windows import PLACEBO_WINDOW, TREATED_WINDOW, window_variance_shares

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/crypto_ohlcv_BTCUSD_5min_20220101_20251231.parquet"
DAILY = ROOT / "data/processed/btc_daily_session_measures.parquet"
EVENT = pd.Timestamp("2024-01-11")

# Set by main() for each report variant.
SOURCE = ROOT / "reports/report_compact.md"
OUT_PDF = ROOT / "reports/report_compact.pdf"
OUT_DOCX = ROOT / "reports/report_compact.docx"


def generate_figures() -> dict[str, Path]:
    """Generate the three figures used in the report."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    bars = pd.read_parquet(DATA)
    daily = pd.read_parquet(DAILY)
    headline = pd.read_csv(TABLE_DIR / "btc_headline_results.csv", index_col=0).iloc[0]

    monthly_path = FIGURE_DIR / "btc_monthly_session_share.png"
    monthly = daily["share_us"].resample("MS").mean() * 100
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.plot(monthly.index, monthly, color="#1f4e79", lw=1.8)
    ax.axvline(EVENT, color="black", ls=":", lw=1.2, label="ETF launch")
    ax.axhline(
        float(headline["mean_share_pre"]) * 100,
        color="#5b9bd5",
        ls=":",
        lw=1.1,
        label="Mean before launch",
    )
    ax.set_title("BTC variance occurring during 09:30 to 16:00 New York time")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of daily realised variance (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(monthly_path, dpi=180)
    plt.close(fig)

    contrast_path = FIGURE_DIR / "btc_equity_vs_cme_evening.png"
    windows = window_variance_shares(bars, 5)
    open_days = windows.loc[windows["etf_primary_open"].astype(bool)]
    equity = open_days[f"share_{TREATED_WINDOW}"].resample("MS").mean() * 100
    evening = open_days[f"share_{PLACEBO_WINDOW}"].resample("MS").mean() * 100
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.plot(
        equity.index,
        equity,
        color="#1f4e79",
        lw=1.8,
        label="09:30 to 16:00, ETF and CME available",
    )
    ax.plot(
        evening.index,
        evening,
        color="#c45911",
        lw=1.8,
        label="18:00 to 24:00, CME available, ETF primary market closed",
    )
    ax.axvline(EVENT, color="black", ls=":", lw=1.2, label="ETF launch")
    ax.set_title("Equity hours compared with the CME evening on ETF trading days")
    ax.set_xlabel("Month")
    ax.set_ylabel("Share of daily realised variance (%)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(contrast_path, dpi=180)
    plt.close(fig)

    placebo_path = FIGURE_DIR / "btc_placebo_date_shifts.png"
    placebo = pd.read_csv(TABLE_DIR / "btc_placebo_dates.csv")
    placebo["date"] = pd.to_datetime(placebo["date"])
    with np.errstate(divide="ignore", invalid="ignore"):
        placebo["se"] = (placebo["diff"] / placebo["t_hac"]).abs()
    order = placebo.sort_values("date").reset_index(drop=True)
    colours = [
        "#1f4e79" if bool(v) else "#999999"
        for v in order["is_hypothesised"].astype(bool)
    ]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.errorbar(
        order["date"],
        order["diff"] * 100,
        yerr=1.96 * order["se"] * 100,
        fmt="none",
        ecolor="#888888",
        elinewidth=1,
        capsize=3,
    )
    ax.scatter(order["date"], order["diff"] * 100, c=colours, s=42, zorder=3)
    ax.axhline(0, color="black", lw=0.9)
    ax.axvline(EVENT, color="#1f4e79", ls=":", lw=1.1)
    ax.set_title("Estimated change using the real date and artificial event dates")
    ax.set_xlabel("Date used as the assumed event")
    ax.set_ylabel("Estimated change in variance share (percentage points)")
    ax.grid(axis="y", alpha=0.2)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(placebo_path, dpi=180)
    plt.close(fig)

    return {
        "btc_monthly_session_share.png": monthly_path,
        "btc_equity_vs_cme_evening.png": contrast_path,
        "btc_placebo_date_shifts.png": placebo_path,
    }


def parse_markdown() -> list[tuple[str, str]]:
    """Parse the deliberately simple report Markdown into presentation blocks."""
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    blocks: list[tuple[str, str]] = []
    paragraph: list[str] = []
    in_code = False
    code: list[str] = []

    def flush() -> None:
        if paragraph:
            blocks.append(("body", " ".join(paragraph).strip()))
            paragraph.clear()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("```"):
            flush()
            if in_code:
                blocks.append(("code", "\n".join(code)))
                code.clear()
            in_code = not in_code
            continue
        if in_code:
            code.append(line)
            continue
        if not stripped:
            flush()
            continue
        image = re.fullmatch(r"!\[(.+)]\((.+)\)", stripped)
        if image:
            flush()
            blocks.append(("image", f"{image.group(1)}|{image.group(2)}"))
            continue
        if stripped.startswith("# "):
            flush()
            blocks.append(("title", stripped[2:]))
            continue
        if stripped.startswith("## "):
            flush()
            blocks.append(("h1", stripped[3:]))
            continue
        if stripped.startswith("### "):
            flush()
            blocks.append(("h2", stripped[4:]))
            continue
        if re.match(r"^\d+\.\s", stripped):
            flush()
            blocks.append(("number", stripped))
            continue
        paragraph.append(stripped)
    flush()
    return blocks


def clean_markdown(text: str) -> str:
    """Remove Markdown emphasis for formats with native styling."""
    return text.replace("**", "").replace("`", "")


def reportlab_text(text: str) -> str:
    """Convert the small amount of Markdown emphasis to ReportLab tags."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = text.split("**")
    return "".join(
        f"<b>{part}</b>" if i % 2 else part for i, part in enumerate(parts)
    ).replace("`", "")


def pdf_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=17,
            leading=21,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#1f4e79"),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "h2": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            spaceBefore=7,
            spaceAfter=3,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.2,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "number": ParagraphStyle(
            "Number",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.2,
            leftIndent=12,
            firstLineIndent=-12,
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.2,
            leading=11,
            leftIndent=12,
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=8,
        ),
    }


def add_page_number(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(A4[0] - 1.5 * cm, 0.8 * cm, str(doc.page))
    canvas.restoreState()


def build_pdf(blocks: list[tuple[str, str]]) -> None:
    styles = pdf_styles()
    story = []
    for kind, value in blocks:
        if kind in {"title", "h1", "h2", "body", "number"}:
            story.append(Paragraph(reportlab_text(value), styles[kind]))
        elif kind == "code":
            story.append(Paragraph(value.replace("\n", "<br/>"), styles["code"]))
        elif kind == "image":
            caption, relative = value.split("|", 1)
            path = (SOURCE.parent / relative).resolve()
            story.append(
                KeepTogether(
                    [
                        RLImage(str(path), width=17.2 * cm, height=7.45 * cm),
                        Paragraph(caption, styles["caption"]),
                    ]
                )
            )
    doc = SimpleDocTemplate(
        str(OUT_PDF),
        pagesize=A4,
        leftMargin=1.6 * cm,
        rightMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.3 * cm,
        title="Did US spot Bitcoin ETFs shift BTC price discovery into US trading hours?",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def build_docx(blocks: list[tuple[str, str]]) -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.65)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.7)
    section.right_margin = Inches(0.7)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(10)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08

    for kind, value in blocks:
        value = clean_markdown(value)
        if kind == "title":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(value)
            run.bold = True
            run.font.size = Pt(17)
            p.paragraph_format.space_after = Pt(12)
        elif kind == "h1":
            p = doc.add_heading(value, level=1)
            p.style.font.name = "Aptos Display"
            p.style.font.size = Pt(13)
        elif kind == "h2":
            doc.add_heading(value, level=2)
        elif kind == "body":
            p = doc.add_paragraph(value)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        elif kind == "number":
            doc.add_paragraph(value, style="List Number")
        elif kind == "code":
            p = doc.add_paragraph()
            run = p.add_run(value)
            run.font.name = "Consolas"
            run.font.size = Pt(8.5)
        elif kind == "image":
            caption, relative = value.split("|", 1)
            path = (SOURCE.parent / relative).resolve()
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(path), width=Inches(6.75))
            cap = doc.add_paragraph(caption)
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.runs[0].italic = True
            cap.runs[0].font.size = Pt(8)

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Compact research report")
    doc.save(OUT_DOCX)


def main() -> int:
    import argparse

    global SOURCE, OUT_PDF, OUT_DOCX

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("compact", "brief"),
        default="compact",
        help="compact = refined longer report; brief = earlier shorter version",
    )
    args = parser.parse_args()

    stem = "report_compact" if args.variant == "compact" else "report_brief"
    SOURCE = ROOT / f"reports/{stem}.md"
    OUT_PDF = ROOT / f"reports/{stem}.pdf"
    OUT_DOCX = ROOT / f"reports/{stem}.docx"

    required = [SOURCE, DATA, DAILY, TABLE_DIR / "btc_headline_results.csv"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    figures = generate_figures()
    blocks = parse_markdown()
    build_pdf(blocks)
    build_docx(blocks)
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_DOCX}")
    print("Figures:")
    for path in figures.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
