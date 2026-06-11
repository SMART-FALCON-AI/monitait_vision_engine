"""Compact Markdown→PDF generator for the user manual (USER_MANUAL.md → .pdf).

Uses reportlab.platypus directly so we don't need pandoc / wkhtmltopdf /
weasyprint / xhtml2pdf — just reportlab + python-markdown (for inline parsing).

Handles the markdown subset the manual actually uses:
- Headings  (# / ## / ### / ####)
- Paragraphs
- Bullet lists (- / *)
- Numbered lists (1. / 2. / ...)
- Inline `code` and **bold** and *italic*
- Fenced code blocks (```...```)
- Tables (GFM pipe tables)
- Blockquotes (> ...)
- Horizontal rules (---)
- Images ![alt](path) — embedded if path exists

Run:
    python md_to_pdf.py USER_MANUAL.md USER_MANUAL.pdf
"""

from __future__ import annotations

import os
import re
import sys
import html
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, ListFlowable, ListItem, Preformatted, HRFlowable,
)
from reportlab.lib.enums import TA_LEFT


# ---------- Styles ----------

styles = getSampleStyleSheet()
base_font = "Helvetica"
mono_font = "Courier"

body = ParagraphStyle(
    "Body", parent=styles["BodyText"],
    fontName=base_font, fontSize=10, leading=14, spaceAfter=6,
)
h1 = ParagraphStyle("H1", parent=body, fontSize=22, leading=26, spaceBefore=18, spaceAfter=12, textColor=colors.HexColor("#0b3d91"), fontName=base_font + "-Bold")
h2 = ParagraphStyle("H2", parent=body, fontSize=16, leading=20, spaceBefore=14, spaceAfter=8,  textColor=colors.HexColor("#0b3d91"), fontName=base_font + "-Bold")
h3 = ParagraphStyle("H3", parent=body, fontSize=13, leading=16, spaceBefore=10, spaceAfter=6,  textColor=colors.HexColor("#163e6e"), fontName=base_font + "-Bold")
h4 = ParagraphStyle("H4", parent=body, fontSize=11, leading=14, spaceBefore=8,  spaceAfter=4,  textColor=colors.HexColor("#163e6e"), fontName=base_font + "-Bold")
code = ParagraphStyle("Code", parent=body, fontName=mono_font, fontSize=8.5, leading=11, leftIndent=10, backColor=colors.HexColor("#f4f4f4"), borderPadding=4)
quote = ParagraphStyle("Quote", parent=body, leftIndent=14, textColor=colors.HexColor("#444"), borderColor=colors.HexColor("#0b3d91"))


# ---------- Inline markdown → reportlab markup ----------

_RE_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_BOLD       = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC     = re.compile(r"(?<!\w)\*(.+?)\*(?!\w)")
_RE_LINK       = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def inline_md_to_rl(text: str) -> str:
    """Convert inline markdown to reportlab Paragraph markup (a tiny HTML subset)."""
    # 1) escape first
    out = html.escape(text)
    # 2) inline code (after escape so the `<` etc inside backticks are safe)
    out = _RE_INLINE_CODE.sub(lambda m: f'<font name="{mono_font}" backColor="#f4f4f4">{m.group(1)}</font>', out)
    # 3) links [text](href) — internal anchors (#foo) render as plain text since
    #    reportlab needs a fully-resolved destination. External links keep their href.
    def _link_repl(m):
        href = m.group(2)
        if href.startswith("#"):
            return m.group(1)
        return f'<link href="{href}" color="#0b3d91">{m.group(1)}</link>'
    out = _RE_LINK.sub(_link_repl, out)
    # 4) bold / italic — note: keep the *not-after-backtick* simple
    out = _RE_BOLD.sub(r"<b>\1</b>", out)
    out = _RE_ITALIC.sub(r"<i>\1</i>", out)
    return out


# ---------- Block parsing ----------

def parse_markdown(text: str, base_dir: Path):
    """Yield (kind, payload) tuples from the markdown source."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.rstrip()

        # Fenced code
        if stripped.startswith("```"):
            j = i + 1
            buf = []
            while j < n and not lines[j].rstrip().startswith("```"):
                buf.append(lines[j])
                j += 1
            yield ("code", "\n".join(buf))
            i = j + 1
            continue

        # Horizontal rule
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            yield ("hr", None)
            i += 1
            continue

        # Headings
        m = re.match(r"(#{1,4})\s+(.+)", stripped)
        if m:
            level = len(m.group(1))
            yield (f"h{level}", m.group(2).strip())
            i += 1
            continue

        # Blockquote
        if stripped.startswith("> "):
            buf = []
            while i < n and lines[i].rstrip().startswith("> "):
                buf.append(lines[i].rstrip()[2:])
                i += 1
            yield ("quote", " ".join(buf))
            continue

        # Bullet list
        if re.match(r"^[\-\*]\s+", stripped):
            items = []
            while i < n and re.match(r"^[\-\*]\s+", lines[i].rstrip()):
                items.append(re.sub(r"^[\-\*]\s+", "", lines[i].rstrip()))
                i += 1
            yield ("ul", items)
            continue

        # Numbered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\d+\.\s+", lines[i].rstrip()):
                items.append(re.sub(r"^\d+\.\s+", "", lines[i].rstrip()))
                i += 1
            yield ("ol", items)
            continue

        # Tables — header row then divider row then body rows
        if "|" in stripped and i + 1 < n and re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?", lines[i+1].strip()):
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            body_rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                body_rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            yield ("table", (header_cells, body_rows))
            continue

        # Image-only line
        m = re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if m:
            yield ("img", (m.group(1), m.group(2)))
            i += 1
            continue

        # Blank line → spacer
        if not stripped:
            yield ("space", None)
            i += 1
            continue

        # Paragraph (consume consecutive non-blank lines)
        buf = [stripped]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,4}\s|>\s|[\-\*]\s|\d+\.\s|```|---|!\[)", lines[i].strip()) and "|" not in lines[i]:
            buf.append(lines[i].strip())
            i += 1
        yield ("p", " ".join(buf))


# ---------- Render ----------

def render(md_path: Path, pdf_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    base_dir = md_path.parent

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=2 * cm,
        title="Monitait Vision Engine — User Manual",
    )
    story = []

    for kind, payload in parse_markdown(text, base_dir):
        if kind == "h1":
            story.append(Paragraph(inline_md_to_rl(payload), h1))
        elif kind == "h2":
            story.append(PageBreak())
            story.append(Paragraph(inline_md_to_rl(payload), h2))
        elif kind == "h3":
            story.append(Paragraph(inline_md_to_rl(payload), h3))
        elif kind == "h4":
            story.append(Paragraph(inline_md_to_rl(payload), h4))
        elif kind == "p":
            story.append(Paragraph(inline_md_to_rl(payload), body))
        elif kind == "quote":
            story.append(Paragraph(inline_md_to_rl(payload), quote))
        elif kind == "hr":
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ccc"), spaceBefore=8, spaceAfter=8))
        elif kind == "ul":
            items = [ListItem(Paragraph(inline_md_to_rl(it), body), leftIndent=12) for it in payload]
            story.append(ListFlowable(items, bulletType="bullet", start="•", leftIndent=18))
        elif kind == "ol":
            items = [ListItem(Paragraph(inline_md_to_rl(it), body), leftIndent=12) for it in payload]
            story.append(ListFlowable(items, bulletType="1", leftIndent=18))
        elif kind == "code":
            story.append(Preformatted(payload, code))
        elif kind == "table":
            header, rows = payload
            data = [[Paragraph(inline_md_to_rl(c), body) for c in header]] + \
                   [[Paragraph(inline_md_to_rl(c), body) for c in r] for r in rows]
            # Equal column widths
            ncols = max(len(header), max((len(r) for r in rows), default=1))
            col_w = (doc.width) / ncols
            t = Table(data, colWidths=[col_w] * ncols, repeatRows=1)
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fa")),
                ("FONTNAME",  (0, 0), (-1, 0), base_font + "-Bold"),
                ("FONTSIZE",  (0, 0), (-1, -1), 9),
                ("GRID",      (0, 0), (-1, -1), 0.4, colors.HexColor("#ccc")),
                ("VALIGN",    (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING",   (0, 0), (-1, -1), 4),
                ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
                ("TOPPADDING",    (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))
        elif kind == "img":
            alt, path = payload
            img_path = (base_dir / path).resolve()
            if img_path.is_file():
                try:
                    img = Image(str(img_path), width=doc.width)
                    img._restrictSize(doc.width, 16 * cm)
                    story.append(img)
                    story.append(Spacer(1, 6))
                except Exception as e:
                    story.append(Paragraph(f"<i>[image not loaded: {alt}]</i>", body))
            else:
                story.append(Paragraph(f"<i>[missing image: {path}]</i>", body))
        elif kind == "space":
            story.append(Spacer(1, 4))

    doc.build(story)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: md_to_pdf.py <input.md> <output.pdf>")
        sys.exit(1)
    render(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"wrote {sys.argv[2]}")
