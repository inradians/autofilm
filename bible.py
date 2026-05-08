"""bible.py — generate a production-bible PDF for a finished experiment.

Run as:
    python bible.py exp_001
    python bible.py latest
    python bible.py --all

The bible is a single PDF that documents everything about that version of
the film: screenplay, cast, locations, look book, storyboard with panels,
shot list, EDL, music briefs, and the critic's report. It serves as the
canonical reference for that experiment — the kind of document a production
designer or 1st AD would carry in a bound binder.

Each experiment dir gets:
    experiments/exp_NNN/bible.pdf

Self-contained — embeds all images. Typical size ~10-30 MB depending on
how many shots and references are present.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image as RLImage,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from prepare import EXPERIMENTS_DIR, Experiment, LOSS_WEIGHTS


# ============================================================================
# STYLE — film-production-document aesthetic
# ============================================================================
PAGE_W, PAGE_H = LETTER
MARGIN = 0.75 * inch

# Color palette: muted, document-like — this is a working bible, not a poster.
COL_INK         = colors.HexColor("#1a1a1a")
COL_PAPER       = colors.HexColor("#fafafa")
COL_ACCENT      = colors.HexColor("#7a3a2e")  # rust — for slug lines & section labels
COL_SUBTLE      = colors.HexColor("#6b6b6b")
COL_RULE        = colors.HexColor("#cccccc")
COL_TABLE_HEAD  = colors.HexColor("#e8e3dc")
COL_GOOD        = colors.HexColor("#3a6b3a")
COL_BAD         = colors.HexColor("#8b3a3a")


def _styles() -> dict[str, ParagraphStyle]:
    """Build a film-bible style sheet on top of ReportLab's defaults."""
    base = getSampleStyleSheet()
    s: dict[str, ParagraphStyle] = {}

    s["title"] = ParagraphStyle(
        "BibleTitle", parent=base["Title"],
        fontName="Helvetica-Bold", fontSize=36, leading=42,
        textColor=COL_INK, alignment=TA_CENTER, spaceAfter=8,
    )
    s["subtitle"] = ParagraphStyle(
        "BibleSubtitle", parent=base["Title"],
        fontName="Helvetica", fontSize=12, leading=16,
        textColor=COL_SUBTLE, alignment=TA_CENTER, spaceAfter=4,
    )
    s["h1"] = ParagraphStyle(
        "BibleH1", parent=base["Heading1"],
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=COL_INK, spaceBefore=18, spaceAfter=10,
    )
    s["h2"] = ParagraphStyle(
        "BibleH2", parent=base["Heading2"],
        fontName="Helvetica-Bold", fontSize=15, leading=18,
        textColor=COL_ACCENT, spaceBefore=14, spaceAfter=6,
    )
    s["h3"] = ParagraphStyle(
        "BibleH3", parent=base["Heading3"],
        fontName="Helvetica-Bold", fontSize=11, leading=14,
        textColor=COL_INK, spaceBefore=8, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "BibleBody", parent=base["BodyText"],
        fontName="Helvetica", fontSize=10, leading=14,
        textColor=COL_INK, alignment=TA_JUSTIFY, spaceAfter=6,
    )
    s["small"] = ParagraphStyle(
        "BibleSmall", parent=base["BodyText"],
        fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=COL_SUBTLE,
    )
    s["meta"] = ParagraphStyle(
        "BibleMeta", parent=base["BodyText"],
        fontName="Helvetica", fontSize=9, leading=12,
        textColor=COL_SUBTLE, alignment=TA_CENTER,
    )
    s["slug"] = ParagraphStyle(
        "BibleSlug", parent=base["BodyText"],
        fontName="Courier-Bold", fontSize=11, leading=14,
        textColor=COL_ACCENT, spaceBefore=10, spaceAfter=6,
    )
    s["action"] = ParagraphStyle(
        "BibleAction", parent=base["BodyText"],
        fontName="Courier", fontSize=10, leading=14,
        textColor=COL_INK, alignment=TA_LEFT, spaceAfter=6,
    )
    s["character"] = ParagraphStyle(
        "BibleCharacter", parent=base["BodyText"],
        fontName="Courier-Bold", fontSize=10, leading=12,
        textColor=COL_INK, alignment=TA_CENTER, leftIndent=2.0 * inch,
        spaceBefore=4,
    )
    s["paren"] = ParagraphStyle(
        "BibleParen", parent=base["BodyText"],
        fontName="Courier-Oblique", fontSize=9, leading=11,
        textColor=COL_SUBTLE, alignment=TA_LEFT, leftIndent=1.5 * inch,
    )
    s["dialogue"] = ParagraphStyle(
        "BibleDialogue", parent=base["BodyText"],
        fontName="Courier", fontSize=10, leading=14,
        textColor=COL_INK, alignment=TA_LEFT,
        leftIndent=1.0 * inch, rightIndent=1.0 * inch, spaceAfter=4,
    )
    s["scene_label"] = ParagraphStyle(
        "BibleSceneLabel", parent=base["BodyText"],
        fontName="Helvetica-Bold", fontSize=12, leading=14,
        textColor=COL_INK, spaceBefore=14, spaceAfter=6,
    )
    s["toc1"] = ParagraphStyle(
        "TOC1", parent=base["BodyText"],
        fontName="Helvetica-Bold", fontSize=11, leading=15,
        textColor=COL_INK, leftIndent=0,
    )
    s["toc2"] = ParagraphStyle(
        "TOC2", parent=base["BodyText"],
        fontName="Helvetica", fontSize=10, leading=13,
        textColor=COL_SUBTLE, leftIndent=14,
    )
    return s


# ============================================================================
# DOCUMENT TEMPLATE — running header & footer with experiment id and page #
# ============================================================================
class BibleDocTemplate(BaseDocTemplate):
    """Custom doc template: cover page (no header) + body pages with headers."""

    def __init__(self, filename: str, exp_id: str, film_title: str, **kw: Any):
        super().__init__(filename, pagesize=LETTER, **kw)
        self.exp_id = exp_id
        self.film_title = film_title
        self._toc_entries: list[tuple[int, str, int]] = []  # (level, text, page)

        # Two templates: cover (no header) and body (with header).
        frame = Frame(MARGIN, MARGIN, PAGE_W - 2 * MARGIN, PAGE_H - 2 * MARGIN,
                      id="normal", showBoundary=0)
        self.addPageTemplates([
            PageTemplate(id="Cover", frames=[frame], onPage=self._cover_page),
            PageTemplate(id="Body",  frames=[frame], onPage=self._body_page),
        ])

    def _cover_page(self, canvas, doc):
        # Bare cover. No header, no footer. Watermark-style "PRODUCTION BIBLE"
        # sideways on the right edge.
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(COL_SUBTLE)
        canvas.translate(PAGE_W - 0.4 * inch, PAGE_H / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, "PRODUCTION BIBLE")
        canvas.restoreState()

    def _body_page(self, canvas, doc):
        canvas.saveState()
        # Header: film title left, experiment id right, thin rule below.
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(COL_SUBTLE)
        canvas.drawString(MARGIN, PAGE_H - MARGIN + 14,
                          f"{self.film_title} · production bible")
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - MARGIN + 14, self.exp_id)
        canvas.setStrokeColor(COL_RULE)
        canvas.setLineWidth(0.4)
        canvas.line(MARGIN, PAGE_H - MARGIN + 8, PAGE_W - MARGIN, PAGE_H - MARGIN + 8)
        # Footer: page number centered.
        canvas.setFont("Helvetica", 8.5)
        canvas.setFillColor(COL_SUBTLE)
        canvas.drawCentredString(PAGE_W / 2, MARGIN - 18, f"— {doc.page} —")
        canvas.restoreState()


# ============================================================================
# SECTION BUILDERS — each returns a list of flowables
# ============================================================================
def _cover(s: dict, exp: Experiment, script: dict, metric: dict | None) -> list:
    """Cover page: title + optional credits + style frame + film_loss tile."""
    out: list = []
    out.append(Spacer(1, 0.5 * inch))
    out.append(Paragraph(script.get("title", "Untitled"), s["title"]))
    out.append(Paragraph(
        script.get("source", ""),
        s["subtitle"],
    ))
    out.append(Spacer(1, 0.15 * inch))
    out.append(Paragraph("Production Bible", ParagraphStyle(
        "CoverLabel", fontName="Helvetica-Bold", fontSize=12,
        leading=16, textColor=COL_ACCENT, alignment=TA_CENTER, spaceAfter=4,
    )))

    # Optional director / DP credits — only shown when the lookbook has them.
    lookbook = exp.read_json("lookbook.json") if exp.has("lookbook.json") else {}
    director = lookbook.get("director", "")
    dp = lookbook.get("cinematographer", "")
    if director or dp:
        credit_style = ParagraphStyle(
            "CoverCredits", fontName="Helvetica", fontSize=10,
            leading=14, textColor=COL_SUBTLE, alignment=TA_CENTER, spaceAfter=2,
        )
        out.append(Spacer(1, 0.1 * inch))
        if director:
            out.append(Paragraph(
                f"a film directed by <b><font color='{COL_INK.hexval()}'>"
                f"{_escape(director)}</font></b>", credit_style))
        if dp:
            out.append(Paragraph(
                f"director of photography <b><font color='{COL_INK.hexval()}'>"
                f"{_escape(dp)}</font></b>", credit_style))

    # Style-frame hero image if available — sized to leave room for the tile.
    sf = exp.path("lookbook/style_frame.png")
    if sf.exists():
        out.append(Spacer(1, 0.2 * inch))
        out.append(_fitted_image(sf, max_w=5.5 * inch, max_h=2.6 * inch))

    out.append(Spacer(1, 0.25 * inch))

    # film_loss tile.
    if metric:
        out.append(_loss_tile(s, metric))
        out.append(Spacer(1, 0.2 * inch))

    # Experiment metadata at bottom.
    today = datetime.now().strftime("%B %d, %Y")
    n_scenes = len(script.get("scenes", []))
    n_chars = len(script.get("characters", []))
    meta = (
        f"<b>{exp.exp_id}</b> · generated {today}<br/>"
        f"{n_scenes} scenes · {n_chars} characters"
    )
    out.append(Paragraph(meta, s["meta"]))

    out.append(NextPageTemplate("Body"))
    out.append(PageBreak())
    return out


def _loss_tile(s: dict, metric: dict) -> Table:
    """The film_loss summary box on the cover."""
    score = metric.get("film_loss", 0.0)
    color = COL_GOOD if score < 0.30 else (COL_INK if score < 0.45 else COL_BAD)
    big = Paragraph(
        f"<font color='{color.hexval()}'><b>{score:.3f}</b></font>",
        ParagraphStyle("Big", fontName="Helvetica-Bold", fontSize=34,
                       leading=38, alignment=TA_CENTER, textColor=color),
    )
    label = Paragraph(
        "film_loss<br/><font size='8' color='#6b6b6b'>"
        "weighted critic score · lower is better · "
        f"{len(metric.get('changes', []))} suggested fixes</font>",
        ParagraphStyle("LossLabel", fontName="Helvetica", fontSize=10,
                       leading=13, alignment=TA_CENTER, textColor=COL_INK),
    )
    # Build a small per-axis bar chart as a Table.
    rows = [[
        Paragraph(f"<b>{axis}</b>", s["small"]),
        _hbar(metric["scores"][axis]),
        Paragraph(f"{metric['scores'][axis]:.3f}", s["small"]),
    ] for axis in LOSS_WEIGHTS]
    bars = Table(rows, colWidths=[1.3 * inch, 2.4 * inch, 0.55 * inch])
    bars.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))

    inner = Table([[big], [label], [Spacer(1, 4)], [bars]], colWidths=[4.6 * inch])
    inner.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    ]))

    box = Table([[inner]], colWidths=[5.0 * inch])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1.0, COL_RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("BACKGROUND", (0, 0), (-1, -1), COL_PAPER),
    ]))
    return box


def _hbar(value: float, width: float = 2.6 * inch) -> Table:
    """A single horizontal bar showing a 0-1 score, drawn as a thin Table."""
    # Cap to [0, 1].
    v = max(0.0, min(1.0, float(value)))
    fill_w = max(0.02, v) * width
    rest_w = width - fill_w
    color = COL_GOOD if v < 0.30 else (COL_INK if v < 0.50 else COL_BAD)
    bar = Table([["", ""]], colWidths=[fill_w, rest_w], rowHeights=[8])
    bar.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), color),
        ("BACKGROUND", (1, 0), (1, 0), COL_RULE),
        ("LINEBELOW", (0, 0), (-1, 0), 0, COL_RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return bar


def _toc(s: dict) -> list:
    """Generate a Table of Contents flowable that auto-populates."""
    toc = TableOfContents()
    toc.levelStyles = [s["toc1"], s["toc2"]]
    return [
        Paragraph("Contents", s["h1"]),
        Spacer(1, 6),
        toc,
        PageBreak(),
    ]


def _section_heading(s: dict, doc: BibleDocTemplate, text: str, level: int = 0) -> Paragraph:
    """Heading that registers itself with the TOC."""
    style = s["h1"] if level == 0 else s["h2"]
    p = Paragraph(text, style)
    # Register with TOC. notify() is the magic hook ReportLab uses.
    p._bible_toc_text = text
    p._bible_toc_level = level
    return p


def _fitted_image(path: Path, max_w: float, max_h: float) -> RLImage:
    """Image scaled to fit within (max_w, max_h) preserving aspect ratio."""
    img = RLImage(str(path))
    iw, ih = img.imageWidth, img.imageHeight
    scale = min(max_w / iw, max_h / ih)
    img.drawWidth = iw * scale
    img.drawHeight = ih * scale
    return img


def _lookbook_section(s: dict, doc: BibleDocTemplate, exp: Experiment) -> list:
    if not exp.has("lookbook.json"):
        return []
    lb = exp.read_json("lookbook.json")
    out: list = [_section_heading(s, doc, "1. Look Book")]

    # Style frame.
    sf = exp.path("lookbook/style_frame.png")
    if sf.exists():
        out.append(_fitted_image(sf, max_w=6.7 * inch, max_h=3.6 * inch))
        out.append(Paragraph(
            f"<i>Style frame — {lb.get('grade_description', '')}</i>", s["small"]))
        out.append(Spacer(1, 10))

    # Style spec table. Director / DP rows only appear when set in lookbook.
    rows: list = []
    if lb.get("director"):
        rows.append(["Director", lb["director"]])
    if lb.get("cinematographer"):
        rows.append(["Cinematographer", lb["cinematographer"]])
    rows.extend([
        ["Era",            lb.get("era", "—")],
        ["Genre",          lb.get("genre", "—")],
        ["Tone",           lb.get("tone", "—")],
        ["Lens package",   lb.get("lens_package", "—")],
        ["Lighting style", lb.get("lighting_style", "—")],
        ["Grade",          lb.get("grade_description", "—")],
        ["References",     ", ".join(lb.get("reference_films", []) or [])],
        ["Style keywords", ", ".join(lb.get("style_keywords", []) or [])],
    ])
    tbl = Table(
        [[Paragraph(f"<b>{k}</b>", s["small"]),
          Paragraph(_escape(str(v)), s["small"])] for k, v in rows],
        colWidths=[1.5 * inch, 5.2 * inch],
    )
    tbl.setStyle(TableStyle([
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW",   (0, 0), (-1, -1), 0.3, COL_RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",(0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
    ]))
    out.append(tbl)

    # ffmpeg grade as a code block. Keep heading + code together.
    grade = lb.get("ffmpeg_grade", "")
    if grade:
        from reportlab.platypus import KeepTogether
        out.append(Spacer(1, 12))
        grade_block = [
            Paragraph("<b>Color grade (ffmpeg filter chain)</b>", s["small"]),
            Spacer(1, 4),
            Paragraph(
                f"<font name='Courier' size='8'>{_escape(grade)}</font>",
                ParagraphStyle("grade", fontName="Courier", fontSize=8, leading=11,
                               textColor=COL_INK, leftIndent=8,
                               backColor=COL_PAPER, borderColor=COL_RULE,
                               borderWidth=0.4, borderPadding=6, spaceAfter=4),
            ),
        ]
        out.append(KeepTogether(grade_block))

    out.append(PageBreak())
    return out


def _cast_section(s: dict, doc: BibleDocTemplate, exp: Experiment) -> list:
    if not exp.has("cast.json") or not exp.has("script.json"):
        return []
    cast = exp.read_json("cast.json")
    script = exp.read_json("script.json")
    char_by_id = {c["id"]: c for c in script.get("characters", [])}

    out: list = [_section_heading(s, doc, "2. Cast")]

    for row in cast:
        cid = row["character_id"]
        char = char_by_id.get(cid, {"name": cid, "description": ""})

        # Find a representative reference image (first scene where char appears).
        ref_imgs = sorted(exp.path(f"references/{cid}").glob("*.png"))
        ref_img = ref_imgs[0] if ref_imgs else None

        left_cells = [
            Paragraph(f"<b>{_escape(char.get('name', cid))}</b>",
                      ParagraphStyle("CastName", fontName="Helvetica-Bold",
                                     fontSize=13, leading=16, textColor=COL_INK)),
            Paragraph(f"<i>played by {_escape(row.get('actor', '—'))}</i>",
                      ParagraphStyle("CastActor", fontName="Helvetica-Oblique",
                                     fontSize=10, leading=13, textColor=COL_ACCENT)),
            Spacer(1, 4),
            Paragraph(f"<b>Description.</b> {_escape(char.get('description', '—'))}",
                      s["small"]),
        ]
        if char.get("arc"):
            left_cells.append(Paragraph(f"<b>Arc.</b> {_escape(char['arc'])}", s["small"]))
        if row.get("rationale"):
            left_cells.append(Paragraph(
                f"<b>Casting rationale.</b> {_escape(row['rationale'])}", s["small"]))
        if row.get("alternative"):
            left_cells.append(Paragraph(
                f"<b>Alternate.</b> {_escape(row['alternative'])}", s["small"]))

        right_cell = (_fitted_image(ref_img, max_w=2.0 * inch, max_h=2.6 * inch)
                       if ref_img else Paragraph("(no reference)", s["small"]))

        card = Table(
            [[left_cells, right_cell]],
            colWidths=[4.5 * inch, 2.2 * inch],
        )
        card.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW",    (0, 0), (-1, -1), 0.3, COL_RULE),
            ("LEFTPADDING",  (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING",   (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 12),
        ]))
        from reportlab.platypus import KeepTogether
        out.append(KeepTogether(card))

    out.append(PageBreak())
    return out


def _locations_section(s: dict, doc: BibleDocTemplate, exp: Experiment) -> list:
    if not exp.has("locations.json"):
        return []
    locations = exp.read_json("locations.json")
    out: list = [_section_heading(s, doc, "3. Locations")]

    for loc in locations:
        out.append(Paragraph(_escape(loc.get("name", loc["slug"])), s["h2"]))
        out.append(Paragraph(_escape(loc.get("description", "")), s["body"]))

        # Palette.
        if loc.get("color_palette"):
            out.append(Paragraph(
                f"<b>Palette:</b> {', '.join(loc['color_palette'])}", s["small"]))

        # Moodboard images, side by side (up to 3).
        mbs = [Path(p) for p in loc.get("moodboard_paths", []) if Path(p).exists()][:3]
        if mbs:
            cols = []
            for p in mbs:
                cols.append(_fitted_image(p, max_w=2.1 * inch, max_h=1.4 * inch))
            grid = Table([cols], colWidths=[2.3 * inch] * len(cols))
            grid.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            out.append(Spacer(1, 6))
            out.append(grid)

        scene_ids = loc.get("scene_ids", [])
        if scene_ids:
            out.append(Paragraph(
                f"<b>Used in scenes:</b> {', '.join(scene_ids)}", s["small"]))
        out.append(Spacer(1, 14))

    out.append(PageBreak())
    return out


def _screenplay_section(s: dict, doc: BibleDocTemplate, exp: Experiment) -> list:
    """Properly formatted screenplay: slug / action / character / dialogue / paren."""
    if not exp.has("script.json"):
        return []
    script = exp.read_json("script.json")
    out: list = [_section_heading(s, doc, "4. Screenplay")]

    for scene in script.get("scenes", []):
        # Slug line.
        slug = (
            f"INT./EXT. {scene.get('location', '').upper()} - "
            f"{scene.get('time_of_day', '').upper()}    [{scene['id']}]"
        )
        out.append(Paragraph(_escape(slug), s["slug"]))

        elements = scene.get("elements") or []
        if elements:
            for el in elements:
                t = el.get("type", "action")
                text = _escape(el.get("text", ""))
                if t == "action":
                    out.append(Paragraph(text, s["action"]))
                elif t == "dialogue":
                    char = el.get("character", "").upper()
                    if char:
                        out.append(Paragraph(_escape(char), s["character"]))
                    if el.get("parenthetical"):
                        out.append(Paragraph(
                            f"({_escape(el['parenthetical'])})", s["paren"]))
                    out.append(Paragraph(text, s["dialogue"]))
                elif t == "transition":
                    out.append(Paragraph(text, s["paren"]))
        else:
            # No expanded screenplay — fall back to summary + dialogue excerpts.
            out.append(Paragraph(_escape(scene.get("summary", "")), s["action"]))
            for line in scene.get("dialogue_excerpts", []) or []:
                cid = line.get("character_id", "")
                out.append(Paragraph(cid.upper(), s["character"]))
                out.append(Paragraph(_escape(line.get("line", "")), s["dialogue"]))
        out.append(Spacer(1, 8))

    out.append(PageBreak())
    return out


def _storyboard_section(s: dict, doc: BibleDocTemplate, exp: Experiment) -> list:
    """Per-scene shot list with B&W panel + first frame side by side."""
    if not exp.has("storyboard.json") or not exp.has("script.json"):
        return []
    storyboard = exp.read_json("storyboard.json")
    script = exp.read_json("script.json")
    scene_meta = {sc["id"]: sc for sc in script.get("scenes", [])}
    edl = exp.read_json("edl.json") if exp.has("edl.json") else {"decisions": []}
    decision_by_shot = {(d["scene_id"], d["shot_id"]): d
                        for d in edl.get("decisions", [])}

    # Routing plan — added in the shot-planning branch. Optional: older
    # experiments don't have this, in which case the Model column is empty.
    shot_plan = exp.read_json("shot_plan.json") if exp.has("shot_plan.json") else {}

    out: list = [_section_heading(s, doc, "5. Storyboard & Shot List")]

    # --- Aggregate render-cost summary at top of section, if available ---
    agg = shot_plan.get("_aggregate") if isinstance(shot_plan, dict) else None
    if agg:
        n_shots = sum(len(v) for v in storyboard.values())
        by_model = agg.get("shots_by_model", {})
        model_breakdown = ", ".join(
            f"{n} on {k.replace('_', ' ')}" for k, n in by_model.items()
        ) or "—"
        summary_rows = [
            ["Total shots",        f"{n_shots}"],
            ["Total runtime",      f"{agg.get('total_seconds', 0)}s "
                                   f"(~{agg.get('total_seconds', 0) / 60:.1f} min)"],
            ["Estimated render",   f"${agg.get('estimated_cost_usd', 0):.2f} USD"],
            ["Model distribution", model_breakdown],
        ]
        summary = Table(
            [[Paragraph(f"<b>{k}</b>", s["small"]),
              Paragraph(_escape(str(v)), s["small"])] for k, v in summary_rows],
            colWidths=[1.6 * inch, 5.1 * inch],
        )
        summary.setStyle(TableStyle([
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW",    (0, 0), (-1, -1), 0.3, COL_RULE),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
            ("BACKGROUND",   (0, 0), (-1, -1), COL_PAPER),
        ]))
        out.append(summary)
        out.append(Spacer(1, 12))

    for scene_id, shots in storyboard.items():
        scene = scene_meta.get(scene_id, {})
        out.append(Paragraph(
            f"Scene {scene_id} — {_escape(scene.get('location', ''))}, "
            f"{_escape(scene.get('time_of_day', ''))}",
            s["scene_label"],
        ))
        if scene.get("summary"):
            out.append(Paragraph(_escape(scene["summary"]), s["small"]))
            out.append(Spacer(1, 6))

        for shot in shots:
            shot_id = shot["shot_id"]
            decision = decision_by_shot.get((scene_id, shot_id), {})
            route = shot_plan.get(scene_id, {}).get(shot_id) if isinstance(shot_plan, dict) else None

            # Panel + first frame side by side.
            panel = exp.path(f"storyboard/{scene_id}/{shot_id}.png")
            frame = exp.path(f"frames/{scene_id}/{shot_id}.png")
            row_imgs: list = []
            if panel.exists():
                row_imgs.append(_fitted_image(panel, max_w=3.0 * inch, max_h=1.7 * inch))
            else:
                row_imgs.append(Paragraph("(no panel)", s["small"]))
            if frame.exists():
                row_imgs.append(_fitted_image(frame, max_w=3.0 * inch, max_h=1.7 * inch))
            else:
                row_imgs.append(Paragraph("(no frame)", s["small"]))

            img_row = Table([row_imgs], colWidths=[3.2 * inch, 3.2 * inch])
            img_row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]))

            # Shot specs table beneath the images. New "Model" row when
            # routing data is present — the most useful single fact a DP
            # would want when looking at the shot list.
            specs = [
                ("Shot",     shot_id),
                ("Size",     shot.get("shot_size", "—")),
                ("Angle",    shot.get("angle", "—")),
                ("Lens",     f"{shot.get('lens_mm', '—')}mm"),
                ("Move",     shot.get("camera_move", "—")),
                ("Duration", f"{shot.get('duration_seconds', '—')}s"),
            ]
            if route:
                model_label = route["model_key"].replace("_", " ")
                segs = route["segments"]
                if len(segs) > 1:
                    model_label += f" · {len(segs)} segs"
                if shot.get("is_oner"):
                    model_label += " · oner"
                specs.append(("Model", model_label))
            spec_cells = [[Paragraph(f"<b>{k}</b>", s["small"]),
                           Paragraph(_escape(str(v)), s["small"])] for k, v in specs]
            spec_table = Table(spec_cells, colWidths=[0.6 * inch, 1.3 * inch])
            spec_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
            ]))

            # Right side: action / composition / dialogue / chosen take.
            details: list = [
                Paragraph(f"<b>Action.</b> {_escape(shot.get('action', ''))}", s["small"]),
                Paragraph(f"<b>Composition.</b> {_escape(shot.get('composition_notes', ''))}",
                          s["small"]),
            ]
            if shot.get("dialogue_excerpt"):
                details.append(Paragraph(
                    f"<b>Dialogue.</b> &ldquo;{_escape(shot['dialogue_excerpt'])}&rdquo;",
                    s["small"]))
            if route and route.get("rationale"):
                details.append(Paragraph(
                    f"<b>Routing.</b> <font name='Courier' size='8'>"
                    f"{_escape(route['rationale'])}</font> "
                    f"(~${route['estimated_cost']:.2f})",
                    s["small"]))
            if decision:
                t = decision.get("chosen_take", 1)
                details.append(Paragraph(
                    f"<b>EDL.</b> chosen take <b>{t}</b>"
                    + (f" — {_escape(decision.get('rationale', ''))}"
                       if decision.get('rationale') else ""),
                    s["small"]))

            spec_row = Table([[spec_table, details]],
                             colWidths=[2.0 * inch, 4.4 * inch])
            spec_row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]))

            shot_block = Table(
                [[img_row], [spec_row]],
                colWidths=[6.7 * inch],
            )
            shot_block.setStyle(TableStyle([
                ("BOX",          (0, 0), (-1, -1), 0.4, COL_RULE),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",   (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
            ]))
            from reportlab.platypus import KeepTogether
            out.append(KeepTogether([shot_block, Spacer(1, 8)]))

    out.append(PageBreak())
    return out


def _music_section(s: dict, doc: BibleDocTemplate, exp: Experiment) -> list:
    if not exp.has("script.json"):
        return []
    script = exp.read_json("script.json")
    out: list = [_section_heading(s, doc, "6. Music & Sound")]

    rows = [["Scene", "Location / Mood", "Music cue", "Ambient bed"]]
    for scene in script["scenes"]:
        sid = scene["id"]
        music = exp.path(f"music/{sid}.wav")
        ambient = exp.path(f"sfx/{sid}/ambient.wav")
        rows.append([
            sid,
            f"{scene.get('location', '')[:30]} — {scene.get('mood', '')}",
            "✓" if music.exists() else "—",
            "✓" if ambient.exists() else "—",
        ])
    tbl = Table(rows, colWidths=[0.9 * inch, 3.4 * inch, 1.2 * inch, 1.2 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), COL_TABLE_HEAD),
        ("FONTNAME",    (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("LINEBELOW",   (0, 0), (-1, -1), 0.3, COL_RULE),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",  (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
    ]))
    out.append(tbl)
    out.append(Spacer(1, 10))
    out.append(Paragraph(
        "<i>Audio files are referenced by path; play them from "
        "<code>music/</code> and <code>sfx/</code> in the experiment directory. "
        "Veo 3.1 also renders synchronized native dialogue + scene-specific FX "
        "into each shot clip.</i>",
        s["small"]))
    out.append(PageBreak())
    return out


def _prompts_section(s: dict, doc: BibleDocTemplate, exp: Experiment) -> list:
    """Render prompts.json grouped by model.

    Each entry shows target (artifact path), optional metadata, and the
    full prompt in monospace. The section is tolerant of long prompts —
    they wrap normally and KeepTogether is left off so prompts can break
    across pages if needed.
    """
    if not exp.has("prompts.json"):
        return []
    prompts: dict = exp.read_json("prompts.json")
    if not prompts:
        return []

    # Group by model. Keep insertion order within a group so a reader sees
    # the same sequence the pipeline used.
    by_model: dict[str, list[tuple[str, dict]]] = {}
    for target, entry in prompts.items():
        model = entry.get("model", "unknown")
        by_model.setdefault(model, []).append((target, entry))

    # Friendly display names — keep a stable order: image → video → audio.
    model_order = [
        "gpt-image-2",
        "gemini-3.1-flash-image-preview",
        "veo-3.1-lite-generate-preview",
        "veo-3.1-fast-generate-preview",
        "veo-3.1-generate-preview",
        "stable-audio-2.5",
        "elevenlabs-sfx",
    ]
    display_name = {
        "gpt-image-2": "GPT Image 2",
        "gemini-3.1-flash-image-preview": "Nano Banana 2 (Gemini 3.1 Flash Image)",
        "veo-3.1-lite-generate-preview": "Veo 3.1 Lite",
        "veo-3.1-fast-generate-preview": "Veo 3.1 Fast",
        "veo-3.1-generate-preview": "Veo 3.1 Standard",
        "stable-audio-2.5": "Stable Audio 2.5",
        "elevenlabs-sfx": "ElevenLabs SFX",
    }

    out: list = [_section_heading(s, doc, "7. Prompts")]
    out.append(Paragraph(
        "<i>The exact text sent to each generative model, grouped by model. "
        "Use this to debug stylistic drift, audit what the agent told each "
        "model on a particular run, or copy a prompt to iterate on by hand.</i>",
        s["small"],
    ))
    out.append(Spacer(1, 8))

    # Aggregate count table.
    summary_rows = [["Model", "Calls"]]
    for m in model_order:
        if m in by_model:
            summary_rows.append([display_name.get(m, m), str(len(by_model[m]))])
    for m in by_model:
        if m not in model_order:
            summary_rows.append([display_name.get(m, m), str(len(by_model[m]))])
    summary = Table(summary_rows, colWidths=[4.5 * inch, 1.0 * inch])
    summary.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), COL_TABLE_HEAD),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("LINEBELOW",    (0, 0), (-1, -1), 0.3, COL_RULE),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("ALIGN",        (1, 0), (1, -1), "RIGHT"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    out.append(summary)
    out.append(Spacer(1, 14))

    # Per-model subsections.
    monospace = ParagraphStyle(
        "PromptBody", fontName="Courier", fontSize=7.5, leading=10,
        textColor=COL_INK, leftIndent=4, rightIndent=4,
        spaceBefore=2, spaceAfter=2,
    )
    target_label = ParagraphStyle(
        "PromptTarget", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
        textColor=COL_ACCENT, spaceBefore=8, spaceAfter=2,
    )
    meta_label = ParagraphStyle(
        "PromptMeta", fontName="Helvetica-Oblique", fontSize=7.5, leading=10,
        textColor=COL_SUBTLE, spaceAfter=4,
    )

    ordered_models = [m for m in model_order if m in by_model] + [
        m for m in by_model if m not in model_order
    ]

    for sub_idx, m in enumerate(ordered_models, start=1):
        out.append(Paragraph(
            f"7.{sub_idx} {display_name.get(m, m)} "
            f"<font color='{COL_SUBTLE.hexval()}' size='8'>"
            f"<i>{_escape(m)}</i></font>",
            ParagraphStyle("PromptModelHead", fontName="Helvetica-Bold",
                           fontSize=12, leading=16, textColor=COL_INK,
                           spaceBefore=10, spaceAfter=6),
        ))

        for target, entry in by_model[m]:
            prompt_text = entry.get("prompt", "")
            meta = entry.get("meta", {})

            # Target as the visual anchor.
            out.append(Paragraph(
                f"→ <font name='Courier'>{_escape(target)}</font>",
                target_label,
            ))

            # One-line meta if present (size, duration, scene/shot/take, etc.)
            if meta:
                bits = []
                for k, v in meta.items():
                    if k in ("stage",):
                        continue
                    bits.append(f"{k}={v}")
                if bits:
                    out.append(Paragraph(
                        " · ".join(_escape(b) for b in bits),
                        meta_label,
                    ))

            # The prompt body — monospace, wrap-friendly.
            # Preserve newlines by replacing them with <br/>, and escape
            # ReportLab markup as we go.
            escaped = _escape(prompt_text)
            escaped = escaped.replace("\n", "<br/>")
            out.append(Paragraph(escaped, monospace))

    out.append(PageBreak())
    return out


def _critique_section(s: dict, doc: BibleDocTemplate, exp: Experiment) -> list:
    if not exp.has("metric.json"):
        return []
    metric = exp.read_json("metric.json")
    out: list = [_section_heading(s, doc, "8. Critic's Report")]

    # Per-axis bar chart — same as cover but with full prose.
    score_rows = [["Axis", "Score", "", "Weight", "Contribution"]]
    for axis, weight in LOSS_WEIGHTS.items():
        v = metric["scores"][axis]
        score_rows.append([
            axis,
            f"{v:.3f}",
            _hbar(v, width=2.0 * inch),
            f"{weight:.2f}",
            f"{v * weight:.3f}",
        ])
    score_rows.append([
        Paragraph("<b>film_loss</b>", s["small"]),
        "",
        "",
        "",
        Paragraph(f"<b>{metric['film_loss']:.3f}</b>", s["small"]),
    ])
    tbl = Table(score_rows, colWidths=[1.2 * inch, 0.7 * inch, 2.2 * inch,
                                       0.8 * inch, 1.0 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), COL_TABLE_HEAD),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("LINEBELOW",    (0, 0), (-1, -1), 0.3, COL_RULE),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("BACKGROUND",   (0, -1), (-1, -1), COL_PAPER),
        ("LINEABOVE",    (0, -1), (-1, -1), 0.7, COL_INK),
    ]))
    out.append(tbl)

    # Per-reviewer breakdown.
    reviewers = metric.get("reviewers", {})
    if reviewers:
        out.append(Spacer(1, 12))
        out.append(Paragraph("Per-reviewer scores", s["h3"]))
        rev_rows = [["Axis", "Gemini 3 Pro (video)", "Claude Opus 4.7 (stills)"]]
        for axis in LOSS_WEIGHTS:
            g = reviewers.get("gemini_scores", {}).get(axis, "—")
            c = reviewers.get("claude_scores", {}).get(axis, "—")
            rev_rows.append([axis,
                             f"{g:.3f}" if isinstance(g, (int, float)) else str(g),
                             f"{c:.3f}" if isinstance(c, (int, float)) else str(c)])
        rev_tbl = Table(rev_rows, colWidths=[1.5 * inch, 2.4 * inch, 2.4 * inch])
        rev_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), COL_TABLE_HEAD),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("LINEBELOW",    (0, 0), (-1, -1), 0.3, COL_RULE),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        out.append(rev_tbl)

    # Suggested changes.
    changes = metric.get("changes", [])
    if changes:
        out.append(Spacer(1, 16))
        out.append(Paragraph("Suggested changes for next experiment", s["h3"]))
        out.append(Paragraph(
            "Ranked by priority. The agent applies these to <code>produce.py</code>.",
            s["small"]))
        out.append(Spacer(1, 6))
        chg_rows = [["#", "Pri", "Axis", "Target", "Suggested change"]]
        for i, c in enumerate(changes[:20], start=1):
            chg_rows.append([
                str(i),
                Paragraph(f"<b>{c.get('priority', '—')}</b>", s["small"]),
                c.get("axis", "—"),
                Paragraph(f"<font name='Courier' size='8'>{_escape(c.get('target', ''))}</font>",
                          s["small"]),
                Paragraph(_escape(c.get("suggested_change", "")), s["small"]),
            ])
        chg_tbl = Table(chg_rows, colWidths=[0.3 * inch, 0.5 * inch, 1.1 * inch,
                                              1.6 * inch, 3.2 * inch])
        chg_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (-1, 0), COL_TABLE_HEAD),
            ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 8.5),
            ("VALIGN",       (0, 0), (-1, -1), "TOP"),
            ("LINEBELOW",    (0, 0), (-1, -1), 0.3, COL_RULE),
            ("LEFTPADDING",  (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING",   (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        out.append(chg_tbl)

    # Prose critique from critique.md (if available).
    crit_path = exp.path("critique.md")
    if crit_path.exists():
        out.append(Spacer(1, 16))
        out.append(Paragraph("Prose critique", s["h3"]))
        prose = crit_path.read_text()
        for para in _markdown_to_paragraphs(prose, s):
            out.append(para)

    # CLIP drift.
    clip = metric.get("clip_drift") or {}
    if clip:
        out.append(Spacer(1, 12))
        out.append(Paragraph("Character identity drift (CLIP cosine, informational)",
                             s["h3"]))
        drift_rows = [["Character", "Mean similarity", ""]]
        for k, v in sorted(clip.items()):
            drift_rows.append([k, f"{v:.3f}",
                               _hbar(1.0 - max(0.0, min(1.0, v)), width=2.0 * inch)])
        drift_tbl = Table(drift_rows,
                          colWidths=[2.0 * inch, 1.2 * inch, 2.2 * inch])
        drift_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COL_TABLE_HEAD),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE",   (0, 0), (-1, -1), 9),
            ("LINEBELOW",  (0, 0), (-1, -1), 0.3, COL_RULE),
            ("VALIGN",     (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING",(0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ]))
        out.append(drift_tbl)

    return out


# ============================================================================
# UTILITIES
# ============================================================================
def _escape(text: str) -> str:
    """ReportLab Paragraphs use a small XML subset; escape user content."""
    if text is None:
        return ""
    return (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))


def _markdown_to_paragraphs(md: str, s: dict) -> list:
    """Crude markdown → flowables conversion. Handles headings, bullets,
    bold, and italic. Good enough for our critique.md format."""
    out: list = []
    for raw in md.split("\n"):
        line = raw.rstrip()
        if not line.strip():
            out.append(Spacer(1, 4))
            continue
        if line.startswith("### "):
            out.append(Paragraph(_md_inline(line[4:]), s["h3"]))
        elif line.startswith("## "):
            out.append(Paragraph(_md_inline(line[3:]), s["h3"]))
        elif line.startswith("# "):
            out.append(Paragraph(_md_inline(line[2:]), s["h2"]))
        elif line.startswith("- ") or line.startswith("* "):
            out.append(Paragraph("• " + _md_inline(line[2:]),
                                 ParagraphStyle("Bullet", parent=s["small"],
                                                leftIndent=14, bulletIndent=4,
                                                spaceAfter=2)))
        else:
            out.append(Paragraph(_md_inline(line), s["small"]))
    return out


def _md_inline(text: str) -> str:
    """Light inline markdown — bold and italic only — for use inside Paragraph."""
    safe = _escape(text)
    # Restore **bold** -> <b>...</b> and *italic* -> <i>...</i>
    import re
    safe = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", safe)
    safe = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", safe)
    safe = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", safe)
    return safe


# ============================================================================
# TOC INTEGRATION — wire heading flowables into the TOC entries.
# ============================================================================
class _TOCAwareDocTemplate(BibleDocTemplate):
    """Subclass that registers headings on afterFlowable so the TOC populates."""

    def afterFlowable(self, flowable):
        if hasattr(flowable, "_bible_toc_text"):
            level = flowable._bible_toc_level
            text = flowable._bible_toc_text
            self.notify("TOCEntry", (level, text, self.page))


# ============================================================================
# Public entry point
# ============================================================================
def build_bible(exp: Experiment) -> Path:
    """Render the production bible PDF for one experiment.

    Returns the output path. Overwrites any existing bible.pdf.
    """
    if not exp.has("script.json"):
        raise RuntimeError(f"{exp.exp_id}: cannot build bible — script.json missing")

    script = exp.read_json("script.json")
    metric = exp.read_json("metric.json") if exp.has("metric.json") else None

    out_path = exp.path("bible.pdf")
    s = _styles()

    doc = _TOCAwareDocTemplate(
        str(out_path),
        exp_id=exp.exp_id,
        film_title=script.get("title", "Untitled"),
        topMargin=MARGIN, bottomMargin=MARGIN,
        leftMargin=MARGIN, rightMargin=MARGIN,
        title=f"{script.get('title', 'Untitled')} — Production Bible",
        author="autofilm",
        subject=f"Experiment {exp.exp_id}",
    )

    story: list = []
    story.extend(_cover(s, exp, script, metric))
    story.extend(_toc(s))
    story.extend(_lookbook_section(s, doc, exp))
    story.extend(_cast_section(s, doc, exp))
    story.extend(_locations_section(s, doc, exp))
    story.extend(_screenplay_section(s, doc, exp))
    story.extend(_storyboard_section(s, doc, exp))
    story.extend(_music_section(s, doc, exp))
    story.extend(_prompts_section(s, doc, exp))
    story.extend(_critique_section(s, doc, exp))

    # multiBuild handles the TOC's two-pass rendering.
    doc.multiBuild(story)
    return out_path


# ============================================================================
# CLI
# ============================================================================
def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_id", nargs="?", help="experiment id, 'latest', or '--all'")
    parser.add_argument("--all", action="store_true",
                        help="generate bibles for every experiment with a script.json")
    args = parser.parse_args()

    if args.all or args.exp_id == "--all":
        targets = sorted(p.name for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
    elif args.exp_id == "latest" or args.exp_id is None:
        all_exps = sorted(p.name for p in EXPERIMENTS_DIR.iterdir() if p.is_dir())
        if not all_exps:
            print("No experiments found.")
            return 1
        targets = [all_exps[-1]]
    else:
        targets = [args.exp_id]

    for exp_id in targets:
        try:
            exp = Experiment.load(exp_id)
            print(f"  {exp_id}: building bible.pdf...")
            path = build_bible(exp)
            size_mb = path.stat().st_size / 1_048_576
            print(f"    ✓ {path.name}  ({size_mb:.1f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"  {exp_id}: failed — {e}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
