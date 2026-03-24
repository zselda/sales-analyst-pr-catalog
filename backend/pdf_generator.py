"""
PDF Report Generator — ING Global Brand Style
================================================
Generates a premium, branded PDF report from the swarm analysis results.
Uses ING brand colors and the official INGMe font family.

Brand identity:
  - ING Orange: RGB (255, 98, 0) = #FF6200
  - ING Navy:   RGB (0, 0, 102) = #000066
  - Font: INGMe (Regular + Bold)

Font setup:
  Place INGMe-Regular.ttf and INGMe-Bold.ttf in the backend/fonts/ directory.
  If font files are not found, falls back to Helvetica.

Dependencies: reportlab
"""

import io
import os
import re
import logging
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether,
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader

logger = logging.getLogger("swarm.pdf")

# ── ING Brand Colors ──
ING_ORANGE = colors.Color(255 / 255, 98 / 255, 0 / 255)       # #FF6200
ING_NAVY = colors.Color(0 / 255, 0 / 255, 102 / 255)          # #000066
ING_ORANGE_LIGHT = colors.Color(255 / 255, 98 / 255, 0 / 255, alpha=0.10)
ING_NAVY_LIGHT = colors.Color(0 / 255, 0 / 255, 102 / 255, alpha=0.08)
ING_GRAY = colors.Color(0.95, 0.95, 0.95)
ING_WHITE = colors.white
ING_TEXT = colors.Color(0.15, 0.15, 0.2)
ING_TEXT_SECONDARY = colors.Color(0.4, 0.4, 0.45)

# ── INGMe Font Registration ──
ASSETS_DIR = Path(__file__).parent / "fonts"
FONTS_DIR = ASSETS_DIR
ING_LOGO_PATH = ASSETS_DIR / "ING_Logo2.png"
FONT_REGULAR = "INGMe"
FONT_BOLD = "INGMe-Bold"

_ingme_available = False

def _register_ingme_fonts():
    """Register INGMe TTF fonts with reportlab if available."""
    global _ingme_available, FONT_REGULAR, FONT_BOLD

    regular_path = FONTS_DIR / "INGMeWeb-Regular.ttf"
    bold_path = FONTS_DIR / "INGMeWeb-Bold.ttf"

    # Check if font files are actual TTF (not HTML redirect pages)
    def _is_valid_ttf(path: Path) -> bool:
        if not path.exists():
            return False
        if path.stat().st_size < 5000:
            return False  # Real TTF files are at least a few KB
        with open(path, 'rb') as f:
            header = f.read(4)
            # TTF files start with 0x00010000 or 'true' or 'OTTO'
            return header in (b'\x00\x01\x00\x00', b'true', b'OTTO')

    if _is_valid_ttf(regular_path) and _is_valid_ttf(bold_path):
        try:
            pdfmetrics.registerFont(TTFont('INGMe', str(regular_path)))
            pdfmetrics.registerFont(TTFont('INGMe-Bold', str(bold_path)))
            _ingme_available = True
            FONT_REGULAR = "INGMe"
            FONT_BOLD = "INGMe-Bold"
            logger.info("[PDF] INGMe font loaded successfully")
            return
        except Exception as e:
            logger.warning(f"[PDF] INGMe font registration failed: {e}")

    elif _is_valid_ttf(regular_path):
        try:
            pdfmetrics.registerFont(TTFont('INGMe', str(regular_path)))
            # Use regular for bold too if bold not available
            pdfmetrics.registerFont(TTFont('INGMe-Bold', str(regular_path)))
            _ingme_available = True
            FONT_REGULAR = "INGMe"
            FONT_BOLD = "INGMe-Bold"
            logger.info("[PDF] INGMe font loaded (regular only, using for bold too)")
            return
        except Exception as e:
            logger.warning(f"[PDF] INGMe font registration failed: {e}")

    # Fallback to Helvetica
    _ingme_available = False
    FONT_REGULAR = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"
    logger.info(
        f"[PDF] INGMe font not found at {FONTS_DIR}. "
        "Using Helvetica fallback. "
        "Place INGMe-Regular.ttf and INGMe-Bold.ttf in backend/fonts/ for ING branding."
    )

# Register fonts at module load
_register_ingme_fonts()


def _build_styles():
    """Create ING-branded paragraph styles using INGMe font."""
    styles = getSampleStyleSheet()

    # White text style for table headers on dark backgrounds
    styles.add(ParagraphStyle(
        name='INGH3White',
        fontName=FONT_BOLD,
        fontSize=10,
        textColor=ING_WHITE,
        spaceBefore=0,
        spaceAfter=0,
        leading=13,
    ))
    styles.add(ParagraphStyle(
        name='INGTitle',
        fontName=FONT_BOLD,
        fontSize=20,
        textColor=ING_NAVY,
        spaceAfter=4,
        leading=24,
    ))
    styles.add(ParagraphStyle(
        name='INGSubtitle',
        fontName=FONT_REGULAR,
        fontSize=10,
        textColor=ING_ORANGE,
        spaceAfter=10,
        leading=13,
    ))
    styles.add(ParagraphStyle(
        name='INGH1',
        fontName=FONT_BOLD,
        fontSize=14,
        textColor=ING_NAVY,
        spaceBefore=6,
        spaceAfter=6,
        leading=17,
    ))
    styles.add(ParagraphStyle(
        name='INGH2',
        fontName=FONT_BOLD,
        fontSize=11,
        textColor=ING_ORANGE,
        spaceBefore=8,
        spaceAfter=4,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        name='INGH3',
        fontName=FONT_BOLD,
        fontSize=9.5,
        textColor=ING_NAVY,
        spaceBefore=6,
        spaceAfter=3,
        leading=12,
    ))
    styles.add(ParagraphStyle(
        name='INGBody',
        fontName=FONT_REGULAR,
        fontSize=8.5,
        textColor=ING_TEXT,
        spaceAfter=4,
        leading=12,
    ))
    styles.add(ParagraphStyle(
        name='INGSmall',
        fontName=FONT_REGULAR,
        fontSize=7.5,
        textColor=ING_TEXT_SECONDARY,
        spaceAfter=3,
        leading=10,
    ))
    styles.add(ParagraphStyle(
        name='INGFooter',
        fontName=FONT_REGULAR,
        fontSize=7,
        textColor=ING_TEXT_SECONDARY,
        alignment=TA_CENTER,
    ))
    return styles


class _INGPageTemplate:
    """Draw ING-branded header/footer on every page."""

    def __init__(self, title: str):
        self.title = title

    def __call__(self, canvas_obj, doc):
        canvas_obj.saveState()
        w, h = A4

        # ── Top bar (ING Orange) ──
        canvas_obj.setFillColor(ING_ORANGE)
        canvas_obj.rect(0, h - 18 * mm, w, 18 * mm, fill=True, stroke=False)

        # ING Logo + Title
        logo_x = 12 * mm
        if ING_LOGO_PATH.exists():
            try:
                logo_h = 12 * mm
                logo_w = logo_h * 2.4  # ING logo aspect ratio ~2.4:1
                canvas_obj.drawImage(
                    str(ING_LOGO_PATH),
                    logo_x, h - 16 * mm,
                    width=logo_w, height=logo_h,
                    preserveAspectRatio=True, mask='auto',
                )
                text_x = logo_x + logo_w + 4 * mm
            except Exception:
                text_x = 15 * mm
        else:
            text_x = 15 * mm

        canvas_obj.setFillColor(ING_WHITE)
        canvas_obj.setFont(FONT_BOLD, 11)
        canvas_obj.drawString(text_x, h - 12 * mm, self.title)

        # Confidential badge
        canvas_obj.setFont(FONT_REGULAR, 7)
        canvas_obj.drawRightString(w - 15 * mm, h - 12 * mm, 'CONFIDENTIAL')

        # ── Bottom bar (ING Navy) ──
        canvas_obj.setFillColor(ING_NAVY)
        canvas_obj.rect(0, 0, w, 10 * mm, fill=True, stroke=False)

        canvas_obj.setFillColor(ING_WHITE)
        canvas_obj.setFont(FONT_REGULAR, 7)
        footer_label = (
            f'Finansal & Satış Analiz Platformu — {datetime.now().strftime("%d %B %Y")}'
            if hasattr(doc, '_ing_language') and doc._ing_language == 'TR'
            else f'Financial & Sales Analysis Platform — {datetime.now().strftime("%d %B %Y")}'
        )
        canvas_obj.drawString(15 * mm, 3.5 * mm, footer_label)
        canvas_obj.drawRightString(w - 15 * mm, 3.5 * mm, f'Page {doc.page}')

        # ── Thin orange accent line under header ──
        canvas_obj.setStrokeColor(ING_ORANGE)
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(0, h - 18 * mm, w, h - 18 * mm)

        canvas_obj.restoreState()


def _parse_markdown_basic(text: str, styles, page_break_on_h1: bool = False) -> list:
    """
    Convert basic Markdown text to reportlab Flowables.
    Handles: #, ##, ###, **, |tables|, -, numbered lists, ---

    If page_break_on_h1 is True, a PageBreak is inserted before each
    top-level heading (# ) so every major section starts on its own page.
    """
    elements = []
    lines = text.strip().split('\n')
    i = 0
    h1_count = 0   # track whether we're past the first H1

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            elements.append(Spacer(1, 3))
            i += 1
            continue

        # Horizontal rule
        if line.startswith('---') or line.startswith('***'):
            elements.append(HRFlowable(width="100%", thickness=0.5, color=ING_ORANGE,
                                        spaceAfter=4, spaceBefore=4))
            i += 1
            continue

        # Headings
        if line.startswith('### '):
            text_content = _clean_md(line[4:])
            elements.append(Paragraph(text_content, styles['INGH3']))
            i += 1
            continue
        if line.startswith('## '):
            text_content = _clean_md(line[3:])
            elements.append(Paragraph(text_content, styles['INGH2']))
            i += 1
            continue
        if line.startswith('# '):
            # Insert a page break before each H1 section (except the first)
            if page_break_on_h1:
                if h1_count > 0:
                    elements.append(PageBreak())
                h1_count += 1
            text_content = _clean_md(line[2:])
            elements.append(Paragraph(text_content, styles['INGH1']))
            elements.append(HRFlowable(width="100%", thickness=1, color=ING_ORANGE,
                                        spaceAfter=4, spaceBefore=2))
            i += 1
            continue

        # Table detection
        if '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
            table_lines = []
            while i < len(lines) and '|' in lines[i]:
                raw = lines[i].strip()
                if '---' not in raw:
                    cells = [c.strip() for c in raw.split('|') if c.strip()]
                    table_lines.append(cells)
                i += 1

            if table_lines:
                elements.append(_build_table(table_lines, styles))
                elements.append(Spacer(1, 4))
            continue

        # Bullet list
        if line.startswith('- ') or line.startswith('• '):
            text_content = _clean_md(line[2:])
            elements.append(Paragraph(f'•  {text_content}', styles['INGBody']))
            i += 1
            continue

        # Numbered list
        if len(line) > 2 and line[0].isdigit() and line[1] in '.):'  :
            text_content = _clean_md(line)
            elements.append(Paragraph(text_content, styles['INGBody']))
            i += 1
            continue

        # Regular paragraph
        text_content = _clean_md(line)
        elements.append(Paragraph(text_content, styles['INGBody']))
        i += 1

    return elements


def _clean_md(text: str) -> str:
    """Convert Markdown formatting to reportlab XML tags."""
    # Bold **text** → <b>text</b>
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    # Italic *text* → <i>text</i>
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    # Code `text` → <font face="Courier">text</font>
    text = re.sub(r'`(.+?)`', r'<font face="Courier" size="8" color="#000066">\1</font>', text)
    return text


def _build_table(rows: list, styles) -> Table:
    """Build a styled ING-branded table from a list of rows."""
    if not rows:
        return Spacer(1, 1)

    # Convert strings to Paragraphs for text wrapping
    table_data = []
    for ri, row in enumerate(rows):
        styled_row = []
        for cell in row:
            style = styles['INGH3White'] if ri == 0 else styles['INGSmall']
            styled_row.append(Paragraph(_clean_md(cell), style))
        table_data.append(styled_row)

    # Calculate column widths
    num_cols = max(len(r) for r in table_data) if table_data else 1
    available_width = A4[0] - 30 * mm
    col_width = available_width / num_cols

    t = Table(table_data, colWidths=[col_width] * num_cols)
    t.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), ING_NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), ING_WHITE),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),

        # Data rows
        ('BACKGROUND', (0, 1), (-1, -1), ING_WHITE),
        ('TEXTCOLOR', (0, 1), (-1, -1), ING_TEXT),
        ('FONTNAME', (0, 1), (-1, -1), FONT_REGULAR),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),

        # Alternating row colors
        *[('BACKGROUND', (0, r), (-1, r), ING_GRAY)
          for r in range(2, len(table_data), 2)],

        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, ING_ORANGE),

        # Alignment
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def generate_report_pdf(result: dict, language: str = "EN") -> bytes:
    """
    Generate a complete ING-branded strategy report PDF.

    Args:
        result: The full swarm result dict containing:
            - strategy_report (markdown)
            - translated_report (markdown, optional — for TR)
            - financial_ratios
            - network_data
            - evaluation
            - agent_metrics
        language: "EN" for English report, "TR" for Turkish translated report

    Returns:
        bytes: The PDF file content
    """
    buffer = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=24 * mm, bottomMargin=16 * mm,
        leftMargin=15 * mm, rightMargin=15 * mm,
    )

    story = []

    # Adapt title based on language
    if language == "TR":
        page_title = "Finansal & Satış Analiz Raporu"
        disclaimer_text = (
            "Bu rapor Finansal &amp; Satış Analiz Platformu tarafından oluşturulmuştur. "
            "Veriler mizan ve işlem kayıtlarına dayanmaktadır. "
            "Bu belge gizlidir ve yalnızca dahili kullanım içindir."
        )
    else:
        page_title = "Financial & Sales Analysis Report"
        disclaimer_text = (
            "This report was generated by the Financial &amp; Sales Analysis Platform. "
            "Data is based on trial balance and transaction records. "
            "This document is confidential and intended for internal use only."
        )

    page_template = _INGPageTemplate(page_title)
    doc._ing_language = language  # Pass language to page template for footer

    # ====================================================================
    # PAGE 1: Cover / Executive Summary
    # ====================================================================
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(page_title.replace("&", "&amp;"), styles['INGTitle']))
    story.append(Paragraph(
        f"{result.get('company_name', 'Company')} — Tax ID: {result.get('tax_id', '1234567890')} — "
        f"{datetime.now().strftime('%d %B %Y')}",
        styles['INGSubtitle']
    ))
    story.append(HRFlowable(width="100%", thickness=2, color=ING_ORANGE,
                              spaceAfter=8, spaceBefore=2))

    # Quick stats summary
    ratios = result.get('financial_ratios', {})
    evaluation = result.get('evaluation', {})

    # Summary cards table
    gm = ratios.get('gross_margin', {}).get('value', '—')
    om = ratios.get('operating_margin', {}).get('value', '—')
    cr = ratios.get('current_ratio', {}).get('value', '—')
    qr = ratios.get('quick_ratio', {}).get('value', '—')
    dte = ratios.get('debt_to_equity', {}).get('value', '—')
    bdr = ratios.get('bank_debt_ratio', {}).get('value', '—')
    colp = ratios.get('collection_period', {}).get('value', '—')
    payp = ratios.get('payment_period', {}).get('value', '—')
    fer = ratios.get('financial_expense_ratio', {}).get('value', '—')
    pcr = ratios.get('pos_commission_ratio', {}).get('value', '—')

    if language == "TR":
        summary_data = [
            [Paragraph('<b>Gösterge</b>', styles['INGH3White']),
             Paragraph('<b>Değer</b>', styles['INGH3White']),
             Paragraph('<b>Değerlendirme</b>', styles['INGH3White'])],
            ['Brüt Kâr Marjı', f'{gm}%', 'Sağlıklı' if isinstance(gm, (int, float)) and gm > 25 else 'Düşük'],
            ['Faaliyet Kâr Marjı', f'{om}%', 'Sağlıklı' if isinstance(om, (int, float)) and om > 10 else 'Gözden Geçirilmeli'],
            ['Cari Oran', f'{cr}x', 'Güçlü' if isinstance(cr, (int, float)) and cr > 1.5 else 'Dar'],
            ['Asit Test Oranı', f'{qr}x', 'Optimum' if isinstance(qr, (int, float)) and qr > 1.0 else 'Düşük'],
            ['Borç/Özkaynak Oranı', f'{dte}x', 'Muhafazakâr' if isinstance(dte, (int, float)) and dte < 2 else 'Kaldıraçlı'],
            ['Banka Borç Oranı', f'{bdr}%', 'Düşük' if isinstance(bdr, (int, float)) and bdr < 20 else 'Yüksek'],
            ['Tahsilat Süresi', f'{colp} gün', 'Hızlı' if isinstance(colp, (int, float)) and colp < 60 else 'Yavaş'],
            ['Ödeme Süresi', f'{payp} gün', 'Uzun' if isinstance(payp, (int, float)) and payp > 60 else 'Kısa'],
            ['Finansman Gider Oranı', f'{fer}%', 'Yüksek' if isinstance(fer, (int, float)) and fer > 3 else 'Normal'],
            ['POS Komisyon Oranı', f'{pcr}%', 'Aşırı' if isinstance(pcr, (int, float)) and pcr > 1 else 'Normal'],
        ]
    else:
        summary_data = [
            [Paragraph('<b>Metric</b>', styles['INGH3White']),
             Paragraph('<b>Value</b>', styles['INGH3White']),
             Paragraph('<b>Assessment</b>', styles['INGH3White'])],
            ['Gross Margin', f'{gm}%', 'Healthy' if isinstance(gm, (int, float)) and gm > 25 else 'Low'],
            ['Operating Margin', f'{om}%', 'Healthy' if isinstance(om, (int, float)) and om > 10 else 'Review'],
            ['Current Ratio', f'{cr}x', 'Strong' if isinstance(cr, (int, float)) and cr > 1.5 else 'Tight'],
            ['Quick Ratio', f'{qr}x', 'Optimum' if isinstance(qr, (int, float)) and qr > 1.0 else 'Low'],
            ['Debt-to-Equity', f'{dte}x', 'Conservative' if isinstance(dte, (int, float)) and dte < 2 else 'Leveraged'],
            ['Bank Debt Ratio', f'{bdr}%', 'Low' if isinstance(bdr, (int, float)) and bdr < 20 else 'High'],
            ['Collection Period', f'{colp} days', 'Fast' if isinstance(colp, (int, float)) and colp < 60 else 'Slow'],
            ['Payment Period', f'{payp} days', 'Long' if isinstance(payp, (int, float)) and payp > 60 else 'Short'],
            ['Financial Expense', f'{fer}%', 'High' if isinstance(fer, (int, float)) and fer > 3 else 'Normal'],
            ['POS Commission', f'{pcr}%', 'Excessive' if isinstance(pcr, (int, float)) and pcr > 1 else 'Normal'],
        ]

    # Convert non-Paragraph cells to Paragraphs
    for ri in range(1, len(summary_data)):
        for ci in range(len(summary_data[ri])):
            if isinstance(summary_data[ri][ci], str):
                summary_data[ri][ci] = Paragraph(summary_data[ri][ci], styles['INGSmall'])

    summary_table = Table(summary_data, colWidths=[60 * mm, 40 * mm, 60 * mm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ING_NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), ING_WHITE),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
        ('TOPPADDING', (0, 0), (-1, 0), 6),
        ('BACKGROUND', (0, 1), (-1, -1), ING_WHITE),
        *[('BACKGROUND', (0, r), (-1, r), ING_GRAY) for r in range(2, len(summary_data), 2)],
        ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, ING_ORANGE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
    ]))

    story.append(Paragraph(
        "Yönetici Özeti" if language == "TR" else "Executive Summary",
        styles['INGH1']
    ))
    story.append(summary_table)
    story.append(Spacer(1, 4))

    # Network stats (compact on same page as summary)
    network = result.get('network_data', {})
    ns = network.get('stats', {})
    if ns:
        story.append(Paragraph(
            "Ticari Ağ Görünümü" if language == "TR" else "Commercial Network Overview",
            styles['INGH2']
        ))
        if language == "TR":
            net_data = [
                [Paragraph('<b>Alacaklar</b>', styles['INGH3White']),
                 Paragraph('<b>Borçlar</b>', styles['INGH3White']),
                 Paragraph('<b>Müşteriler</b>', styles['INGH3White']),
                 Paragraph('<b>Tedarikçiler</b>', styles['INGH3White']),
                 Paragraph('<b>Bankalar</b>', styles['INGH3White'])],
            ]
        else:
            net_data = [
                [Paragraph('<b>Receivables</b>', styles['INGH3White']),
                 Paragraph('<b>Payables</b>', styles['INGH3White']),
                 Paragraph('<b>Customers</b>', styles['INGH3White']),
                 Paragraph('<b>Suppliers</b>', styles['INGH3White']),
                 Paragraph('<b>Banks</b>', styles['INGH3White'])],
            ]
        net_data.append(
            [Paragraph(f"{ns.get('total_receivables', 0):,.0f} TL", styles['INGSmall']),
             Paragraph(f"{ns.get('total_payables', 0):,.0f} TL", styles['INGSmall']),
             Paragraph(str(ns.get('customer_count', 0)), styles['INGSmall']),
             Paragraph(str(ns.get('supplier_count', 0)), styles['INGSmall']),
             Paragraph(str(ns.get('bank_count', 0)), styles['INGSmall'])],
        )
        net_table = Table(net_data, colWidths=[36 * mm] * 5)
        net_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), ING_NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), ING_WHITE),
            ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
            ('LINEBELOW', (0, 0), (-1, 0), 1.5, ING_ORANGE),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(net_table)

    # ====================================================================
    # PAGE 2+: Strategy Report — each # section on its own page
    # ====================================================================
    story.append(PageBreak())

    # Select the right report based on language
    if language == "TR":
        strategy_report = result.get('translated_report', '') or result.get('strategy_report', '')
    else:
        strategy_report = result.get('strategy_report', '')

    if strategy_report:
        report_elements = _parse_markdown_basic(
            strategy_report, styles, page_break_on_h1=True
        )
        story.extend(report_elements)

    # ====================================================================
    # Disclaimer (at the bottom of the last page)
    # ====================================================================
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ING_NAVY,
                              spaceAfter=4, spaceBefore=4))
    story.append(Paragraph(disclaimer_text, styles['INGFooter']))

    # Build the PDF
    doc.build(story, onFirstPage=page_template, onLaterPages=page_template)

    pdf_bytes = buffer.getvalue()
    buffer.close()

    logger.info(f"[PDF] Generated {language} report: {len(pdf_bytes)} bytes (font: {FONT_REGULAR})")
    return pdf_bytes


def save_report_pdf(result: dict, output_dir: str, company_name: str, language: str = "EN") -> str:
    """
    Generate and save a PDF report to disk.

    Args:
        result: Pipeline result dict
        output_dir: Directory to write the PDF
        company_name: Company name for filename
        language: "EN" or "TR"

    Returns:
        str: Full path to the saved PDF file
    """
    pdf_bytes = generate_report_pdf(result, language=language)

    # Sanitize company name for filename
    safe_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
    filename = f"{safe_name}_Report_{language}.pdf"
    filepath = os.path.join(output_dir, filename)

    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(pdf_bytes)

    logger.info(f"[PDF] Saved: {filepath} ({len(pdf_bytes)} bytes)")
    return filepath

