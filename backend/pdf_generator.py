"""
PDF Report Generator — ING Global Brand Style
================================================
Generates a premium, branded PDF report from the swarm analysis results.
Uses ING brand colors and the official INGMe font family.
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
from reportlab.pdfbase.pdfmetrics import registerFontFamily
 
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
    global _ingme_available, FONT_REGULAR, FONT_BOLD
    regular_path = FONTS_DIR / "INGMeWeb-Regular.ttf"
    bold_path = FONTS_DIR / "INGMeWeb-Bold.ttf"
    def _is_valid_ttf(path: Path) -> bool:
        if not path.exists(): return False
        if path.stat().st_size < 5000: return False
        with open(path, 'rb') as f:
            header = f.read(4)
            return header in (b'\x00\x01\x00\x00', b'true', b'OTTO')
    if _is_valid_ttf(regular_path) and _is_valid_ttf(bold_path):
        try:
            pdfmetrics.registerFont(TTFont('INGMe', str(regular_path)))
            pdfmetrics.registerFont(TTFont('INGMe-Bold', str(bold_path)))
            # BOLD ve ITALIC tag'lerinin düzgün çalışması için font ailesini kaydediyoruz
            registerFontFamily('INGMe', normal='INGMe', bold='INGMe-Bold', italic='INGMe', boldItalic='INGMe-Bold')
            registerFontFamily('INGMe-Bold', normal='INGMe-Bold', bold='INGMe-Bold', italic='INGMe-Bold', boldItalic='INGMe-Bold')
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
            # HATA BURADAYDI: Eskiden regular_path bold olarak kaydediliyordu. 
            # Şimdi kalın font yoksa mecburen Helvetica-Bold'a fallback yapıyoruz.
            registerFontFamily('INGMe', normal='INGMe', bold='Helvetica-Bold', italic='INGMe', boldItalic='Helvetica-Bold')
            _ingme_available = True
            FONT_REGULAR = "INGMe"
            FONT_BOLD = "Helvetica-Bold"
            logger.info("[PDF] INGMe font loaded (regular only, using Helvetica-Bold for bold)")
            return
        except Exception as e:
            logger.warning(f"[PDF] INGMe font registration failed: {e}")
    _ingme_available = False
    FONT_REGULAR = "Helvetica"
    FONT_BOLD = "Helvetica-Bold"
    logger.info(f"[PDF] INGMe font not found at {FONTS_DIR}. Using Helvetica fallback.")
#def _register_ingme_fonts():
#    global _ingme_available, FONT_REGULAR, FONT_BOLD
#    regular_path = FONTS_DIR / "INGMeWeb-Regular.ttf"
#    bold_path = FONTS_DIR / "INGMeWeb-Bold.ttf"
# 
#    def _is_valid_ttf(path: Path) -> bool:
#        if not path.exists(): return False
#        if path.stat().st_size < 5000: return False
#        with open(path, 'rb') as f:
#            header = f.read(4)
#            return header in (b'\x00\x01\x00\x00', b'true', b'OTTO')
# 
#    if _is_valid_ttf(regular_path) and _is_valid_ttf(bold_path):
#        try:
#            pdfmetrics.registerFont(TTFont('INGMe', str(regular_path)))
#            pdfmetrics.registerFont(TTFont('INGMe-Bold', str(bold_path)))
#            _ingme_available = True
#            FONT_REGULAR = "INGMe"
#            FONT_BOLD = "INGMe-Bold"
#            logger.info("[PDF] INGMe font loaded successfully")
#            return
#        except Exception as e:
#            logger.warning(f"[PDF] INGMe font registration failed: {e}")
#    elif _is_valid_ttf(regular_path):
#        try:
#            pdfmetrics.registerFont(TTFont('INGMe', str(regular_path)))
#            pdfmetrics.registerFont(TTFont('INGMe-Bold', str(regular_path)))
#            _ingme_available = True
#            FONT_REGULAR = "INGMe"
#            FONT_BOLD = "INGMe-Bold"
#            logger.info("[PDF] INGMe font loaded (regular only, using for bold too)")
#            return
#        except Exception as e:
#            logger.warning(f"[PDF] INGMe font registration failed: {e}")
# 
#    _ingme_available = False
#    FONT_REGULAR = "Helvetica"
#    FONT_BOLD = "Helvetica-Bold"
#    logger.info(f"[PDF] INGMe font not found at {FONTS_DIR}. Using Helvetica fallback.")
_register_ingme_fonts()
def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='INGH3White', fontName=FONT_BOLD, fontSize=10, textColor=ING_WHITE, leading=13))
    styles.add(ParagraphStyle(name='INGTitle', fontName=FONT_BOLD, fontSize=20, textColor=ING_NAVY, spaceAfter=4, leading=24))
    styles.add(ParagraphStyle(name='INGSubtitle', fontName=FONT_REGULAR, fontSize=10, textColor=ING_ORANGE, spaceAfter=10, leading=13))
    styles.add(ParagraphStyle(name='INGH1', fontName=FONT_BOLD, fontSize=14, textColor=ING_NAVY, spaceBefore=6, spaceAfter=6, leading=17))
    styles.add(ParagraphStyle(name='INGH2', fontName=FONT_BOLD, fontSize=11, textColor=ING_ORANGE, spaceBefore=8, spaceAfter=4, leading=14))
    styles.add(ParagraphStyle(name='INGH3', fontName=FONT_BOLD, fontSize=9.5, textColor=ING_NAVY, spaceBefore=6, spaceAfter=3, leading=12))
    styles.add(ParagraphStyle(name='INGBody', fontName=FONT_REGULAR, fontSize=8.5, textColor=ING_TEXT, spaceAfter=4, leading=12))
    styles.add(ParagraphStyle(name='INGSmall', fontName=FONT_REGULAR, fontSize=7.5, textColor=ING_TEXT, spaceAfter=3, leading=10))
    styles.add(ParagraphStyle(name='INGFooter', fontName=FONT_REGULAR, fontSize=7, textColor=ING_TEXT_SECONDARY, alignment=TA_CENTER))
    return styles
class _INGPageTemplate:
    def __init__(self, title: str):
        self.title = title
    def __call__(self, canvas_obj, doc):
        canvas_obj.saveState()
        w, h = A4
        canvas_obj.setFillColor(ING_ORANGE)
        canvas_obj.rect(0, h - 18 * mm, w, 18 * mm, fill=True, stroke=False)
        logo_x = 12 * mm
        if ING_LOGO_PATH.exists():
            try:
                logo_h = 12 * mm
                logo_w = logo_h * 2.4
                canvas_obj.drawImage(str(ING_LOGO_PATH), logo_x, h - 16 * mm, width=logo_w, height=logo_h, preserveAspectRatio=True, mask='auto')
                text_x = logo_x + logo_w + 4 * mm
            except Exception:
                text_x = 15 * mm
        else:
            text_x = 15 * mm
        canvas_obj.setFillColor(ING_WHITE)
        canvas_obj.setFont(FONT_BOLD, 11)
        canvas_obj.drawString(text_x, h - 12 * mm, self.title)
        canvas_obj.setFont(FONT_REGULAR, 7)
        canvas_obj.drawRightString(w - 15 * mm, h - 12 * mm, 'CONFIDENTIAL')
        canvas_obj.setFillColor(ING_NAVY)
        canvas_obj.rect(0, 0, w, 10 * mm, fill=True, stroke=False)
        canvas_obj.setFillColor(ING_WHITE)
        canvas_obj.setFont(FONT_REGULAR, 7)
        footer_label = f'CAO GenAI Satış Analiz Platformu — {datetime.now().strftime("%d %B %Y")}' if hasattr(doc, '_ing_language') and doc._ing_language == 'TR' else f'CAO GenAI Sales Analysis Platform — {datetime.now().strftime("%d %B %Y")}'
        canvas_obj.drawString(15 * mm, 3.5 * mm, footer_label)
        canvas_obj.drawRightString(w - 15 * mm, 3.5 * mm, f'Page {doc.page}')
        canvas_obj.setStrokeColor(ING_ORANGE)
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(0, h - 18 * mm, w, h - 18 * mm)
        canvas_obj.restoreState()
def _parse_markdown_basic(text: str, styles, page_break_on_h1: bool = False) -> list:
    elements = []
    lines = text.strip().split('\n')
    i = 0
    h1_count = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            elements.append(Spacer(1, 3))
            i += 1
            continue
        if line.startswith('---') or line.startswith('***'):
            elements.append(HRFlowable(width="100%", thickness=0.5, color=ING_ORANGE, spaceAfter=4, spaceBefore=4))
            i += 1
            continue
        if line.startswith('### '):
            elements.append(Paragraph(_clean_md(line[4:]), styles['INGH3']))
            i += 1
            continue
        if line.startswith('## '):
            elements.append(Paragraph(_clean_md(line[3:]), styles['INGH2']))
            i += 1
            continue
        if line.startswith('# '):
            if page_break_on_h1 and h1_count > 0:
                elements.append(PageBreak())
            h1_count += 1
            elements.append(Paragraph(_clean_md(line[2:]), styles['INGH1']))
            elements.append(HRFlowable(width="100%", thickness=1, color=ING_ORANGE, spaceAfter=4, spaceBefore=2))
            i += 1
            continue
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
        if line.startswith('- ') or line.startswith('• '):
            elements.append(Paragraph(f'•  {_clean_md(line[2:])}', styles['INGBody']))
            i += 1
            continue
        if len(line) > 2 and line[0].isdigit() and line[1] in '.):':
            elements.append(Paragraph(_clean_md(line), styles['INGBody']))
            i += 1
            continue
        elements.append(Paragraph(_clean_md(line), styles['INGBody']))
        i += 1
    return elements
#def _clean_md(text: str) -> str:
#    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
#    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
#    text = re.sub(r'`(.+?)`', r'<font face="Courier" size="8" color="#000066">\1</font>', text)
#    return text
def _clean_md(text: str) -> str:
    # ReportLab Paragraph yapısında XML parsing hatası almamak için:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<font face="Courier" size="8" color="#000066">\1</font>', text)
    return text 
def _build_table(rows: list, styles) -> Table:
    if not rows: return Spacer(1, 1)
    table_data = []
    for ri, row in enumerate(rows):
        styled_row = []
        for cell in row:
            style = styles['INGH3White'] if ri == 0 else styles['INGSmall']
            styled_row.append(Paragraph(_clean_md(cell), style))
        table_data.append(styled_row)
    num_cols = max(len(r) for r in table_data) if table_data else 1
    available_width = A4[0] - 30 * mm
    col_width = available_width / num_cols
    t = Table(table_data, colWidths=[col_width] * num_cols)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ING_NAVY),
        ('TEXTCOLOR', (0, 0), (-1, 0), ING_WHITE),
        ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
        ('FONTSIZE', (0, 0), (-1, 0), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), ING_WHITE),
        ('TEXTCOLOR', (0, 1), (-1, -1), ING_TEXT),
        ('FONTNAME', (0, 1), (-1, -1), FONT_REGULAR),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        *[('BACKGROUND', (0, r), (-1, r), ING_GRAY) for r in range(2, len(table_data), 2)],
        ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, ING_ORANGE),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t
def generate_report_pdf(result: dict, language: str = "EN") -> bytes:
    buffer = io.BytesIO()
    styles = _build_styles()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=24 * mm, bottomMargin=16 * mm, leftMargin=15 * mm, rightMargin=15 * mm)
    story = []
    if language == "TR":
        page_title = "GenAI Satış Analiz Raporu"
        disclaimer_text = "Bu rapor CAO GenAI Satış Analiz Platformu tarafından oluşturulmuştur. Veriler mizan ve işlem kayıtlarına dayanmaktadır. Bu belge gizlidir ve yalnızca dahili kullanım içindir."
    else:
        page_title = "GenAI Sales Analysis Report"
        disclaimer_text = "This report was generated by the CAO GenAI Sales Analysis Platform. Data is based on trial balance and transaction records. This document is confidential and intended for internal use only."
    page_template = _INGPageTemplate(page_title)
    doc._ing_language = language
    # ====================================================================
    # PAGE 1: Cover / Executive Summary
    # ====================================================================
    # Güvenli dönem verisi çekme (Eğer donem_info bir dictionary değilse veya eksikse hata vermesini önler)
    donem_info = result.get('donem_info', {})
    donem_raw = donem_info.get('raw', '190012') if isinstance(donem_info, dict) else str(donem_info)
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph(page_title.replace("&", "&amp;"), styles['INGTitle']))
    if language == "TR":
        subtitle_text = f"{result.get('company_name', 'Firma')} — Vergi NO: {result.get('tax_id', '1234567890')} — {datetime.now().strftime('%d %B %Y')} — Dönem: {donem_raw}"
    else:
        subtitle_text = f"{result.get('company_name', 'Company')} — Tax ID: {result.get('tax_id', '1234567890')} — {datetime.now().strftime('%d %B %Y')} — Period: {donem_raw}"
    story.append(Paragraph(subtitle_text, styles['INGSubtitle']))
    story.append(HRFlowable(width="100%", thickness=2, color=ING_ORANGE, spaceAfter=8, spaceBefore=2))

    # ── 1. Financial Ratios Summary Table (COMMENTED OUT) ──
    # ratios = result.get('financial_ratios', {})
    # gm = ratios.get('gross_margin', {}).get('value', '—')
    # om = ratios.get('operating_margin', {}).get('value', '—')
    # cr = ratios.get('current_ratio', {}).get('value', '—')
    # qr = ratios.get('quick_ratio', {}).get('value', '—')
    # dte = ratios.get('debt_to_equity', {}).get('value', '—')
    # bdr = ratios.get('bank_debt_ratio', {}).get('value', '—')
    # colp = ratios.get('collection_period', {}).get('value', '—')
    # payp = ratios.get('payment_period', {}).get('value', '—')
    # fer = ratios.get('financial_expense_ratio', {}).get('value', '—')
    # pcr = ratios.get('pos_commission_ratio', {}).get('value', '—')
    #
    # if language == "TR":
    #     summary_data = [
    #         [Paragraph('<b>Gösterge</b>', styles['INGH3White']), Paragraph('<b>Değer</b>', styles['INGH3White']), Paragraph('<b>Değerlendirme</b>', styles['INGH3White'])],
    #         ['Brüt Kâr Marjı', f'{gm}%' if gm != '—' else '—', 'Sağlıklı' if isinstance(gm, (int, float)) and gm > 25 else 'Düşük'],
    #         ['Faaliyet Kâr Marjı', f'{om}%' if om != '—' else '—', 'Sağlıklı' if isinstance(om, (int, float)) and om > 10 else 'Gözden Geçirilmeli'],
    #         ['Cari Oran', f'{cr}x' if cr != '—' else '—', 'Güçlü' if isinstance(cr, (int, float)) and cr > 1.5 else 'Dar'],
    #         ['Asit Test Oranı', f'{qr}x' if qr != '—' else '—', 'Optimum' if isinstance(qr, (int, float)) and qr > 1.0 else 'Düşük'],
    #         ['Borç/Özkaynak Oranı', f'{dte}x' if dte != '—' else '—', 'Muhafazakâr' if isinstance(dte, (int, float)) and dte < 2 else 'Kaldıraçlı'],
    #         ['Banka Borç Oranı', f'{bdr}%' if bdr != '—' else '—', 'Düşük' if isinstance(bdr, (int, float)) and bdr < 20 else 'Yüksek'],
    #         ['Tahsilat Süresi', f'{colp} gün' if colp != '—' else '—', 'Hızlı' if isinstance(colp, (int, float)) and colp < 60 else 'Yavaş'],
    #         ['Ödeme Süresi', f'{payp} gün' if payp != '—' else '—', 'Uzun' if isinstance(payp, (int, float)) and payp > 60 else 'Kısa'],
    #         ['Finansman Gider Oranı', f'{fer}%' if fer != '—' else '—', 'Yüksek' if isinstance(fer, (int, float)) and fer > 3 else 'Normal'],
    #         ['POS Komisyon Oranı', f'{pcr}%' if pcr != '—' else '—', 'Aşırı' if isinstance(pcr, (int, float)) and pcr > 1 else 'Normal'],
    #     ]
    # else:
    #     summary_data = [
    #         [Paragraph('<b>Metric</b>', styles['INGH3White']), Paragraph('<b>Value</b>', styles['INGH3White']), Paragraph('<b>Assessment</b>', styles['INGH3White'])],
    #         ['Gross Margin', f'{gm}%' if gm != '—' else '—', 'Healthy' if isinstance(gm, (int, float)) and gm > 25 else 'Low'],
    #         ['Operating Margin', f'{om}%' if om != '—' else '—', 'Healthy' if isinstance(om, (int, float)) and om > 10 else 'Review'],
    #         ['Current Ratio', f'{cr}x' if cr != '—' else '—', 'Strong' if isinstance(cr, (int, float)) and cr > 1.5 else 'Tight'],
    #         ['Quick Ratio', f'{qr}x' if qr != '—' else '—', 'Optimum' if isinstance(qr, (int, float)) and qr > 1.0 else 'Low'],
    #         ['Debt-to-Equity', f'{dte}x' if dte != '—' else '—', 'Conservative' if isinstance(dte, (int, float)) and dte < 2 else 'Leveraged'],
    #         ['Bank Debt Ratio', f'{bdr}%' if bdr != '—' else '—', 'Low' if isinstance(bdr, (int, float)) and bdr < 20 else 'High'],
    #         ['Collection Period', f'{colp} days' if colp != '—' else '—', 'Fast' if isinstance(colp, (int, float)) and colp < 60 else 'Slow'],
    #         ['Payment Period', f'{payp} days' if payp != '—' else '—', 'Long' if isinstance(payp, (int, float)) and payp > 60 else 'Short'],
    #         ['Financial Expense', f'{fer}%' if fer != '—' else '—', 'High' if isinstance(fer, (int, float)) and fer > 3 else 'Normal'],
    #         ['POS Commission', f'{pcr}%' if pcr != '—' else '—', 'Excessive' if isinstance(pcr, (int, float)) and pcr > 1 else 'Normal'],
    #     ]
    #
    # for ri in range(1, len(summary_data)):
    #     for ci in range(len(summary_data[ri])):
    #         if isinstance(summary_data[ri][ci], str):
    #             summary_data[ri][ci] = Paragraph(summary_data[ri][ci], styles['INGSmall'])
    #
    # summary_table = Table(summary_data, colWidths=[60 * mm, 40 * mm, 60 * mm])
    # summary_table.setStyle(TableStyle([
    #     ('BACKGROUND', (0, 0), (-1, 0), ING_NAVY),
    #     ('TEXTCOLOR', (0, 0), (-1, 0), ING_WHITE),
    #     ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
    #     ('FONTSIZE', (0, 0), (-1, 0), 8),
    #     ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
    #     ('TOPPADDING', (0, 0), (-1, 0), 6),
    #     ('BACKGROUND', (0, 1), (-1, -1), ING_WHITE),
    #     *[('BACKGROUND', (0, r), (-1, r), ING_GRAY) for r in range(2, len(summary_data), 2)],
    #     ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
    #     ('LINEBELOW', (0, 0), (-1, 0), 1.5, ING_ORANGE),
    #     ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    #     ('LEFTPADDING', (0, 0), (-1, -1), 6),
    #     ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    #     ('TOPPADDING', (0, 1), (-1, -1), 5),
    #     ('BOTTOMPADDING', (0, 1), (-1, -1), 5),
    # ]))
    #
    # story.append(Paragraph("Yönetici Özeti" if language == "TR" else "Executive Summary", styles['INGH1']))
    # story.append(summary_table)
    # story.append(Spacer(1, 4))
    # ── 1. Competitor Bank Wallet Share Table ──
    wallet_data = result.get("wallet_dict", {})
    if wallet_data:
        if language == "TR":
            wallet_data_matrix = [
                [
                    Paragraph('<b>ING Payı/Diğer</b>', styles['INGH3White']), Paragraph('<b>Mevduat</b>', styles['INGH3White']),
                    Paragraph('<b>Top. Kredi</b>', styles['INGH3White']), Paragraph('<b>KV Kredi</b>', styles['INGH3White']),
                    Paragraph('<b>UV Kredi</b>', styles['INGH3White']), Paragraph('<b>Öd. Çeki</b>', styles['INGH3White']),
                    Paragraph('<b>Tah. Çeki</b>', styles['INGH3White']), Paragraph('<b>POS</b>', styles['INGH3White'])
                ],
                ['ING (Tutar)'], ['Diğer (Tutar)'], ['Diğer (Adet)'], ['ING Payı']
            ]
            section_title = "Rakip Banka Cüzdan Payı"
        else:
            wallet_data_matrix = [
                [
                    Paragraph('<b>ING Share/Other</b>', styles['INGH3White']), Paragraph('<b>Deposits</b>', styles['INGH3White']),
                    Paragraph('<b>Total Loans</b>', styles['INGH3White']), Paragraph('<b>ST Loans</b>', styles['INGH3White']),
                    Paragraph('<b>LT Loans</b>', styles['INGH3White']), Paragraph('<b>Iss. Checks</b>', styles['INGH3White']),
                    Paragraph('<b>Rec. Checks</b>', styles['INGH3White']), Paragraph('<b>POS</b>', styles['INGH3White'])
                ],
                ['ING (Amount)'], ['Other (Amount)'], ['Other (Count)'], ['ING Share']
            ]
            section_title = "Competitor Bank Wallet Share"
        keys_order = ["deposit", "total_loan", "st_loan", "lt_loan", "issued_check", "received_check", "pos"]
        for key in keys_order:
            ing_val = wallet_data.get(key, {}).get("ing", 0.0)
            other_val = wallet_data.get(key, {}).get("other", 0.0)
            count_val = wallet_data.get(key, {}).get("other_count", 0)
            total_val = ing_val + other_val
            wallet_data_matrix[1].append(f"₺{ing_val:,.0f}")
            wallet_data_matrix[2].append(f"₺{other_val:,.0f}")
            wallet_data_matrix[3].append(f"{count_val:,.0f}")
            if total_val > 0:
                share_pct = (ing_val / total_val) * 100
                wallet_data_matrix[4].append(f"<b>%{share_pct:.1f}</b>")
            else:
                wallet_data_matrix[4].append("<b>-</b>")
        for ri in range(1, len(wallet_data_matrix)):
            for ci in range(len(wallet_data_matrix[ri])):
                if isinstance(wallet_data_matrix[ri][ci], str):
                    wallet_data_matrix[ri][ci] = Paragraph(wallet_data_matrix[ri][ci], styles['INGSmall'])
        col_widths = [34 * mm] + [21 * mm] * 7
        wallet_table = Table(wallet_data_matrix, colWidths=col_widths)
        wallet_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), ING_NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), ING_WHITE),
            ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('BACKGROUND', (0, 1), (-1, 1), ING_WHITE),
            ('BACKGROUND', (0, 2), (-1, 2), ING_GRAY),
            ('BACKGROUND', (0, 3), (-1, 3), ING_WHITE),
            ('BACKGROUND', (0, 4), (-1, 4), ING_GRAY),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
            ('LINEBELOW', (0, 0), (-1, 0), 1.5, ING_ORANGE),
        ]))
        story.append(Paragraph(section_title, styles['INGH2']))
        story.append(wallet_table)
        story.append(Spacer(1, 4))
    # ── 2. Commercial Network Overview ──
    network = result.get('network_data', {})
    ns = network.get('stats', {})
    if ns:
        story.append(Paragraph("Özet" if language == "TR" else "Summary", styles['INGH2']))
        if language == "TR":
            net_data = [[Paragraph('<b>Alacaklar</b>', styles['INGH3White']), Paragraph('<b>Borçlar</b>', styles['INGH3White']), Paragraph('<b>Müşteriler</b>', styles['INGH3White']), Paragraph('<b>Tedarikçiler</b>', styles['INGH3White']), Paragraph('<b>Bankalar</b>', styles['INGH3White'])]]
        else:
            net_data = [[Paragraph('<b>Receivables</b>', styles['INGH3White']), Paragraph('<b>Payables</b>', styles['INGH3White']), Paragraph('<b>Customers</b>', styles['INGH3White']), Paragraph('<b>Suppliers</b>', styles['INGH3White']), Paragraph('<b>Banks</b>', styles['INGH3White'])]]
        net_data.append([
            Paragraph(f"{ns.get('total_receivables', 0):,.0f} TL", styles['INGSmall']),
            Paragraph(f"{ns.get('total_payables', 0):,.0f} TL", styles['INGSmall']),
            Paragraph(str(ns.get('customer_count', 0)), styles['INGSmall']),
            Paragraph(str(ns.get('supplier_count', 0)), styles['INGSmall']),
            Paragraph(str(ns.get('bank_count', 0)), styles['INGSmall'])
        ])
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
        story.append(Spacer(1, 4))
    # ── 3. Account Balances Table (600, 601, 602) ──
    balances_data = result.get('account_balances', {})
    if balances_data:
        balances_title = "Gelir Hesapları Bakiye Dağılımı" if language == "TR" else "Revenue Account Balance Distribution"
        accounts_tr = {
            "600": "600 - Yurtiçi Satışlar",
            "601": "601 - Yurtdışı Satışlar",
            "602": "602 - Diğer Gelirler"
        }
        accounts_en = {
            "600": "600 - Domestic Sales",
            "601": "601 - Foreign Sales",
            "602": "602 - Other Revenues"
        }
        if language == "TR":
            balances_matrix = [
                [
                    Paragraph('<b>Hesap Adı</b>', styles['INGH3White']),
                    Paragraph('<b>Borç (Debit)</b>', styles['INGH3White']),
                    Paragraph('<b>Alacak (Credit)</b>', styles['INGH3White']),
                    Paragraph('<b>Net Bakiye</b>', styles['INGH3White'])
                ]
            ]
            account_names = accounts_tr
        else:
            balances_matrix = [
                [
                    Paragraph('<b>Account Name</b>', styles['INGH3White']),
                    Paragraph('<b>Debit</b>', styles['INGH3White']),
                    Paragraph('<b>Credit</b>', styles['INGH3White']),
                    Paragraph('<b>Net Balance</b>', styles['INGH3White'])
                ]
            ]
            account_names = accounts_en
        for code in ["600", "601", "602"]:
            c_data = balances_data.get(code, {"debit": 0.0, "credit": 0.0, "balance": 0.0})
            deb = c_data.get("debit", 0.0)
            cre = c_data.get("credit", 0.0)
            bal = c_data.get("balance", 0.0)
            acc_name = account_names.get(code, code)
            balances_matrix.append([
                f"<b>{acc_name}</b>",
                f"₺{deb:,.2f}",
                f"₺{cre:,.2f}",
                f"<b>₺{bal:,.2f}</b>"
            ])
        for ri in range(1, len(balances_matrix)):
            for ci in range(len(balances_matrix[ri])):
                if isinstance(balances_matrix[ri][ci], str):
                    balances_matrix[ri][ci] = Paragraph(balances_matrix[ri][ci], styles['INGSmall'])
        # Tablo genişliklerini de daha iyi bir görünüm için ayarlayabilirsiniz
        # İlk sütun (Hesap Adı) daha geniş, diğerleri eşit olabilir.
        balances_table = Table(balances_matrix, colWidths=[60 * mm, 30 * mm, 30 * mm, 30 * mm])
        #balances_table = Table(balances_matrix, colWidths=[40 * mm] * 4) -old
        balances_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), ING_NAVY),
            ('TEXTCOLOR', (0, 0), (-1, 0), ING_WHITE),
            ('FONTNAME', (0, 0), (-1, 0), FONT_BOLD),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('BACKGROUND', (0, 1), (-1, 1), ING_WHITE),
            ('BACKGROUND', (0, 2), (-1, 2), ING_GRAY),
            ('BACKGROUND', (0, 3), (-1, 3), ING_WHITE),
            ('ALIGN', (0, 1), (0, -1), 'CENTER'),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.Color(0.85, 0.85, 0.85)),
            ('LINEBELOW', (0, 0), (-1, 0), 1.5, ING_ORANGE),
        ]))
        story.append(Paragraph(balances_title, styles['INGH2']))
        story.append(balances_table)
        story.append(Spacer(1, 4))
    # ====================================================================
    # PAGE 2+: Strategy Report — each # section on its own page
    # ====================================================================
    #story.append(PageBreak())
    strategy_report = result.get('translated_report', '') or result.get('strategy_report', '') if language == "TR" else result.get('strategy_report', '')
    if strategy_report:
        report_elements = _parse_markdown_basic(strategy_report, styles, page_break_on_h1=True)
        story.extend(report_elements)
    # ====================================================================
    # Disclaimer (at the bottom of the last page)
    # ====================================================================
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=ING_NAVY, spaceAfter=4, spaceBefore=4))
    story.append(Paragraph(disclaimer_text, styles['INGFooter']))
    doc.build(story, onFirstPage=page_template, onLaterPages=page_template)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    logger.info(f"[PDF] Generated {language} report: {len(pdf_bytes)} bytes (font: {FONT_REGULAR})")
    return pdf_bytes
def save_report_pdf(result: dict, output_dir: str, company_name: str, language: str = "EN") -> str:
    pdf_bytes = generate_report_pdf(result, language=language)
    safe_name = re.sub(r'[^\w\s-]', '', company_name).strip().replace(' ', '_')
    filename = f"{safe_name}_Report_{language}.pdf"
    filepath = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(filepath, 'wb') as f:
        f.write(pdf_bytes)
    logger.info(f"[PDF] Saved: {filepath} ({len(pdf_bytes)} bytes)")
    return filepath
 
