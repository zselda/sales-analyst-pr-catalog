"""
HTML Report Generator — ING Global Brand Style
================================================
Converts the strategy report (Markdown) to a self-contained, ING-branded HTML file.

Brand identity:
  - ING Orange: #FF6200
  - ING Navy:   #000066
  - Font: INGMe (with system font fallback)

Output is a single standalone HTML file with inline CSS — no external dependencies.
"""

import re
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("swarm.html")


def _md_to_html_body(md_text: str) -> str:
    """Convert basic Markdown to HTML body content."""
    lines = md_text.strip().split('\n')
    html_parts = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Empty line
        if not line:
            i += 1
            continue

        # Horizontal rule
        if line.startswith('---') or line.startswith('***'):
            html_parts.append('<hr class="ing-hr">')
            i += 1
            continue

        # Headings
        if line.startswith('### '):
            text = _inline_md(line[4:])
            html_parts.append(f'<h3>{text}</h3>')
            i += 1
            continue
        if line.startswith('## '):
            text = _inline_md(line[3:])
            html_parts.append(f'<h2>{text}</h2>')
            i += 1
            continue
        if line.startswith('# '):
            text = _inline_md(line[2:])
            html_parts.append(f'<h1>{text}</h1>')
            i += 1
            continue

        # Table detection
        if '|' in line and i + 1 < len(lines) and '---' in lines[i + 1]:
            table_html = ['<div class="table-wrapper"><table>']
            header_cells = [c.strip() for c in line.split('|') if c.strip()]
            table_html.append('<thead><tr>')
            for cell in header_cells:
                table_html.append(f'<th>{_inline_md(cell)}</th>')
            table_html.append('</tr></thead>')
            i += 2  # Skip header and separator line

            table_html.append('<tbody>')
            while i < len(lines) and '|' in lines[i]:
                row_line = lines[i].strip()
                if '---' in row_line:
                    i += 1
                    continue
                cells = [c.strip() for c in row_line.split('|') if c.strip()]
                table_html.append('<tr>')
                for cell in cells:
                    table_html.append(f'<td>{_inline_md(cell)}</td>')
                table_html.append('</tr>')
                i += 1
            table_html.append('</tbody></table></div>')
            html_parts.append('\n'.join(table_html))
            continue

        # Bullet list
        if line.startswith('- ') or line.startswith('• '):
            items = []
            while i < len(lines) and (lines[i].strip().startswith('- ') or lines[i].strip().startswith('• ')):
                item_text = lines[i].strip()[2:]
                items.append(f'<li>{_inline_md(item_text)}</li>')
                i += 1
            html_parts.append('<ul>' + '\n'.join(items) + '</ul>')
            continue

        # Numbered list
        if len(line) > 2 and line[0].isdigit() and line[1] in '.):':
            items = []
            while i < len(lines):
                l = lines[i].strip()
                if len(l) > 2 and l[0].isdigit() and l[1] in '.):':
                    item_text = re.sub(r'^\d+[.):\s]+', '', l)
                    items.append(f'<li>{_inline_md(item_text)}</li>')
                    i += 1
                else:
                    break
            html_parts.append('<ol>' + '\n'.join(items) + '</ol>')
            continue

        # Regular paragraph
        html_parts.append(f'<p>{_inline_md(line)}</p>')
        i += 1

    return '\n'.join(html_parts)


def _inline_md(text: str) -> str:
    """Convert inline Markdown formatting to HTML."""
    # Bold **text**
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic *text*
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    # Code `text`
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    return text


def generate_report_html(result: dict, language: str = "EN") -> str:
    """
    Generate a self-contained ING-branded HTML report.

    Args:
        result: The full pipeline result dict containing:
            - strategy_report (markdown)
            - translated_report (markdown)
            - financial_ratios
            - network_data
            - company_name
            - tax_id
        language: "EN" or "TR"

    Returns:
        str: Complete HTML document as string
    """
    company_name = result.get("company_name", "Company")
    tax_id = result.get("tax_id", "1234567890")
    report_date = datetime.now().strftime("%d %B %Y")
    
    # Select the report content based on requested language
    if language == "TR":
        report_content = result.get("translated_report") or result.get("strategy_report", "")
    else:
        report_content = result.get("strategy_report", "")

    # Convert report markdown to HTML body
    report_body = _md_to_html_body(report_content) if report_content else "<p>No report available.</p>"

    # Build KPI summary from ratios
    ratios = result.get("financial_ratios", {})
    kpi_rows = ""
    kpi_items_en = [
        ("Gross Margin", "gross_margin", "%"),
        ("Operating Margin", "operating_margin", "%"),
        ("Current Ratio", "current_ratio", "x"),
        ("Quick Ratio", "quick_ratio", "x"),
        ("Debt-to-Equity", "debt_to_equity", "x"),
        ("Bank Debt Ratio", "bank_debt_ratio", "%"),
        ("Collection Period", "collection_period", " days"),
        ("Payment Period", "payment_period", " days"),
        ("Fin. Expense Ratio", "financial_expense_ratio", "%"),
        ("POS Commission", "pos_commission_ratio", "%"),
    ]
    kpi_items_tr = [
        ("Brüt Kâr Marjı", "gross_margin", "%"),
        ("Faaliyet Kâr Marjı", "operating_margin", "%"),
        ("Cari Oran", "current_ratio", "x"),
        ("Asit Test Oranı", "quick_ratio", "x"),
        ("Borç/Özkaynak Oranı", "debt_to_equity", "x"),
        ("Banka Borç Oranı", "bank_debt_ratio", "%"),
        ("Tahsilat Süresi", "collection_period", " gün"),
        ("Ödeme Süresi", "payment_period", " gün"),
        ("Finansman Gider Oranı", "financial_expense_ratio", "%"),
        ("POS Komisyon Oranı", "pos_commission_ratio", "%"),
    ]
    kpi_items = kpi_items_tr if language == "TR" else kpi_items_en
    for label, key, unit in kpi_items:
        val = ratios.get(key, {}).get("value", "—")
        kpi_rows += f"<tr><td>{label}</td><td>{val}{unit}</td></tr>\n"

    # Network summary
    network = result.get("network_data", {})
    ns = network.get("stats", {})
    net_recv = f"₺{ns.get('total_receivables', 0):,.0f}" if ns else "—"
    net_pay = f"₺{ns.get('total_payables', 0):,.0f}" if ns else "—"
    net_cust = ns.get("customer_count", 0) if ns else "—"
    net_supp = ns.get("supplier_count", 0) if ns else "—"

    # Localized UI labels
    if language == "TR":
        html_lang = "tr"
        header_title = "Finansal &amp; Satış Analiz Raporu"
        badge_text = "GİZLİ"
        meta_company = "Firma"
        meta_taxid = "Vergi No"
        meta_date = "Tarih"
        meta_platform = "Platform"
        kpi_title = "📊 Temel Performans Göstergeleri"
        net_recv_label = "Toplam Alacaklar"
        net_pay_label = "Toplam Borçlar"
        net_cust_label = "Müşteriler"
        net_supp_label = "Tedarikçiler"
        footer_text = f"Finansal &amp; Satış Analiz Platformu — {report_date} — Bu belge gizlidir ve yalnızca kurum içi kullanım amaçlıdır."
    else:
        html_lang = "en"
        header_title = "Financial &amp; Sales Analysis Report"
        badge_text = "CONFIDENTIAL"
        meta_company = "Company"
        meta_taxid = "Tax ID"
        meta_date = "Date"
        meta_platform = "Platform"
        kpi_title = "📊 Key Performance Indicators"
        net_recv_label = "Total Receivables"
        net_pay_label = "Total Payables"
        net_cust_label = "Customers"
        net_supp_label = "Suppliers"
        footer_text = f"Financial &amp; Sales Analysis Platform — {report_date} — This document is confidential and intended for internal use only."

    html = f"""<!DOCTYPE html>
<html lang="{html_lang}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Financial Report — {company_name}</title>
    <style>
        :root {{
            --ing-orange: #FF6200;
            --ing-navy: #000066;
            --ing-gray: #f5f5f5;
            --ing-text: #262633;
            --ing-text-secondary: #666673;
            --ing-border: #e0e0e0;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'INGMe', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--ing-text);
            background: #fff;
            line-height: 1.6;
            font-size: 14px;
        }}

        .header {{
            background: var(--ing-orange);
            color: white;
            padding: 20px 40px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .header h1 {{
            font-size: 20px;
            font-weight: 700;
            color: white;
        }}

        .header .badge {{
            font-size: 11px;
            opacity: 0.9;
            letter-spacing: 1px;
        }}

        .meta-bar {{
            background: var(--ing-navy);
            color: white;
            padding: 10px 40px;
            font-size: 12px;
            display: flex;
            gap: 30px;
        }}

        .meta-bar span {{ opacity: 0.9; }}

        .container {{
            max-width: 900px;
            margin: 0 auto;
            padding: 30px 40px;
        }}

        /* KPI Summary */
        .kpi-section {{
            margin-bottom: 30px;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0;
        }}

        .kpi-grid table {{
            width: 100%;
            border-collapse: collapse;
        }}

        .kpi-grid td {{
            padding: 8px 12px;
            border-bottom: 1px solid var(--ing-border);
            font-size: 13px;
        }}

        .kpi-grid td:first-child {{
            font-weight: 600;
            color: var(--ing-navy);
        }}

        .kpi-grid td:last-child {{
            text-align: right;
            color: var(--ing-text);
        }}

        .network-bar {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 30px;
        }}

        .net-card {{
            background: var(--ing-gray);
            padding: 12px 16px;
            border-radius: 4px;
            border-left: 3px solid var(--ing-orange);
        }}

        .net-card .label {{ font-size: 11px; color: var(--ing-text-secondary); }}
        .net-card .value {{ font-size: 16px; font-weight: 700; color: var(--ing-navy); }}

        /* Report Body */
        .report-body h1 {{
            color: var(--ing-navy);
            font-size: 20px;
            margin: 30px 0 10px;
            padding-bottom: 6px;
            border-bottom: 2px solid var(--ing-orange);
        }}

        .report-body h2 {{
            color: var(--ing-orange);
            font-size: 16px;
            margin: 24px 0 8px;
        }}

        .report-body h3 {{
            color: var(--ing-navy);
            font-size: 14px;
            margin: 16px 0 6px;
        }}

        .report-body p {{
            margin: 8px 0;
            color: var(--ing-text);
        }}

        .report-body ul, .report-body ol {{
            margin: 8px 0 8px 24px;
        }}

        .report-body li {{
            margin: 4px 0;
        }}

        .report-body strong {{
            color: var(--ing-navy);
        }}

        .report-body code {{
            background: var(--ing-gray);
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 12px;
            color: var(--ing-navy);
        }}

        .table-wrapper {{
            overflow-x: auto;
            margin: 12px 0;
        }}

        .report-body table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}

        .report-body th {{
            background: var(--ing-navy);
            color: white;
            padding: 8px 10px;
            text-align: left;
            font-weight: 600;
        }}

        .report-body td {{
            padding: 7px 10px;
            border-bottom: 1px solid var(--ing-border);
        }}

        .report-body tr:nth-child(even) td {{
            background: var(--ing-gray);
        }}

        .ing-hr {{
            border: none;
            height: 1px;
            background: var(--ing-orange);
            margin: 20px 0;
        }}

        /* Footer */
        .footer {{
            background: var(--ing-navy);
            color: white;
            padding: 15px 40px;
            text-align: center;
            font-size: 11px;
            opacity: 0.9;
            margin-top: 40px;
        }}

        @media print {{
            .header {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
            .meta-bar {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
            .footer {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
            .report-body h1 {{ page-break-before: always; }}
            .report-body h1:first-child {{ page-break-before: avoid; }}
        }}
    </style>
</head>
<body>

<div class="header">
    <h1>{header_title}</h1>
    <span class="badge">{badge_text}</span>
</div>

<div class="meta-bar">
    <span><strong>{meta_company}:</strong> {company_name}</span>
    <span><strong>{meta_taxid}:</strong> {tax_id}</span>
    <span><strong>{meta_date}:</strong> {report_date}</span>
    <span><strong>{meta_platform}:</strong> Financial Intelligence Platform</span>
</div>

<div class="container">

    <div class="kpi-section">
        <h2 style="color: var(--ing-navy); margin-bottom: 10px;">{kpi_title}</h2>
        <div class="kpi-grid">
            <table>
                {kpi_rows}
            </table>
        </div>
    </div>

    <div class="network-bar">
        <div class="net-card">
            <div class="label">{net_recv_label}</div>
            <div class="value">{net_recv}</div>
        </div>
        <div class="net-card">
            <div class="label">{net_pay_label}</div>
            <div class="value">{net_pay}</div>
        </div>
        <div class="net-card">
            <div class="label">{net_cust_label}</div>
            <div class="value">{net_cust}</div>
        </div>
        <div class="net-card">
            <div class="label">{net_supp_label}</div>
            <div class="value">{net_supp}</div>
        </div>
    </div>

    <div class="report-body">
        {report_body}
    </div>

</div>

<div class="footer">
    {footer_text}
</div>

</body>
</html>"""

    logger.info(f"[HTML] Generated report: {len(html)} chars")
    return html
