"""
HTML Report Generator — ING Global Brand Style
================================================
Converts the strategy report (Markdown) to a self-contained, ING-branded HTML file.
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
def get_wallet_share_rows(wallet_data: dict, language: str = "TR") -> str:
    # Dil Sözlükleri
    lang_dict_tr = {
        "headers": ["Mevduat", "Kredi - Toplam", "Kredi - Kısa", "Kredi - Uzun", "Ödeme Çeki", "Tahsil Çeki", "POS"],
        "first_col": "Mizandan",
        "ing_label": "ING (Tutar)",
        "other_label": "Diğer (Tutar)",
        "count_label": "Diğer (Adet)",
        "share_label": "ING Payı"
    }
    lang_dict_en = {
        "headers": ["Deposits", "Total Loans", "ST Loans", "LT Loans", "Issued Checks", "Received Checks", "POS"],
        "first_col": "Trial Balance",
        "ing_label": "ING (Amount)",
        "other_label": "Other (Amount)",
        "count_label": "Other (Count)",
        "share_label": "ING Share"
    }
    texts = lang_dict_tr if language.upper() == "TR" else lang_dict_en
    keys_order = ["deposit", "total_loan", "st_loan", "lt_loan", "issued_check", "received_check", "pos"]
    rows_html = "<thead>\n  <tr>\n"
    rows_html += f"    <th style='text-align: left;'>{texts['first_col']}</th>\n"
    for header in texts["headers"]:
        rows_html += f"    <th style='text-align: center;'>{header}</th>\n"
    rows_html += "  </tr>\n</thead>\n<tbody>\n"
    ing_cells = ""
    diger_cells = ""
    count_cells = ""
    share_cells = ""
    for key in keys_order:
        ing_val = wallet_data.get(key, {}).get("ing", 0)
        diger_val = wallet_data.get(key, {}).get("other", 0)
        count_val = wallet_data.get(key, {}).get("other_count", 0)
        total_val = ing_val + diger_val
        ing_cells += f"    <td style='text-align: right;'>₺{ing_val:,.0f}</td>\n"
        diger_cells += f"    <td style='text-align: right;'>₺{diger_val:,.0f}</td>\n"
        count_cells += f"    <td style='text-align: right;'>{count_val:,.0f}</td>\n"
        if total_val > 0:
            share_pct = (ing_val / total_val) * 100
            share_cells += f"    <td style='text-align: right;'>%{share_pct:.1f}</td>\n"
        else:
            share_cells += f"    <td style='text-align: right;'>-</td>\n"
    rows_html += f"  <tr>\n    <td style='text-align: left;'><strong>{texts['ing_label']}</strong></td>\n{ing_cells}  </tr>\n"
    rows_html += f"  <tr>\n    <td style='text-align: left;'><strong>{texts['other_label']}</strong></td>\n{diger_cells}  </tr>\n"
    rows_html += f"  <tr>\n    <td style='text-align: left;'><strong>{texts['count_label']}</strong></td>\n{count_cells}  </tr>\n"
    rows_html += f"  <tr>\n    <td style='text-align: left;'><strong>{texts['share_label']}</strong></td>\n{share_cells}  </tr>\n"
    rows_html += "</tbody>\n"
    return rows_html
def get_sales_accounts_rows(account_data: dict, language: str = "TR") -> str:
    """Gelir hesapları (600, 601, 602) tablosu — PDF raporundaki
    'Revenue Account Balance Distribution' tablosuyla aynı format."""
    lang_dict_tr = {
        "headers": ["Hesap Adı", "Borç (Debit)", "Alacak (Credit)", "Net Bakiye"],
        "accounts": {
            "600": "600 - Yurtiçi Satışlar",
            "601": "601 - Yurtdışı Satışlar",
            "602": "602 - Diğer Gelirler"
        }
    }
    lang_dict_en = {
        "headers": ["Account Name", "Debit", "Credit", "Net Balance"],
        "accounts": {
            "600": "600 - Domestic Sales",
            "601": "601 - Foreign Sales",
            "602": "602 - Other Revenues"
        }
    }
    texts = lang_dict_tr if language.upper() == "TR" else lang_dict_en
    rows_html = "<thead>\n  <tr>\n"
    for header in texts["headers"]:
        rows_html += f"    <th style='text-align: center;'>{header}</th>\n"
    rows_html += "  </tr>\n</thead>\n<tbody>\n"
    for code in ["600", "601", "602"]:
        acc_name = texts["accounts"][code]
        data = account_data.get(code, {"debit": 0, "credit": 0, "balance": 0})
        rows_html += "  <tr>\n"
        rows_html += f"    <td style='text-align: left;'><strong>{acc_name}</strong></td>\n"
        rows_html += f"    <td style='text-align: right;'>₺{data.get('debit', 0):,.2f}</td>\n"
        rows_html += f"    <td style='text-align: right;'>₺{data.get('credit', 0):,.2f}</td>\n"
        rows_html += f"    <td style='text-align: right; font-weight: bold; color: var(--ing-navy);'>₺{data.get('balance', 0):,.2f}</td>\n"
        rows_html += "  </tr>\n"
    rows_html += "</tbody>\n"
    return rows_html
def get_network_summary_rows(ns: dict, language: str = "TR") -> str:
    """Ticari ağ özeti tablosu — PDF raporundaki 'Summary/Özet' tablosuyla
    aynı format (Alacaklar / Borçlar / Müşteriler / Tedarikçiler / Bankalar)."""
    if language.upper() == "TR":
        headers = ["Alacaklar", "Borçlar", "Müşteriler", "Tedarikçiler", "Bankalar"]
    else:
        headers = ["Receivables", "Payables", "Customers", "Suppliers", "Banks"]
    rows_html = "<thead>\n  <tr>\n"
    for header in headers:
        rows_html += f"    <th style='text-align: center;'>{header}</th>\n"
    rows_html += "  </tr>\n</thead>\n<tbody>\n  <tr>\n"
    rows_html += f"    <td style='text-align: center;'>{ns.get('total_receivables', 0):,.0f} TL</td>\n"
    rows_html += f"    <td style='text-align: center;'>{ns.get('total_payables', 0):,.0f} TL</td>\n"
    rows_html += f"    <td style='text-align: center;'>{ns.get('customer_count', 0)}</td>\n"
    rows_html += f"    <td style='text-align: center;'>{ns.get('supplier_count', 0)}</td>\n"
    rows_html += f"    <td style='text-align: center;'>{ns.get('bank_count', 0)}</td>\n"
    rows_html += "  </tr>\n</tbody>\n"
    return rows_html
def _inline_md(text: str) -> str:
    """Convert inline Markdown formatting to HTML."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    return text
def generate_report_html(result: dict, language: str = "EN") -> str:
    company_name = result.get("company_name", "Company")
    tax_id = result.get("tax_id", "1234567890")
    report_date = datetime.now().strftime("%d %B %Y")
    # Güvenli dönem verisi çekme — pdf_generator ile aynı yaklaşım
    donem_info = result.get("donem_info", {})
    mizan_donem = donem_info.get("raw", "—") if isinstance(donem_info, dict) else str(donem_info or "—")
    if language == "TR":
        report_content = result.get("translated_report") or result.get("strategy_report", "")
    else:
        report_content = result.get("strategy_report", "")
    report_body = _md_to_html_body(report_content) if report_content else "<p>No report available.</p>"
    # KPI ALANLARI VE HESAPLAMALARI YORUM SATIRINA ALINDI
    # ratios = result.get("financial_ratios", {})
    # kpi_rows = ""
    # kpi_items_en = [
    #     ("Gross Margin", "gross_margin", "%"), ("Operating Margin", "operating_margin", "%"),
    #     ("Current Ratio", "current_ratio", "x"), ("Quick Ratio", "quick_ratio", "x"),
    #     ("Debt-to-Equity", "debt_to_equity", "x"), ("Bank Debt Ratio", "bank_debt_ratio", "%"),
    #     ("Collection Period", "collection_period", " days"), ("Payment Period", "payment_period", " days"),
    #     ("Fin. Expense Ratio", "financial_expense_ratio", "%")
    # ]
    # kpi_items_tr = [
    #     ("Brüt Kâr Marjı", "gross_margin", "%"), ("Faaliyet Kâr Marjı", "operating_margin", "%"),
    #     ("Cari Oran", "current_ratio", "x"), ("Asit Test Oranı", "quick_ratio", "x"),
    #     ("Borç/Özkaynak Oranı", "debt_to_equity", "x"), ("Banka Borç Oranı", "bank_debt_ratio", "%"),
    #     ("Tahsilat Süresi", "collection_period", " gün"), ("Ödeme Süresi", "payment_period", " gün"),
    #     ("Finansman Gider Oranı", "financial_expense_ratio", "%")
    # ]
    # kpi_items = kpi_items_tr if language == "TR" else kpi_items_en
    # for label, key, unit in kpi_items:
    #     val = ratios.get(key, {}).get("value", "—")
    #     kpi_rows += f"<tr><td>{label}</td><td>{val}{unit}</td></tr>\n"
    wallet_dict = result.get("wallet_dict", {})
    account_balances = result.get("account_balances", {})
    network = result.get("network_data", {})
    ns = network.get("stats", {}) if isinstance(network, dict) else {}
    if language == "TR":
        html_lang = "tr"
        # PDF raporundaki page_title ile aynı
        header_title = "GenAI Satış Analiz Raporu"
        badge_text = "GİZLİ"
        meta_company = "Firma"
        meta_taxid = "Vergi No"
        meta_date = "Tarih"
        meta_mizan_donem = 'Dönem'
        meta_platform = "Platform"
        # PDF rapor bölüm başlıklarıyla birebir aynı
        wallet_share_title = "Rakip Banka Cüzdan Payı"
        network_summary_title = "Özet"
        sales_accounts_title = "Gelir Hesapları Bakiye Dağılımı"
        footer_text = f"CAO GenAI Satış Analiz Platformu — {report_date}"
        disclaimer_text = ("Bu rapor CAO GenAI Satış Analiz Platformu tarafından oluşturulmuştur. "
                           "Veriler mizan ve işlem kayıtlarına dayanmaktadır. "
                           "Bu belge gizlidir ve yalnızca dahili kullanım içindir.")
    else:
        html_lang = "en"
        # Same page_title as the PDF report
        header_title = "GenAI Sales Analysis Report"
        badge_text = "CONFIDENTIAL"
        meta_company = "Company"
        meta_taxid = "Tax ID"
        meta_date = "Date"
        meta_mizan_donem = 'Period'
        meta_platform = "Platform"
        # Section titles identical to the PDF report
        wallet_share_title = "Competitor Bank Wallet Share"
        network_summary_title = "Summary"
        sales_accounts_title = "Revenue Account Balance Distribution"
        footer_text = f"CAO GenAI Sales Analysis Platform — {report_date}"
        disclaimer_text = ("This report was generated by the CAO GenAI Sales Analysis Platform. "
                           "Data is based on trial balance and transaction records. "
                           "This document is confidential and intended for internal use only.")

    # ── Data sections (rendered only when data exists — same as the PDF) ──
    data_sections = []
    if wallet_dict:
        data_sections.append(
            f'<div class="data-section-container">\n'
            f'  <h2 style="color: var(--ing-navy); margin-bottom: 10px;">{wallet_share_title}</h2>\n'
            f'  <table class="ing-data-table wallet-table">\n'
            f'{get_wallet_share_rows(wallet_dict, language=language)}'
            f'  </table>\n</div>'
        )
    if ns:
        data_sections.append(
            f'<div class="data-section-container">\n'
            f'  <h2 style="color: var(--ing-navy); margin-bottom: 10px;">{network_summary_title}</h2>\n'
            f'  <table class="ing-data-table sales-table">\n'
            f'{get_network_summary_rows(ns, language=language)}'
            f'  </table>\n</div>'
        )
    if account_balances:
        data_sections.append(
            f'<div class="data-section-container">\n'
            f'  <h2 style="color: var(--ing-navy); margin-bottom: 10px;">{sales_accounts_title}</h2>\n'
            f'  <table class="ing-data-table sales-table">\n'
            f'{get_sales_accounts_rows(account_balances, language=language)}'
            f'  </table>\n</div>'
        )
    data_sections_html = "\n".join(data_sections)
    html = f"""<!DOCTYPE html>
<html lang="{html_lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sales Report — {company_name}</title>
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
            font-family: 'INGMeWeb-Regular.ttf', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
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
        .header h1 {{ font-size: 20px; font-weight: 700; color: white; }}
        .header .badge {{ font-size: 11px; opacity: 0.9; letter-spacing: 1px; }}
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
        /* KPI Summary CSS */
        .kpi-section {{ margin-bottom: 30px; }}
        .kpi-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 0; }}
        .kpi-grid table {{ width: 100%; border-collapse: collapse; }}
        .kpi-grid td {{ padding: 8px 12px; border-bottom: 1px solid var(--ing-border); font-size: 13px; }}
        .kpi-grid td:first-child {{ font-weight: 600; color: var(--ing-navy); }}
        .kpi-grid td:last-child {{ text-align: right; color: var(--ing-text); }}
        /* Data Tables General */
        .data-section-container {{ width: 100%; border-radius: 4px; margin-bottom: 30px; }}
        .ing-data-table {{
            width: 100%;
            border-collapse: collapse;
            font-family: 'INGMeWeb-Regular.ttf';
            font-size: 11px;
            background-color: #ffffff;
            table-layout: auto;
        }}
        .ing-data-table th {{
            background-color: var(--ing-navy);
            color: #ffffff;
            font-weight: 400;
            padding: 8px 6px;
            border: 1px solid #e0e0e0;
            white-space: normal;
        }}
        .ing-data-table td {{
            padding: 8px 6px;
            border: 1px solid #e0e0e0;
            color: #333333;
            white-space: nowrap;
        }}
        /* Wallet Share Specific */
        .wallet-table tbody tr:nth-child(1) {{ background-color: #fdfdfd; }}
        .wallet-table tbody tr:nth-child(2) {{ background-color: #fafafa; }}
        .wallet-table tbody tr:nth-child(3) {{ background-color: #f5f5f5; color: #666666; }}
        .wallet-table tbody tr:last-child {{ background-color: #fff5ec; font-weight: bold; color: #ff6600; }}
        .wallet-table tbody tr:hover {{ background-color: #f1f4f9; }}
        /* Sales Accounts Specific */
        .sales-table tbody tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .sales-table tbody tr:hover {{ background-color: #f1f4f9; }}
        /* Disclaimer — PDF raporundaki kapanış bölümüyle aynı format */
        .disclaimer-hr {{ border: none; height: 1px; background: var(--ing-navy); margin: 24px 0 8px; }}
        .disclaimer {{ font-size: 10px; color: var(--ing-text-secondary); text-align: center; }}
        /* Report Body */
        .report-body h1 {{ color: var(--ing-navy); font-size: 20px; margin: 30px 0 10px; padding-bottom: 6px; border-bottom: 2px solid var(--ing-orange); }}
        .report-body h2 {{ color: var(--ing-orange); font-size: 16px; margin: 24px 0 8px; }}
        .report-body h3 {{ color: var(--ing-navy); font-size: 14px; margin: 16px 0 6px; }}
        .report-body p {{ margin: 8px 0; color: var(--ing-text); }}
        .report-body ul, .report-body ol {{ margin: 8px 0 8px 24px; }}
        .report-body li {{ margin: 4px 0; }}
        .report-body strong {{ color: var(--ing-navy); }}
        .report-body code {{ background: var(--ing-gray); padding: 2px 6px; border-radius: 3px; font-size: 12px; color: var(--ing-navy); }}
        .table-wrapper {{ overflow-x: auto; margin: 12px 0; }}
        .report-body table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .report-body th {{ background: var(--ing-navy); color: white; padding: 8px 10px; text-align: left; font-weight: 600; }}
        .report-body td {{ padding: 7px 10px; border-bottom: 1px solid var(--ing-border); }}
        .report-body tr:nth-child(even) td {{ background: var(--ing-gray); }}
        .ing-hr {{ border: none; height: 1px; background: var(--ing-orange); margin: 20px 0; }}
        .footer {{ background: var(--ing-navy); color: white; padding: 15px 40px; text-align: center; font-size: 11px; opacity: 0.9; margin-top: 40px; }}
        @media print {{
            .header, .meta-bar, .footer {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }}
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
    <span><strong>{meta_mizan_donem}:</strong> {mizan_donem}</span>
    <span><strong>{meta_platform}:</strong> Sales Intelligence Platform</span>
</div>
    <div class="container">
    {data_sections_html}
    <div class="report-body">
                    {report_body}
    </div>
    <hr class="disclaimer-hr">
    <p class="disclaimer">{disclaimer_text}</p>
</div>

<div class="footer">
    {footer_text}
</div>

</body>
</html>"""
    logger.info(f"[HTML] Generated report: {len(html)} chars")
    return html
