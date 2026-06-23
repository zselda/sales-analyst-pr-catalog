"""
Agent 6: Sales & Credit Strategist — LLM-generated report with local calculation context
 
Generates a comprehensive 8-section Corporate Credit & Sales Strategy Report in ENGLISH.
INCLUDES CREDIT LIMIT PROPOSAL: Calculates Working Capital Need based on
Cash Conversion Cycle and sets Covenants based on Insider Lending/Check Risks.
 
Now includes:
- PRODUCT SIGNALS integration from product_analyst
- ING Bank Turkey–centric competitive positioning
- Compatibility with nested bank breakdown format from quant_analyst
 
The LLM receives ALL calculated values and produces the final report with banking reasoning.
"""
 
import json
import logging
from agents.base import BaseAgent
from agents.product_analyst import build_strategist_payload
from agents.few_shot_library import (
    classify_product_opportunities,
    build_few_shot_injection,
    build_recommendation_catalog,
    render_catalog_for_prompt,
)
from sector_analysis import compare_company_to_sector
from llm_config import invoke_llm, STRATEGY_AGENT_SYSTEM_PROMPT
 
logger = logging.getLogger("swarm.agents.strategist")
 
 
class SalesStrategistAgent(BaseAgent):
    name = "strategist"
    description = "Generate LLM-reasoned corporate credit limit and sales strategy report for ING Bank Turkey"
    required_inputs = ["financial_ratios", "network_data", "product_signals"]
    output_keys = ["strategy_report"]
 
    def execute(self, state: dict) -> dict:
        ratios = state.get("financial_ratios", {})
        network = state.get("network_data", {})
        product_signals = state.get("product_signals", {})
 
        # ── Extract calculated metrics from quant_analyst ──
        gm = ratios.get("gross_margin", {}).get("value", 0)
        om = ratios.get("operating_margin", {}).get("value", 0)
        cr = ratios.get("current_ratio", {}).get("value", 0)
        qr = ratios.get("quick_ratio", {}).get("value", 0)
        dte = ratios.get("debt_to_equity", {}).get("value", 0)
        bdr = ratios.get("bank_debt_ratio", {}).get("value", 0)
        fer = ratios.get("financial_expense_ratio", {}).get("value", 0)
        pr = ratios.get("pos_commission_ratio", {}).get("value", 0)
 
        # Banking Metrics
        cp = ratios.get("collection_period", {}).get("value", 0)
        pp = ratios.get("payment_period", {}).get("value", 0)
        inv_p = ratios.get("inventory_period", {}).get("value", 0)
        ccc = ratios.get("cash_conversion_cycle", {}).get("value", 0)
        insider_r = ratios.get("insider_lending_ratio", {}).get("value", 0)
        check_r = ratios.get("check_risk_ratio", {}).get("value", 0)
 
        # Raw value extractor
        def raw_val(ratio_name: str, key: str) -> float:
            raw = ratios.get(ratio_name, {}).get("raw_values", {}).get(key, 0)
            if isinstance(raw, dict):
                return raw.get("value", 0)
            return raw
 
        # ── Extract all raw values ──
        net_revenue = raw_val("gross_margin", "net_revenue")
        cogs = raw_val("gross_margin", "cogs_62x")
        gross_profit = raw_val("gross_margin", "gross_profit")
        op_expenses = raw_val("operating_margin", "op_expenses_63x")
        operating_profit = raw_val("operating_margin", "operating_profit")
        total_bank_loans = raw_val("bank_debt_ratio", "total_bank_loans")
        fin_expenses_780 = raw_val("financial_expense_ratio", "finansman_giderleri_780")
        trade_recv = raw_val("collection_period", "trade_receivables")
        trade_pay = raw_val("payment_period", "trade_payables")
        # Balance sheet raw values
        current_assets = raw_val("current_ratio", "current_assets") or raw_val("insider_lending_ratio", "current_assets")
        total_assets = raw_val("insider_lending_ratio", "total_assets")
        total_equity = raw_val("debt_to_equity", "total_equity")
        insider_131 = raw_val("insider_lending_ratio", "insider_lending_131")
        insider_331 = raw_val("insider_lending_ratio", "insider_borrowing_331")
        given_checks_103 = raw_val("check_risk_ratio", "given_checks_103")
        banks_102_total = raw_val("check_risk_ratio", "banks_102_total")
        # Cash flow
        cash_flow = ratios.get("cash_flow_summary", {})
        period_net_cash = cash_flow.get("period_net_movement", 0)
        future_net_position = cash_flow.get("future_net_position", 0)
 
        # ══════════════════════════════════════════════════════════════
        # COMPETITOR BANK ANALYSIS — Handles nested category format
        # ══════════════════════════════════════════════════════════════
        cb = ratios.get("competitor_banks", {})
 
        def fmt_cb(shares: list) -> str:
            """Format nested bank breakdown from quant_analyst.
            Each entry has: category_name, balance, share_of_total_pct, sub_accounts.
            """
            if not shares:
                return "No data"
            lines = []
            for s in shares[:5]:
                name = s.get("category_name", s.get("name", "Unknown"))
                balance = s.get("balance", 0)
                pct = s.get("share_of_total_pct", s.get("share_pct", 0))
                lines.append(f"{name} (₺{balance:,.0f} - %{pct:.1f})")
                # Include top 2 sub-accounts if present
                subs = s.get("sub_accounts", [])
                for sub in subs[:2]:
                    lines.append(f"  └ {sub.get('name', '?')}: ₺{sub.get('balance', 0):,.0f}")
            return "\n".join(lines)
 
        def check_ing_presence(shares: list) -> str:
            """Check if ING Bank appears in competitor bank sub-accounts."""
            ing_keywords = ["ING", "İNG"]
            for s in shares:
                cat_name = s.get("category_name", "")
                if any(kw in cat_name.upper() for kw in ing_keywords):
                    return f"✅ ING PRESENT: {cat_name} (₺{s.get('balance', 0):,.0f}, %{s.get('share_of_total_pct', 0):.1f})"
                for sub in s.get("sub_accounts", []):
                    sub_name = sub.get("name", "")
                    if any(kw in sub_name.upper() for kw in ing_keywords):
                        return f"✅ ING PRESENT (sub): {sub_name} (₺{sub.get('balance', 0):,.0f})"
            return "❌ ING NOT PRESENT — New client acquisition opportunity"
 
        cb_102 = fmt_cb(cb.get("102", []))
        cb_300 = fmt_cb(cb.get("300", []))
        cb_400 = fmt_cb(cb.get("400", []))
 
        # ING presence check
        ing_102 = check_ing_presence(cb.get("102", []))
        ing_300 = check_ing_presence(cb.get("300", []))
        ing_400 = check_ing_presence(cb.get("400", []))
 
        # ── Network Data ──
 
        ns = network.get("stats", {})
        nodes = network.get("nodes", [])
        # Filter and sort by balance descending to get the highest balance items first
        custs = sorted([n for n in nodes if n.get("type") == "customer"], key=lambda x: x.get("balance", 0), reverse=True)
        supps = sorted([n for n in nodes if n.get("type") == "supplier"], key=lambda x: x.get("balance", 0), reverse=True)
        t_recv = ns.get("total_receivables", 0)
        t_pay = ns.get("total_payables", 0)
        concentration_flag = ns.get("concentration_flag", False)
        concentration_warning = ""
        if concentration_flag:
           concentration_warning = (
               f"\n⚠️ CRITICAL: High Concentration Risk! "
               f"Max Customer Dependency: %{ns.get('max_customer_dependency_ratio', 0)*100:.1f}, "
               f"Max Supplier Dependency: %{ns.get('max_supplier_dependency_ratio', 0)*100:.1f}. "
               f"MANDATE: Check their KKB Score (Credit Bureau Score) for risk monitoring and tracking purposes. "
               f"Suggest acquiring them as new customers to leverage cash flow, deposit income."
               #f"MANDATE: Request their Mizan for contagion risk assessment."
           )
        # Extracts the top 1 highest balance items from the pre-sorted lists
        cust_str = ", ".join(c["label"] + " (₺" + f'{c["balance"]:,.0f}' + ")" for c in custs[:1]) or "No customer data"
        supp_str = ", ".join(s["label"] + " (₺" + f'{s["balance"]:,.0f}' + ")" for s in supps[:1]) or "No supplier data"
 
 
        quant_interp = ratios.get("llm_interpretation", "")
        company_name = state.get("company_name", "Company")
        company_tax_id = state.get("tax_id", "1234567890")
 
        # ── Temporal Context ──
        donem_ctx = ratios.get("donem_context", {})
        period_days = donem_ctx.get("period_days", 360)
        donem_label = donem_ctx.get("label", "Annual (12M)")
 
        estimated_wc_need = (ccc / period_days) * cogs if period_days > 0 and cogs > 0 else 0
 
        # ══════════════════════════════════════════════════════════════
        # PRODUCT SIGNALS SECTION — from product_analyst (ALL 5 COLUMNS)
        # ══════════════════════════════════════════════════════════════
        product_signal_lines = []
        product_interp = ""
        sector = state.get('sector', 'General')
        if product_signals:
            product_interp = product_signals.get("llm_interpretation", "")
            for name, data in product_signals.items():
                if name == "llm_interpretation":
                    continue
                if isinstance(data, dict) and (data.get("balance", 0) != 0 or data.get("volume", 0) != 0):
                    mapping_str = ", ".join(f"'{c}-{n}'" for c, n in data.get("account_mapping", {}).items())
                    product_signal_lines.append(
                        f"- **{name}**: Debit=₺{data.get('debit',0):,.0f}, Credit=₺{data.get('credit',0):,.0f}, "
                        f"BalDebit=₺{data.get('balance_debit',0):,.0f}, BalCredit=₺{data.get('balance_credit',0):,.0f}, "
                        f"Net=₺{data['balance']:,.0f}, Vol=₺{data['volume']:,.0f}, "
                        f"Accounts: {{{mapping_str}}}"
                    )
        product_signal_text = "\n".join(product_signal_lines) if product_signal_lines else "No active product signals detected."
 
        # Build compressed JSON payload for LLM context
        strategist_payload = build_strategist_payload(
            {"financial_ratios": ratios},
            {"product_signals": product_signals}
        )

        # ── TCMB SECTOR BENCHMARK (sector_analysis foundation) ──
        # Mizan-derived ratios are the primary source; bank-DB metrics
        # (local_db en_description keys) fill the gaps (net margin, leverage).
        sector_comparison = compare_company_to_sector(
            ratios, sector, db_metrics=state.get("db_financial_metrics") or {}
        )
        sector_enabled = sector_comparison.get("enabled", True)
        if not sector_enabled:
            # OPTIONAL: TCMB benchmark data is stale/disabled — the report
            # is produced without the sector-comparison section.
            logger.info(
                f"📐 Sector benchmark section omitted for '{company_name}' "
                f"(sector analysis disabled / benchmark data not current)."
            )
        elif sector_comparison.get("is_fallback"):
            # FALLBACK PATH: sector prediction absent or not in the TCMB
            # benchmark set — the report is generated without sector info.
            logger.warning(
                f"⚠️ Strategy report for '{company_name}' generated WITHOUT matched "
                f"sector info (predicted sector: '{sector}') — TCMB 'General' "
                f"aggregate and General product priorities used as fallback."
            )
        else:
            logger.info(
                f"📐 Sector benchmark matched: {sector_comparison['matched_sector']} "
                f"(secondary: {sector_comparison['secondary_sectors'] or '—'}, "
                f"modifiers: {sector_comparison['modifiers'] or '—'})"
            )
        # Only inject the benchmark section when sector analysis is enabled
        sector_block = (
            f"{sector_comparison['markdown']}\n"
            f"Use this TCMB sector benchmark to contextualize the company's financial health "
            f"(better/worse than sector) and to sharpen sector-driven product prioritization "
            f"in Sections 1, 3 and 6.\n\n"
        ) if sector_enabled and sector_comparison.get("markdown") else ""

        # ── SELECTIVE FEW-SHOT INJECTION + CURRENT-USAGE EXCLUSION ──
        # Same classifier as product_analyst: only scenarios with detected
        # signals are injected; products already used per the local bank DB
        # (db_product_flags == 1) are excluded from recommendations.
        db_product_flags = state.get("db_product_flags") or {}
        classification = classify_product_opportunities(
            product_signals, sector=sector, product_flags=db_product_flags
        )
        injection = build_few_shot_injection(classification, sector=sector)

        # ── DETERMINISTIC RECOMMENDATION CATALOG (stable matrix rows) ──
        # Built by product_analyst (all active signals + sector cross-sell
        # gaps, minus already-used products). The matrix MUST render exactly
        # these rows — this is what stabilizes the row count across runs.
        catalog = product_signals.get("recommendation_catalog")
        if not isinstance(catalog, dict) or "active" not in catalog:
            catalog = build_recommendation_catalog(
                product_signals, sector=sector, product_flags=db_product_flags
            )
        catalog_prompt = render_catalog_for_prompt(catalog)
        logger.info(
            f"📋 Matrix catalog: {catalog.get('total_rows', 0)} fixed rows "
            f"({len(catalog.get('active', []))} active, "
            f"{len(catalog.get('cross_sell', []))} cross-sell)"
        )

        system_prompt = (
            STRATEGY_AGENT_SYSTEM_PROMPT
            + injection["system_addition"]
            + catalog_prompt["system_addition"]
        )

        prompt = (
            f"Generate the COMPREHENSIVE Sales Strategy Report for **{company_name}** (Tax ID: {company_tax_id}).\n\n"
            f"### ⏱️ TEMPORAL CONTEXT\n"
            f"- **Data Period:** {donem_label} ({period_days} days)\n"
            f"- **Warning:** Flow metrics represent this specific period. Annualize where appropriate.\n\n"
            f"### 🏦 COMPETITOR BANK WALLET SHARE\n"
            f"**102-BANKALAR (Deposits):**\n{cb_102}\n*ING Status:* {ing_102}\n\n"
            f"**300-BANKA KREDİLERİ KV (ST Loans):**\n{cb_300}\n*ING Status:* {ing_300}\n\n"
            f"**400-BANKA KREDİLERİ UV (LT Loans):**\n{cb_400}\n*ING Status:* {ing_400}\n\n"
            f"### 📦 PRODUCT SIGNALS & SECTOR\n"
            f"**Sector:** {sector} (Prioritize sector-relevant products)\n"
            f"**Product Analyst Intelligence:**\n{str(product_interp)}\n\n"
            f"{sector_block}"
            f"**Compressed Strategist Payload (JSON):**\n```json\n{strategist_payload}\n```\n\n"
            f"### 🔍 QUANT ANALYST INTELLIGENCE\n"
            f"{str(quant_interp)}\n\n"
            f"### 🌐 COMMERCIAL NETWORK & METRICS\n"
            f"**Customers ({len(custs)}):** {cust_str}\n"
            f"**Suppliers ({len(supps)}):** {supp_str}\n"
            f"**Receivables/Payables:** Total Recv: ₺{t_recv:,.0f} | Total Pay: ₺{t_pay:,.0f}\n"
            f"**Concentration Warning:** {concentration_warning}\n\n"
            f"### 📊 SPECIFIC FINANCIAL DATA FOR REPORT SECTIONS\n"
            f"- **For Refinancing Section:** Estimated WC Need is ₺{estimated_wc_need:,.0f}. Insider Lending (131: ₺{insider_131:,.0f} / 331: ₺{insider_331:,.0f}). Check Risk (103: ₺{given_checks_103:,.0f} vs 101: ₺{banks_102_total:,.0f}).\n"
            f"- **For Financial Health Section:** CCC is {ccc:.0f} days. Net Period Cash is ₺{period_net_cash:,.0f}. Future Position is ₺{future_net_position:,.0f}.\n\n"
            f"### EXECUTION\n"
            f"Using ALL the provided data above, generate the report STRICTLY following the 6-section structure and rules defined in your system prompt. Ensure deep analytical reasoning.\n"
            + injection["user_addition"]
            + catalog_prompt["user_addition"]
        )
 
#         # ── Build comprehensive prompt with ALL local calculations ──
#         prompt = (
#             f"Generate a COMPREHENSIVE Corporate Credit Limit & Sales Strategy Report for "
#             f"**{company_name}** (Tax ID: {company_tax_id}).\n\n"
#             f"IMPORTANT: WRITE THE ENTIRE FINAL REPORT IN PROFESSIONAL ENGLISH BANKING TERMINOLOGY.\n"
#             f"This report is for **ING Bank Turkey** Relationship Manager.\n\n"
# 
#             f"## ⏱️ TEMPORAL CONTEXT\n"
#             f"- **Data Period:** {donem_label} ({period_days} days)\n"
#             f"- **Warning:** Flow metrics represent this specific {donem_label} period. "
#             f"Annualize in strategic reasoning where appropriate.\n\n"
# 
#             # f"## 📊 CALCULATED KEY PERFORMANCE INDICATORS\n"
#             # f"| Metric | Value | Metric | Value |\n|---|---|---|---|\n"
#             # f"| Gross Margin | {gm}% | Operating Margin | {om}% |\n"
#             # f"| Current Ratio | {cr}x | Quick Ratio | {qr}x |\n"
#             # f"| Collection Period | {cp} days | Payment Period | {pp} days |\n"
#             # f"| Inventory Period | {inv_p} days | Cash Conv. Cycle | {ccc:.0f} days |\n"
#             # f"| Bank Debt Ratio | {bdr}% | Debt-to-Equity | {dte}x |\n"
#             # f"| Insider Lending (131/331) | {insider_r}% | Check Risk (103/102)| {check_r}x |\n"
#             # f"| Fin. Expense Ratio | {fer}% | POS Comm. Ratio | {pr}% |\n\n"
# # 
#             # f"## 💰 RAW ACCOUNT BALANCES & ESTIMATES\n"
#             # f"- Net Revenue: ₺{net_revenue:,.0f} | COGS: ₺{cogs:,.0f} | Gross Profit: ₺{gross_profit:,.0f}\n"
#             # f"- Operating Expenses (63x): ₺{op_expenses:,.0f} | Operating Profit: ₺{operating_profit:,.0f}\n"
#             # f"- Current Assets: ₺{current_assets:,.0f} | Total Assets: ₺{total_assets:,.0f} | Total Equity: ₺{total_equity:,.0f}\n"
#             # f"- Trade Receivables: ₺{trade_recv:,.0f} | Trade Payables: ₺{trade_pay:,.0f}\n"
#             # f"- Total Bank Loans: ₺{total_bank_loans:,.0f} | Fin. Expenses (780): ₺{fin_expenses_780:,.0f}\n"
#             # f"- Insider Lending (131): ₺{insider_131:,.0f} | Insider Borrowing (331): ₺{insider_331:,.0f}\n"
#             # f"- Given Checks (103): ₺{given_checks_103:,.0f} | Bank Deposits (102): ₺{banks_102_total:,.0f}\n"
#             # f"- Period Net Cash Movement: ₺{period_net_cash:,.0f} | Future Net Position: ₺{future_net_position:,.0f}\n"
#             # f"- Estimated Working Capital Need: ₺{estimated_wc_need:,.0f}\n\n"
# # 
#             f"## 🏦 COMPETITOR BANK WALLET SHARE\n"
#             f"### 102-BANKALAR (Deposits):\n{cb_102}\n"
#             f"**ING Status (Deposits):** {ing_102}\n\n"
#             f"### 300-BANKA KREDİLERİ KV (ST Loans):\n{cb_300}\n"
#             f"**ING Status (ST Loans):** {ing_300}\n\n"
#             f"### 400-BANKA KREDİLERİ UV (LT Loans):\n{cb_400}\n"
#             f"**ING Status (LT Loans):** {ing_400}\n\n"
# 
#             f"## 📦 PRODUCT SIGNALS — ALL 5 COLUMNS (from Product Analyst)\n"
#             # f"**Company Sector: {sector}** — Prioritize sector-relevant products.\n\n"
#             # f"{product_signal_text}\n\n"
#             f"### Product Analyst Intelligence:\n"
#             f"{str(product_interp)}\n\n"
# 
#             f"### Compressed Strategist Payload (JSON):\n"
#             f"```json\n{strategist_payload}\n```\n\n"
# 
#             f"## 🔍 QUANT ANALYST INTELLIGENCE\n"
#             f"{str(quant_interp)}\n\n"
# 
#             f"## 🌐 COMMERCIAL NETWORK\n"
#             f"**Customers ({len(custs)}):** {cust_str}\n"
#             f"**Suppliers ({len(supps)}):** {supp_str}\n"
#             f"**Total Receivables:** ₺{t_recv:,.0f} | **Total Payables:** ₺{t_pay:,.0f}\n\n"
#             f"{concentration_warning}\n"
# 
#             f"## INSTRUCTIONS\n"
#             f"Using ALL the calculated values above, generate a report with deep analytical REASONING. "
#             
#             f"Act as a Senior Credit Allocation Manager at ING Bank Turkey. "
#             
#             f"CRITICAL INSTRUCTION: The PRODUCT SIGNALS & CROSS-SELL OPPORTUNITIES section is the most vital part of your output and should be the most detailed. "
#             
#             f"Structure the report into exactly 6 sections in ENGLISH:\n\n"
# 
#             f"1. EXECUTIVE SUMMARY: Make a summary overall reward assessment. List top priorities with reasonings and show them with bullet points. DO NOT mention or reference the attached JSON payload; extract and present the values directly.\n"
#             
#             f"2. COMPETITOR BANK ANALYSIS & ING POSITIONING: Use the ING status flags above in COMPETITOR BANK WALLET SHARE. Evaluate each product separately. Identify refinancing targets FOR ING Bank Turkey. Make wallet share analysis.\n"
#             
#             f"3. PRODUCT SIGNALS & CROSS-SELL OPPORTUNITIES (CORE CRITICAL SECTION): This section must have the highest weight and detail. Use the product signal data AND sector ({sector}) to prioritize by ING revenue potential. Cite explicit account codes. Provide granular recommendations for each detected signal. Summarize findings in Product Recommendations Matrix (table: client need → product → data evidence). In detail explanations, include all product recommendations. \n"
#             
#             f"4. REFINANCING: \n"
#             f"   - Analyze WC Need of ({estimated_wc_need:,.0f}) and propose appropriate solutions. Do not propose credit limit.\n"
#             f"   - Break down into Cash Loans, Non-Cash Loans, and Refinancing strategy.\n"
#             f"   - Integrate analysis of Insider Lending (131: {insider_131:,.0f} / 331: {insider_331:,.0f}), Check Risk (103: {given_checks_103:,.0f} vs 101: {banks_102_total:,.0f}), and Network concentration.\n"
#             
#             f"5. SALES ACTION PLAN:\n"
#             f"   PRIMARY ACTIONS (Revenue-generating): Credit limit proposal, product cross-sell, refinancing targets, key client meetings.\n"
#             # f"   SECONDARY ACTIONS (Risk mitigation): Mizan acquisition of dominant network entities, covenant monitoring, insider lending remediation.\n"
#             
#             f"6. FINANCIAL HEALTH & CASH CYCLE ANALYSIS: Cite explicit account codes. Analyze CCC ({ccc:.0f} days), EBITDA, and Cash Flow (Net Period: {period_net_cash:,.0f}, Future Position: {future_net_position:,.0f}).\n\n"
#             
#             f"   OVERRIDE: Only if liquidity crisis (QR<0.5), capital leakage (131>15%), or negative equity → 1st Priority = secure position.\n\n"
#             + FEW_SHOT_PROMPT_ADDITION
# 
# #             f"## INSTRUCTIONS\n"
# #             f"Using ALL the calculated values above, generate a report with deep analytical REASONING. "
# #             f"Act as a Senior Credit Allocation Manager at ING Bank Turkey. "
# #             f"CRITICAL INSTRUCTION: The COMPETITOR BANK INTELLIGENCE & ING POSITIONING, PRODUCT SIGNALS & CROSS-SELL OPPORTUNITIES, CREDIT PROPOSAL & STRUCTURING sections are the most important parts of your output.\n"
# #             f"Structure the report into exactly 8 sections in ENGLISH:\n\n"
# #             f"1. EXECUTIVE SUMMARY: KPI dashboard table, 3-sentence overall risk/reward assessment, top 3 priorities. DO NOT mention or reference the attached JSON payload; extract and present the values directly.\n"
# #             f"2. FINANCIAL HEALTH & CASH CYCLE ANALYSIS: Cite explicit account codes. Analyze CCC ({ccc:.0f} days), EBITDA, and Cash Flow (Net Period: ₺{period_net_cash:,.0f}, Future Position: ₺{future_net_position:,.0f}).\n"
# #             f"3. COMPETITOR BANK INTELLIGENCE & ING POSITIONING (**CRITICAL**): Use the ING status flags above. Identify refinancing targets FOR ING. Wallet share analysis.\n"
# #             f"4. PRODUCT SIGNALS & CROSS-SELL OPPORTUNITIES (**CRITICAL**): Use the product signal data AND sector ({sector}) to prioritize by ING revenue potential. Only follow IF→THINKING→ACTION→PROPOSAL reasoning, DO NOT include IF→THINKING→ACTION→PROPOSAL wording in the report. Cite explicit account codes.\n"
# #             f"5. CREDIT PROPOSAL & STRUCTURING (**CRITICAL**):\n"
# #             f"   - Propose Working Capital Limit justified by WC Need (₺{estimated_wc_need:,.0f}).\n"
# #             f"   - Break down into Cash Loans, Non-Cash Loans, and Refinancing.\n"
# #             f"   - Establish Covenants based on hidden risks.\n"
# #             f"6. HIDDEN RISKS, CAPITAL LEAKAGE & CONCENTRATION: Analyze Insider Lending (131: ₺{insider_131:,.0f} / 331: ₺{insider_331:,.0f}), Check Risk (103: ₺{given_checks_103:,.0f} vs 102: ₺{banks_102_total:,.0f}), and Network concentration.\n" ### 102 yerine 101 olarak değiştirelim
# #             f"7. RISK ASSESSMENT & MITIGATION: Severity ratings with evidence from calculated ratios.\n"
# #             f"8. CRITICAL ACTION PLAN (Two-Tier):\n"
# #             f"   **PRIMARY ACTIONS** (Revenue-generating): Credit limit proposal, product cross-sell, refinancing targets, key client meetings.\n"
# #             f"   **SECONDARY ACTIONS** (Risk mitigation): Mizan acquisition of dominant network entities, covenant monitoring, insider lending remediation.\n"
# #             f"   OVERRIDE: Only if liquidity crisis (QR<0.5), capital leakage (131>15%), or negative equity → 1st Priority = secure position.\n\n"
# #             + FEW_SHOT_PROMPT_ADDITION
#         )
        print("strategist prompt: ", prompt)
 
        # ── Call LLM for report generation ──
        try:
            report = invoke_llm(system_prompt, prompt, temperature=0.2, max_tokens=4096)
            self.metrics.record_llm_call(tokens=len(report.split()))
            logger.info(f"✅ LLM report generated: {len(report)} chars")
        except Exception as e:
            logger.warning(f"⚠️ LLM call failed ({e}), returning error report")
            report = (
                f"# ⚠️ Strategy Report — LLM Unavailable\n\n"
                f"The LLM service was unavailable during report generation.\n"
                f"**Error:** {e}\n\n"
                f"## Key Metrics (Local Calculations)\n"
                f"- Cash Conversion Cycle: {ccc} days | Inventory Period: {inv_p} days\n"
                f"- Current Ratio: {cr}x | Quick Ratio: {qr}x\n"
                f"- Check Risk (103/102): {check_r}x | Insider Lending Risk: {insider_r}%\n"
                f"- Estimated Working Capital Need: ₺{estimated_wc_need:,.0f}\n"
                f"- Total Bank Loans: ₺{total_bank_loans:,.0f}\n\n"
                f"## Product Signals\n{product_signal_text}\n\n"
                f"## ING Positioning\n"
                f"- Deposits: {ing_102}\n"
                f"- ST Loans: {ing_300}\n"
                f"- LT Loans: {ing_400}\n\n"
                f"**Please retry when the LLM service is available for a full credit analysis and proposal.**\n"
            )
 
        return {"strategy_report": report}
 
 
# Module-level callable for LangGraph
sales_strategist_agent = SalesStrategistAgent()
