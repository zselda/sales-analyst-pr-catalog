"""
Agent 6: Sales Strategist — LLM-generated report with local calculation context

Generates a comprehensive 7-section Corporate Sales Strategy Report by:
1. Extracting all local calculations from quant_analyst.py (ratios, raw values, citations)
2. Building a rich data context with hesap kodu açıklama citations
3. Sending everything to LLM for intelligent reasoning and report generation

The LLM receives ALL calculated values and produces the final report with reasoning.
"""
"""
import logging
from agents.base import BaseAgent
from llm_config import invoke_llm, STRATEGIST_SYSTEM_PROMPT

logger = logging.getLogger("swarm.agents.strategist")


class SalesStrategistAgent(BaseAgent):
    name = "strategist"
    description = "Generate LLM-reasoned corporate sales strategy report from local calculations"
    required_inputs = ["financial_ratios", "network_data"]
    output_keys = ["strategy_report"]

    def execute(self, state: dict) -> dict:
        ratios = state.get("financial_ratios", {})
        network = state.get("network_data", {})

        # ── Extract calculated metrics from quant_analyst ──
        gm = ratios.get("gross_margin", {}).get("value", 0)
        om = ratios.get("operating_margin", {}).get("value", 0)
        cr = ratios.get("current_ratio", {}).get("value", 0)
        qr = ratios.get("quick_ratio", {}).get("value", 0)
        dte = ratios.get("debt_to_equity", {}).get("value", 0)
        bdr = ratios.get("bank_debt_ratio", {}).get("value", 0)
        fer = ratios.get("financial_expense_ratio", {}).get("value", 0)
        pr = ratios.get("pos_commission_ratio", {}).get("value", 0)
        cp = ratios.get("collection_period", {}).get("value", 0)
        pp = ratios.get("payment_period", {}).get("value", 0)

        # Raw value extractor
        def raw_val(ratio_name: str, key: str) -> float:
            raw = ratios.get(ratio_name, {}).get("raw_values", {}).get(key, 0)
            if isinstance(raw, dict):
                return raw.get("value", 0)
            return raw

        # Hesap kodu citation formatter
        def raw_cite(ratio_name: str, key: str) -> str:
            raw = ratios.get(ratio_name, {}).get("raw_values", {}).get(key, {})
            if isinstance(raw, dict) and "hesap_kodu" in raw:
                return f"{raw['hesap_kodu']}-{raw['hesap_kodu_aciklama']}: ₺{raw['value']:,.0f}"
            if isinstance(raw, (int, float)):
                return f"₺{raw:,.0f}"
            return "N/A"

        # ── Extract all raw values ──
        revenue = raw_val("gross_margin", "revenue_600")
        cogs = raw_val("gross_margin", "cogs_620")
        gross_profit = revenue - cogs
        trade_recv = raw_val("collection_period", "trade_receivables")
        trade_pay = raw_val("payment_period", "trade_payables")
        checks_101 = raw_val("quick_ratio", "received_checks_101")
        checks_103 = raw_val("quick_ratio", "given_checks_103")
        pos_raw = raw_val("pos_commission_ratio", "pos_komisyon_780_01")
        fin_exp = raw_val("financial_expense_ratio", "finansman_giderleri_780")
        bank_loans_300 = raw_val("bank_debt_ratio", "banka_kredileri_kv_300")
        bank_loans_400 = raw_val("bank_debt_ratio", "banka_kredileri_uv_400")
        other_fin_309 = raw_val("bank_debt_ratio", "diger_mali_borclar_309")
        total_bank_loans = raw_val("bank_debt_ratio", "total_bank_loans")
        cash_cycle = cp - pp

        # ── Build hesap kodu citation lines ──
        cite_revenue = raw_cite("gross_margin", "revenue_600")
        cite_cogs = raw_cite("gross_margin", "cogs_620")
        cite_recv_120 = raw_cite("collection_period", "trade_receivables_120")
        cite_recv_121 = raw_cite("collection_period", "trade_receivables_121")
        cite_pay_320 = raw_cite("payment_period", "trade_payables_320")
        cite_pay_321 = raw_cite("payment_period", "trade_payables_321")
        cite_checks_101 = raw_cite("quick_ratio", "received_checks_101")
        cite_checks_103 = raw_cite("quick_ratio", "given_checks_103")
        cite_loans_300 = raw_cite("bank_debt_ratio", "banka_kredileri_kv_300")
        cite_loans_400 = raw_cite("bank_debt_ratio", "banka_kredileri_uv_400")
        cite_other_309 = raw_cite("bank_debt_ratio", "diger_mali_borclar_309")
        cite_fin_780 = raw_cite("financial_expense_ratio", "finansman_giderleri_780")
        cite_pos_780_01 = raw_cite("pos_commission_ratio", "pos_komisyon_780_01")

        # ── Competitor Bank Analysis ──
        cb = ratios.get("competitor_banks", {})

        def fmt_cb(shares):
            if not shares:
                return "No data"
            return ", ".join(
                f"{s['name']} (₺{s['balance']:,.0f} - %{s.get('share_pct', 0):.1f})"
                for s in shares[:5]
            )

        cb_102 = fmt_cb(cb.get("102", []))
        cb_300 = fmt_cb(cb.get("300", []))
        cb_400 = fmt_cb(cb.get("400", []))

        # ── Network Data ──
        ns = network.get("stats", {})
        nodes = network.get("nodes", [])
        custs = [n for n in nodes if n.get("type") == "customer"]
        supps = [n for n in nodes if n.get("type") == "supplier"]
        t_recv = ns.get("total_receivables", 0)
        t_pay = ns.get("total_payables", 0)

        cust_str = ", ".join(
            c["label"] + " (₺" + f'{c["balance"]:,.0f}' + ")" for c in custs[:10]
        ) or "No customer data"
        supp_str = ", ".join(
            s["label"] + " (₺" + f'{s["balance"]:,.0f}' + ")" for s in supps[:10]
        ) or "No supplier data"

        quant_interp = ratios.get("llm_interpretation", "")

        company_name = state.get("company_name", "Company")
        company_tax_id = state.get("tax_id", "1234567890")

        # ── Build comprehensive prompt with ALL local calculations ──
        prompt = (
            f"Generate a COMPREHENSIVE Corporate Sales Strategy Report for "
            f"{company_name} (Tax ID: {company_tax_id}).\n\n"

            f"## 📊 CALCULATED KEY PERFORMANCE INDICATORS\n"
            f"| Metric | Value | Metric | Value |\n|---|---|---|---|\n"
            f"| Gross Margin | {gm}% | Operating Margin | {om}% |\n"
            f"| Current Ratio | {cr}x | Quick Ratio | {qr}x |\n"
            f"| Bank Debt Ratio | {bdr}% | Debt-to-Equity | {dte}x |\n"
            f"| Collection Period | {cp} days | Payment Period | {pp} days |\n"
            f"| Fin. Expense Ratio | {fer}% | POS Commission Ratio | {pr}% |\n"
            f"| Cash Conversion Cycle | {cash_cycle:.0f} days | Gross Profit | ₺{gross_profit:,.0f} |\n\n"

            f"## 💰 RAW ACCOUNT BALANCES (with Hesap Kodu Açıklama)\n"
            f"### Profitability\n"
            f"- {cite_revenue}\n"
            f"- {cite_cogs}\n"
            f"- Gross Profit: ₺{gross_profit:,.0f}\n\n"
            f"### Liquidity & Working Capital\n"
            f"- {cite_recv_120} | {cite_recv_121} → Total Trade Receivables: ₺{trade_recv:,.0f}\n"
            f"- {cite_pay_320} | {cite_pay_321} → Total Trade Payables: ₺{trade_pay:,.0f}\n"
            f"- {cite_checks_101} | {cite_checks_103}\n\n"
            f"### Leverage & Debt\n"
            f"- {cite_loans_300} | {cite_loans_400} | {cite_other_309}\n"
            f"- Total Bank Loans: ₺{total_bank_loans:,.0f}\n\n"
            f"### Transactional Cost\n"
            f"- {cite_fin_780}\n"
            f"- {cite_pos_780_01}\n\n"

            f"## 🏦 COMPETITOR BANK WALLET SHARE\n"
            f"- **102-BANKALAR (Deposits):** {cb_102}\n"
            f"- **300-BANKA KREDİLERİ KV (ST Loans):** {cb_300}\n"
            f"- **400-BANKA KREDİLERİ UV (LT Loans):** {cb_400}\n\n"

            f"## 🔍 QUANT ANALYST INTELLIGENCE\n"
            f"{str(quant_interp)[:1000]}\n\n"

            f"## 🌐 COMMERCIAL NETWORK\n"
            f"**Customers ({len(custs)}):** {cust_str}\n"
            f"**Suppliers ({len(supps)}):** {supp_str}\n"
            f"**Total Receivables:** ₺{t_recv:,.0f} | **Total Payables:** ₺{t_pay:,.0f}\n\n"

            f"## INSTRUCTIONS\n"
            f"Using ALL the calculated values and raw data above, generate a report "
            f"with deep analytical REASONING for each section. Do NOT just repeat the numbers — "
            f"explain what they MEAN for the company and what actions the bank should take.\n\n"
            f"Structure the report into exactly 7 sections:\n"
            f"1. Executive Summary (KPI dashboard table, 3-sentence overall assessment with reasoning, top 3 priorities)\n"
            f"2. Deep Financial Health Analysis (4 pillars — cite every hesap kodu açıklama, explain implications)\n"
            f"3. Competitor Bank Intelligence (wallet share analysis, identify refinancing/takeover targets)\n"
            f"4. Sales Opportunities (prioritized by revenue potential, backed by specific data points)\n"
            f"5. Product Recommendations Matrix (table: client need → product → data evidence → estimated impact)\n"
            f"6. Risk Assessment & Mitigation (severity ratings with evidence from calculated ratios)\n"
            f"7. Action Plan & Timeline (Week 1, Month 1, Quarter 1 — concrete actions with data reasoning)"
        )

        # ── Call LLM for report generation ──
        try:
            report = invoke_llm(STRATEGIST_SYSTEM_PROMPT, prompt, temperature=0.5, max_tokens=4096)
            self.metrics.record_llm_call(tokens=len(report.split()))
            logger.info(f"✅ LLM report generated: {len(report)} chars")
        except Exception as e:
            logger.warning(f"⚠️ LLM call failed ({e}), returning error report")
            report = (
                f"# ⚠️ Strategy Report — LLM Unavailable\n\n"
                f"The LLM service was unavailable during report generation.\n"
                f"**Error:** {e}\n\n"
                f"## Key Metrics (from local calculations)\n"
                f"- Gross Margin: {gm}% | Operating Margin: {om}%\n"
                f"- Current Ratio: {cr}x | Quick Ratio: {qr}x\n"
                f"- Collection Period: {cp} days | Payment Period: {pp} days\n"
                f"- Financial Expense Ratio: {fer}% | POS Commission: {pr}%\n"
                f"- Bank Debt Ratio: {bdr}% | Debt-to-Equity: {dte}x\n"
                f"- Total Bank Loans: ₺{total_bank_loans:,.0f}\n\n"
                f"## Account Balances\n"
                f"- {cite_revenue}\n- {cite_cogs}\n"
                f"- {cite_fin_780}\n- {cite_pos_780_01}\n\n"
                f"**Please retry when the LLM service is available for a full analysis report with reasoning.**\n"
            )

        return {"strategy_report": report}


# Module-level callable for LangGraph
sales_strategist_agent = SalesStrategistAgent()
"""

"""
Agent 6: Sales & Credit Strategist — LLM-generated report with local calculation context

Generates a comprehensive 8-section Corporate Credit & Sales Strategy Report in ENGLISH.
INCLUDES CREDIT LIMIT PROPOSAL: Calculates Working Capital Need based on 
Cash Conversion Cycle and sets Covenants based on Insider Lending/Check Risks.

The LLM receives ALL calculated values and produces the final report with banking reasoning.
"""

import logging
from agents.base import BaseAgent
from llm_config import invoke_llm, STRATEGY_AGENT_SYSTEM_PROMPT

logger = logging.getLogger("swarm.agents.strategist")


class SalesStrategistAgent(BaseAgent):
    name = "strategist"
    description = "Generate LLM-reasoned corporate credit limit and sales strategy report in English"
    required_inputs = ["financial_ratios", "network_data"]
    output_keys = ["strategy_report"]

    def execute(self, state: dict) -> dict:
        ratios = state.get("financial_ratios", {})
        network = state.get("network_data", {})

        # ── Extract calculated metrics from quant_analyst ──
        gm = ratios.get("gross_margin", {}).get("value", 0)
        om = ratios.get("operating_margin", {}).get("value", 0)
        cr = ratios.get("current_ratio", {}).get("value", 0)
        qr = ratios.get("quick_ratio", {}).get("value", 0)
        dte = ratios.get("debt_to_equity", {}).get("value", 0)
        bdr = ratios.get("bank_debt_ratio", {}).get("value", 0)
        fer = ratios.get("financial_expense_ratio", {}).get("value", 0)
        pr = ratios.get("pos_commission_ratio", {}).get("value", 0)
        
        # New Banking Metrics
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

        # Hesap kodu citation formatter
        def raw_cite(ratio_name: str, key: str) -> str:
            raw = ratios.get(ratio_name, {}).get("raw_values", {}).get(key, {})
            if isinstance(raw, dict) and "hesap_kodu" in raw:
                return f"{raw['hesap_kodu']}-{raw['hesap_kodu_aciklama']}: ₺{raw['value']:,.0f}"
            if isinstance(raw, (int, float)):
                return f"₺{raw:,.0f}"
            return "N/A"

        # ── Extract all raw values ──
        revenue = raw_val("gross_margin", "revenue_600")
        cogs = raw_val("gross_margin", "cogs_620")
        gross_profit = revenue - cogs
        
        # Safe extraction for new raw values (fallback to 0 if not present in older states)
        ebitda_proxy = ratios.get("llm_interpretation", "") 
        trade_recv = raw_val("collection_period", "trade_receivables")
        trade_pay = raw_val("payment_period", "trade_payables")
        total_bank_loans = raw_val("bank_debt_ratio", "total_bank_loans")

        # ── Build hesap kodu citation lines ──
        cite_revenue = raw_cite("gross_margin", "revenue_600")
        cite_cogs = raw_cite("gross_margin", "cogs_620")
        cite_recv_120 = raw_cite("collection_period", "trade_receivables_120")
        cite_recv_121 = raw_cite("collection_period", "trade_receivables_121")
        cite_pay_320 = raw_cite("payment_period", "trade_payables_320")
        cite_checks_101 = raw_cite("quick_ratio", "received_checks_101")
        cite_checks_103 = raw_cite("quick_ratio", "given_checks_103")
        cite_loans_300 = raw_cite("bank_debt_ratio", "banka_kredileri_kv_300")
        cite_loans_400 = raw_cite("bank_debt_ratio", "banka_kredileri_uv_400")

        # ── Competitor Bank Analysis ──
        cb = ratios.get("competitor_banks", {})

        def fmt_cb(shares):
            if not shares:
                return "No data"
            return ", ".join(
                f"{s['name']} (₺{s['balance']:,.0f} - %{s.get('share_pct', 0):.1f})"
                for s in shares[:5]
            )

        cb_102 = fmt_cb(cb.get("102", []))
        cb_300 = fmt_cb(cb.get("300", []))
        cb_400 = fmt_cb(cb.get("400", []))

        # ── Network Data ──
        ns = network.get("stats", {})
        nodes = network.get("nodes", [])
        custs = [n for n in nodes if n.get("type") == "customer"]
        supps = [n for n in nodes if n.get("type") == "supplier"]
        t_recv = ns.get("total_receivables", 0)
        t_pay = ns.get("total_payables", 0)
        concentration_flag = ns.get("concentration_flag", False)
        concentration_warning = ""
        if concentration_flag:
            concentration_warning = (
                f"\n⚠️ CRITICAL SYSTEM WARNING: High Concentration Risk Detected! "
                f"Max Customer Dependency is %{ns.get('max_customer_dependency_ratio', 0)*100:.1f} and "
                f"Max Supplier Dependency is %{ns.get('max_supplier_dependency_ratio', 0)*100:.1f}. "
                f"You MUST enforce the Ecosystem & Network Rule and demand their Mizan."
        )
        cust_str = ", ".join(c["label"] + " (₺" + f'{c["balance"]:,.0f}' + ")" for c in custs[:10]) or "No customer data"
        supp_str = ", ".join(s["label"] + " (₺" + f'{s["balance"]:,.0f}' + ")" for s in supps[:10]) or "No supplier data"

        quant_interp = ratios.get("llm_interpretation", "")
        company_name = state.get("company_name", "Company")
        company_tax_id = state.get("tax_id", "1234567890")

        # Basic Working Capital Calculation for Prompt Context
        #period_days = ratios.get("donem_context", {}).get("period_days", 360)
        #estimated_wc_need = (ccc / period_days) * cogs if period_days > 0 and cogs > 0 else 0
        # Basic Working Capital Calculation for Prompt Context
        donem_ctx = ratios.get("donem_context", {})
        period_days = donem_ctx.get("period_days", 360)
        donem_label = donem_ctx.get("label", "Annual (12M)") # <--- added
        
        estimated_wc_need = (ccc / period_days) * cogs if period_days > 0 and cogs > 0 else 0
        # ── Build comprehensive prompt with ALL local calculations ──
        prompt = (
            f"Generate a COMPREHENSIVE Corporate Credit Limit & Sales Strategy Report for "
            f"{company_name} (Tax ID: {company_tax_id}).\n\n"
            f"IMPORTANT: WRITE THE ENTIRE FINAL REPORT IN PROFESSIONAL ENGLISH BANKING TERMINOLOGY.\n\n"
            
            f"## ⏱️ TEMPORAL CONTEXT\n"
            f"- **Data Period:** {donem_label} ({period_days} days)\n"
            f"- **Warning for LLM:** All flow metrics (Revenue, COGS, EBITDA, Fin. Expenses) represent this specific {donem_label} period. Do NOT evaluate absolute volumes as if they were a full calendar year unless it is a 12M period. Annualize these figures in your strategic reasoning where appropriate.\n\n"

            f"## 📊 CALCULATED KEY PERFORMANCE INDICATORS\n"
            f"| Metric | Value | Metric | Value |\n|---|---|---|---|\n"
            f"| Gross Margin | {gm}% | Operating Margin | {om}% |\n"
            f"| Current Ratio | {cr}x | Quick Ratio | {qr}x |\n"
            f"| Collection Period | {cp} days | Payment Period | {pp} days |\n"
            f"| Inventory Period | {inv_p} days | Cash Conv. Cycle | {ccc:.0f} days |\n"
            f"| Bank Debt Ratio | {bdr}% | Debt-to-Equity | {dte}x |\n"
            f"| Insider Lending (131/331) | {insider_r}% | Check Risk (103/102)| {check_r}x |\n"
            f"| Fin. Expense Ratio | {fer}% | POS Comm. Ratio | {pr}% |\n\n"

            f"## 💰 RAW ACCOUNT BALANCES & ESTIMATES\n"
            f"### Operational & Profitability\n"
            f"- {cite_revenue}\n"
            f"- {cite_cogs}\n"
            f"- Estimated Working Capital Need formula result: ₺{estimated_wc_need:,.0f}\n\n"
            f"### Liquidity & Risk Accounts\n"
            f"- {cite_checks_101} | {cite_checks_103}\n"
            f"- Insider accounts (131/331) details are present in Quant Analyst Intelligence.\n\n"
            f"### Leverage & Debt\n"
            f"- {cite_loans_300} | {cite_loans_400}\n"
            f"- Total Bank Loans: ₺{total_bank_loans:,.0f}\n\n"

            f"## 🏦 COMPETITOR BANK WALLET SHARE\n"
            f"- **102-BANKALAR (Deposits):** {cb_102}\n"
            f"- **300-BANKA KREDİLERİ KV (ST Loans):** {cb_300}\n"
            f"- **400-BANKA KREDİLERİ UV (LT Loans):** {cb_400}\n\n"

            f"## 🔍 QUANT ANALYST INTELLIGENCE\n"
            f"{str(quant_interp)[:1500]}\n\n"

            f"## 🌐 COMMERCIAL NETWORK\n"
            f"**Customers ({len(custs)}):** {cust_str}\n"
            f"**Suppliers ({len(supps)}):** {supp_str}\n"
            f"**Total Receivables:** ₺{t_recv:,.0f} | **Total Payables:** ₺{t_pay:,.0f}\n\n"
            f"{concentration_warning}\n"

            f"## INSTRUCTIONS\n"
            f"Using ALL the calculated values above, generate a report with deep analytical REASONING. "
            f"Act as a Senior Credit Allocation Manager. "
            f"Structure the report into exactly 8 sections in ENGLISH:\n\n"
            f"1. Executive Summary: KPI dashboard table, 3-sentence overall risk/reward assessment.\n"
            f"2. Financial Health & Cash Cycle Analysis: Cite explicit account codes (Hesap Kodu). Analyze the Cash Conversion Cycle ({ccc:.0f} days) and EBITDA proxies.\n"
            f"3. Hidden Risks & Capital Leakage: Analyze Insider Lending (131/331) and Check Risk Ratio (103 vs 102). Flag any window dressing or capital leakage.\n"
            f"4. Competitor Intelligence: Wallet share analysis, identify explicit refinancing/takeover targets based on 300/400 accounts.\n"
            f"5. Sales & Cross-Sell Opportunities: Prioritize by revenue potential (POS, FX, Cash Management, Payroll).\n"
            f"6. CREDIT PROPOSAL & STRUCTURING (**CRITICAL**):\n"
            f"   - Propose an explicit Working Capital Limit (in ₺) justified by the Estimated WC Need (₺{estimated_wc_need:,.0f}).\n"
            f"   - Break down the proposed limit into Cash Loans (Revolving/BCH), Non-Cash Loans (Letters of Guarantee), and Refinancing (Buyouts).\n"
            f"   - Establish strict Covenants based on the hidden risks (e.g., 'Account 131 must be reduced below X%', 'Monthly POS volume commitment').\n"
            f"7. Risk Assessment & Mitigation: Severity ratings with evidence from calculated ratios.\n"
            f"8. Action Plan: deliverables for the Relationship Manager."
        )

        # ── Call LLM for report generation ──
        try:
            report = invoke_llm(STRATEGY_AGENT_SYSTEM_PROMPT, prompt, temperature=0.3, max_tokens=4096)
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
                f"- Estimated Working Capital Need: ₺{estimated_wc_need:,.0f}\n\n"
                f"**Please retry when the LLM service is available for a full credit analysis and proposal.**\n"
            )

        return {"strategy_report": report}


# Module-level callable for LangGraph
sales_strategist_agent = SalesStrategistAgent()