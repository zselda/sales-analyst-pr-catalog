"""
Agent 6: Sales Strategist — LLM-generated report with local calculation context

Generates a comprehensive 7-section Corporate Sales Strategy Report by:
1. Extracting all local calculations from quant_analyst.py (ratios, raw values, citations)
2. Building a rich data context with hesap kodu açıklama citations
3. Sending everything to LLM for intelligent reasoning and report generation

The LLM receives ALL calculated values and produces the final report with reasoning.
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