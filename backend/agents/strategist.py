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

import logging
from agents.base import BaseAgent
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
        total_bank_loans = raw_val("bank_debt_ratio", "total_bank_loans")
        trade_recv = raw_val("collection_period", "trade_receivables")
        trade_pay = raw_val("payment_period", "trade_payables")

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
        custs = [n for n in nodes if n.get("type") == "customer"]
        supps = [n for n in nodes if n.get("type") == "supplier"]
        t_recv = ns.get("total_receivables", 0)
        t_pay = ns.get("total_payables", 0)
        concentration_flag = ns.get("concentration_flag", False)
        concentration_warning = ""
        if concentration_flag:
            concentration_warning = (
                f"\n⚠️ CRITICAL: High Concentration Risk! "
                f"Max Customer Dependency: %{ns.get('max_customer_dependency_ratio', 0)*100:.1f}, "
                f"Max Supplier Dependency: %{ns.get('max_supplier_dependency_ratio', 0)*100:.1f}. "
                f"MANDATE: Request their Mizan for contagion risk assessment."
            )
        cust_str = ", ".join(c["label"] + " (₺" + f'{c["balance"]:,.0f}' + ")" for c in custs[:10]) or "No customer data"
        supp_str = ", ".join(s["label"] + " (₺" + f'{s["balance"]:,.0f}' + ")" for s in supps[:10]) or "No supplier data"

        quant_interp = ratios.get("llm_interpretation", "")
        company_name = state.get("company_name", "Company")
        company_tax_id = state.get("tax_id", "1234567890")

        # ── Temporal Context ──
        donem_ctx = ratios.get("donem_context", {})
        period_days = donem_ctx.get("period_days", 360)
        donem_label = donem_ctx.get("label", "Annual (12M)")

        estimated_wc_need = (ccc / period_days) * cogs if period_days > 0 and cogs > 0 else 0

        # ══════════════════════════════════════════════════════════════
        # PRODUCT SIGNALS SECTION — from product_analyst
        # ══════════════════════════════════════════════════════════════
        product_signal_lines = []
        product_interp = ""
        if product_signals:
            product_interp = product_signals.get("llm_interpretation", "")
            for name, data in product_signals.items():
                if name == "llm_interpretation":
                    continue
                if isinstance(data, dict) and (data.get("balance", 0) != 0 or data.get("volume", 0) != 0):
                    product_signal_lines.append(
                        f"- **{name}**: Balance=₺{data['balance']:,.0f}, Volume=₺{data['volume']:,.0f}"
                    )
        product_signal_text = "\n".join(product_signal_lines) if product_signal_lines else "No active product signals detected."

        # ── Build comprehensive prompt with ALL local calculations ──
        prompt = (
            f"Generate a COMPREHENSIVE Corporate Credit Limit & Sales Strategy Report for "
            f"**{company_name}** (Tax ID: {company_tax_id}).\n\n"
            f"IMPORTANT: WRITE THE ENTIRE FINAL REPORT IN PROFESSIONAL ENGLISH BANKING TERMINOLOGY.\n"
            f"This report is for **ING Bank Turkey** Relationship Manager.\n\n"

            f"## ⏱️ TEMPORAL CONTEXT\n"
            f"- **Data Period:** {donem_label} ({period_days} days)\n"
            f"- **Warning:** Flow metrics represent this specific {donem_label} period. "
            f"Annualize in strategic reasoning where appropriate.\n\n"

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
            f"- Net Revenue: ₺{net_revenue:,.0f}\n"
            f"- COGS: ₺{cogs:,.0f}\n"
            f"- Gross Profit: ₺{gross_profit:,.0f}\n"
            f"- Trade Receivables: ₺{trade_recv:,.0f}\n"
            f"- Trade Payables: ₺{trade_pay:,.0f}\n"
            f"- Total Bank Loans: ₺{total_bank_loans:,.0f}\n"
            f"- Estimated Working Capital Need: ₺{estimated_wc_need:,.0f}\n\n"

            f"## 🏦 COMPETITOR BANK WALLET SHARE\n"
            f"### 102-BANKALAR (Deposits):\n{cb_102}\n"
            f"**ING Status (Deposits):** {ing_102}\n\n"
            f"### 300-BANKA KREDİLERİ KV (ST Loans):\n{cb_300}\n"
            f"**ING Status (ST Loans):** {ing_300}\n\n"
            f"### 400-BANKA KREDİLERİ UV (LT Loans):\n{cb_400}\n"
            f"**ING Status (LT Loans):** {ing_400}\n\n"

            f"## 📦 PRODUCT SIGNALS (from Product Analyst)\n"
            f"{product_signal_text}\n\n"
            f"### Product Analyst Intelligence:\n"
            f"{str(product_interp)[:1000]}\n\n"

            f"## 🔍 QUANT ANALYST INTELLIGENCE\n"
            f"{str(quant_interp)[:1500]}\n\n"

            f"## 🌐 COMMERCIAL NETWORK\n"
            f"**Customers ({len(custs)}):** {cust_str}\n"
            f"**Suppliers ({len(supps)}):** {supp_str}\n"
            f"**Total Receivables:** ₺{t_recv:,.0f} | **Total Payables:** ₺{t_pay:,.0f}\n\n"
            f"{concentration_warning}\n"

            f"## INSTRUCTIONS\n"
            f"Using ALL the calculated values above, generate a report with deep analytical REASONING. "
            f"Act as a Senior Credit Allocation Manager at ING Bank Turkey. "
            f"Structure the report into exactly 8 sections in ENGLISH:\n\n"
            f"1. Executive Summary: KPI dashboard table, 3-sentence overall risk/reward assessment.\n"
            f"2. Financial Health & Cash Cycle Analysis: Cite explicit account codes. Analyze CCC ({ccc:.0f} days) and EBITDA.\n"
            f"3. Hidden Risks & Capital Leakage: Analyze Insider Lending (131/331) and Check Risk (103 vs 102).\n"
            f"4. Competitor Intelligence & ING Positioning: Use the ING status flags above. Identify refinancing targets FOR ING.\n"
            f"5. Product Signals & Cross-Sell Opportunities: Use the product signal data to prioritize by ING revenue potential.\n"
            f"6. CREDIT PROPOSAL & STRUCTURING (**CRITICAL**):\n"
            f"   - Propose Working Capital Limit justified by WC Need (₺{estimated_wc_need:,.0f}).\n"
            f"   - Break down into Cash Loans, Non-Cash Loans, and Refinancing.\n"
            f"   - Establish Covenants based on hidden risks.\n"
            f"7. Risk Assessment & Mitigation: Severity ratings with evidence from calculated ratios.\n"
            f"8. Action Plan: Concrete deliverables for the ING Bank Relationship Manager."
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