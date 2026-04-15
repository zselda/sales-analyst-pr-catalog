"""
Agent: Product Analyst — Banking Product Signal Extraction from Mizan

Extracts product-level signals from standardized Mizan data to identify:
- Current banking product usage (loans, POS, DBS, trade finance, etc.)
- Cross-sell/up-sell opportunities for ING Bank RM/sales managers
- Volume and balance metrics for each product category

Returns product_signals dict consumed by the Strategist agent.
"""

import logging
import pandas as pd
from agents.base import BaseAgent
from llm_config import invoke_llm, PRODUCT_ANALYST_SYSTEM_PROMPT

logger = logging.getLogger("swarm.agents.product_analyst")


class ProductAnalystAgent(BaseAgent):
    name = "product_analyst"
    description = "Extract banking product signals and cross-sell opportunities from Mizan data"
    required_inputs = ["standardized_mizan"]
    output_keys = ["product_signals"]

    def execute(self, state: dict) -> dict:
        retry_count = state.get("retry_count", 0)
        period_months = 12
        period_days = 360
        raw_donem = "Unknown"
        donem_label = "Annual (12M) - Default"

        standardized = state.get("standardized_mizan", [])
        if not standardized:
            return {"product_signals": {"error": "No standardized mizan data"}, "retry_count": retry_count + 1}

        df = pd.DataFrame(standardized)

        # ── Period Extraction ──
        donem_col = "donem" if "donem" in df.columns else "period" if "period" in df.columns else None
        if donem_col:
            valid_donems = df[donem_col].dropna().astype(str).unique()
            if len(valid_donems) > 0:
                raw_donem = valid_donems[0].replace(".0", "").strip()
                if len(raw_donem) >= 6 and raw_donem[-2:].isdigit():
                    extracted_month = int(raw_donem[-2:])
                    if 1 <= extracted_month <= 12:
                        period_months = extracted_month
                        period_days = period_months * 30
                        donem_label = f"{period_months} Months ({raw_donem})"
                        logger.info(f"✅ Dynamic Period Extracted: {raw_donem} -> {period_months} Months ({period_days} days)")
                    else:
                        logger.warning(f"⚠️ Invalid month extracted from donem '{raw_donem}'. Falling back to 12M.")
                else:
                    logger.warning(f"⚠️ Unrecognized donem format '{raw_donem}'. Falling back to 12M.")
        else:
            logger.warning("⚠️ 'donem' column not found in data. Falling back to 12M.")

        # Ensure account_code is string for startswith operations
        df["account_code"] = df["account_code"].astype(str)

        # ══════════════════════════════════════════════════════════════
        # SIGNAL EXTRACTION ENGINE
        # ══════════════════════════════════════════════════════════════
        def get_signal_metrics(prefixes: list, keywords: list, account_type: str = "debit") -> dict:
            """
            Search for specified account codes and keywords.
            Prevents double-counting via leaf-node filtering.

            Returns:
                - balance: Net remaining balance at period end
                - volume: Total gross activity volume during the period
            """
            prefix_tuple = tuple(prefixes)
            clean_codes = df["account_code"].astype(str).str.replace(",", ".").str.strip()
            mask_code = clean_codes.str.startswith(prefix_tuple)

            if keywords:
                pattern = "|".join(keywords)
                mask_word = df["account_name"].str.contains(pattern, case=False, na=False)
                filtered_df = df[mask_code & mask_word]
            else:
                filtered_df = df[mask_code]

            if filtered_df.empty:
                return {"balance": 0.0, "volume": 0.0, "account_mapping": {}}

            # Leaf-node filter to prevent double-counting parent/child rows
            codes = filtered_df["account_code"].astype(str).str.replace(",", ".").str.strip().tolist()
            leaf_codes = []
            for code in codes:
                is_parent = any(c.startswith(code) and c != code for c in codes)
                if not is_parent:
                    leaf_codes.append(code)
            final_df = filtered_df[clean_codes.isin(leaf_codes)]
            account_mapping = dict(zip(final_df["account_code"].astype(str), final_df["account_name"].astype(str)))

            # Balance and Volume calculation
            if account_type == "debit":
                balance = float(final_df["debit"].sum() - final_df["credit"].sum())
                volume = float(final_df["debit"].sum())
            else:
                balance = float(final_df["credit"].sum() - final_df["debit"].sum())
                volume = float(final_df["credit"].sum())

            return {"balance": balance, "volume": volume, "account_mapping": account_mapping}

        # ── 1. LOAN SIGNALS ──
        loan_300 = get_signal_metrics(["300"], [], "credit")
        loan_400 = get_signal_metrics(["400"], [], "credit")
        loan_metrics = {
            "balance": loan_300["balance"] + loan_400["balance"],
            "volume": loan_300["volume"] + loan_400["volume"],
            "account_mapping": {**loan_300.get("account_mapping", {}), **loan_400.get("account_mapping", {})}
        }
        financial_expenses = get_signal_metrics(["780"], [], "debit")

        # ── 2. POS / VPOS (Virtual POS) SIGNALS ──
        pos_collection = get_signal_metrics(["108"], [], "debit")
        pos_expenses = get_signal_metrics(["760", "780"], ["POS", "SANAL", "YAZARKASA"], "debit")

        # ── 3. DIRECT DEBIT SYSTEM (DBS) POTENTIAL ──
        dbs_usage = get_signal_metrics(["120", "320", "300"], ["DBS", "DOĞRUDAN BORÇLANDIRMA"], "debit")

        # ── 4. SUPPLIER FINANCE (TFS/SCF) POTENTIAL ──
        tfs_usage = get_signal_metrics(["320", "300"], ["TFS", "TEDARİKÇİ", "TEDARİK FİNANSMANI"], "credit")

        # ── 5. CORPORATE CREDIT CARD SIGNAL ──
        corporate_credit_card = get_signal_metrics(["309", "300", "336"], ["KREDİ KARTI", "K.KARTI", "ŞİRKET KARTI"], "credit")

        # ── 6. VEHICLE FLEET & INSURANCE ──
        vehicle_fleet_assets = get_signal_metrics(["254"], [], "debit")
        insurance_expenses = get_signal_metrics(["730", "760", "770"], ["SİGORTA", "KASKO", "TRAFİK", "POLİÇE"], "debit")

        # ── 7. CHECK PRODUCTS ──
        received_checks = get_signal_metrics(["101"], [], "debit")
        issued_checks = get_signal_metrics(["103"], [], "credit")

        # ── 8. TRADE FINANCE & LETTER OF CREDIT ──
        export_revenue = get_signal_metrics(["601"], [], "credit")
        trade_finance_expenses = get_signal_metrics(["159", "340", "120", "320"], ["İTHALAT", "İHRACAT", "AKREDİTİF", "GÜMRÜK"], "debit")

        # ── 9. FX & SWIFT ──
        fx_profits = get_signal_metrics(["646"], [], "credit")
        fx_losses = get_signal_metrics(["656"], [], "debit")
        fx_volume = {
            "balance": fx_profits["balance"] + fx_losses["balance"],
            "volume": fx_profits["volume"] + fx_losses["volume"],
            "account_mapping": {**fx_profits.get("account_mapping", {}), **fx_losses.get("account_mapping", {})}
        }
        swift_transfer_expenses = get_signal_metrics(["780"], ["SWIFT", "TRANSFER", "HAVALE", "EFT", "YURTDIŞI"], "debit")

        # ── 10. PAYROLL & PERSONNEL ──
        payroll_personnel_volume = get_signal_metrics(["720", "730", "760", "770"], ["PERSONEL", "İŞÇİ", "MAAŞ", "ÜCRET", "SGK"], "debit")

        # ── 11. SECTORAL CALCULATIONS ──
        construction_costs = get_signal_metrics(["170"], [], "debit")
        progress_billings = get_signal_metrics(["350"], [], "credit")
        machinery_equipment = get_signal_metrics(["253"], [], "debit")
        direct_labor_costs = get_signal_metrics(["720"], [], "debit")
        manufacturing_overhead = get_signal_metrics(["730"], [], "debit")
        commercial_goods = get_signal_metrics(["153"], [], "debit")
        marketing_sales_expenses = get_signal_metrics(["760"], [], "debit")

        # ══════════════════════════════════════════════════════════════
        # FINAL PRODUCT SIGNALS DICTIONARY
        # ══════════════════════════════════════════════════════════════
        product_signals = {
            "loan_products": loan_metrics,
            "financial_expenses_780": financial_expenses,
            "pos_collection_volume_108": pos_collection,
            "pos_vpos_expenses": pos_expenses,
            "direct_debit_system_usage": dbs_usage,
            "supplier_finance_usage": tfs_usage,
            "corporate_credit_card": corporate_credit_card,
            "vehicle_fleet_assets_254": vehicle_fleet_assets,
            "insurance_casco_expenses": insurance_expenses,
            "received_checks_101": received_checks,
            "issued_checks_103": issued_checks,
            "export_revenue_601": export_revenue,
            "trade_finance_expenses": trade_finance_expenses,
            "foreign_exchange_volume": fx_volume,
            "swift_and_transfer_expenses": swift_transfer_expenses,
            "payroll_and_personnel_volume": payroll_personnel_volume,
            "construction_costs_170": construction_costs,
            "progress_billings_350": progress_billings,
            "machinery_equipment_253": machinery_equipment,
            "direct_labor_costs_720": direct_labor_costs,
            "manufacturing_overhead_730": manufacturing_overhead,
            "commercial_goods_153": commercial_goods,
            "marketing_sales_expenses_760": marketing_sales_expenses,
        }

        # Log non-zero signals
        active_signals = {k: v for k, v in product_signals.items()
                         if isinstance(v, dict) and (v.get("balance", 0) != 0 or v.get("volume", 0) != 0)}
        logger.info(f"📦 Product signals extracted: {len(active_signals)} active out of {len(product_signals)} total")
        for name, data in active_signals.items():
            logger.info(f"  - {name}: balance=₺{data['balance']:,.0f}, volume=₺{data['volume']:,.0f}")

        # ── LLM INTERPRETATION ──
        llm_text = ""
        try:
            # Build comprehensive product signal summary for LLM
            signal_lines = []
            for name, data in product_signals.items():
                if isinstance(data, dict) and (data.get("balance", 0) != 0 or data.get("volume", 0) != 0):
                    mapping_str = ", ".join(f"'{c}': '{n}'" for c, n in data.get("account_mapping", {}).items())
                    signal_lines.append(
                        f"- **{name}**: Balance=₺{data.get('balance', 0):,.0f}, Volume=₺{data.get('volume', 0):,.0f}, Accounts: {{{mapping_str}}}"
                    )


            signal_summary = "\n".join(signal_lines) if signal_lines else "No significant product signals detected."

            # Annualization factor
            annualization = round(12 / period_months, 2) if period_months else 1.0

            prompt = (
                f"⏱️ DATA PERIOD: {donem_label} ({period_days} days). "
                f"Annualization factor: {annualization}x\n\n"
                f"Analyze the banking product signals for **{state.get('company_name', 'Company')}** "
                f"(Sector: {state.get('sector', 'General')}).\n\n"

                f"## ACTIVE PRODUCT SIGNALS:\n"
                f"{signal_summary}\n\n"

                f"## DETAILED PRODUCT BREAKDOWN:\n"

                f"### 1. LOAN PRODUCTS (300+400):\n"
                f"- Short-Term Loans (300): Balance=₺{loan_300['balance']:,.0f}, Volume=₺{loan_300['volume']:,.0f}\n"
                f"- Long-Term Loans (400): Balance=₺{loan_400['balance']:,.0f}, Volume=₺{loan_400['volume']:,.0f}\n"
                f"- Total Financial Expenses (780): ₺{financial_expenses['balance']:,.0f}\n\n"

                f"### 2. POS / VIRTUAL POS:\n"
                f"- POS Collection (108): Volume=₺{pos_collection['volume']:,.0f}\n"
                f"- POS/VPOS Expenses: ₺{pos_expenses['balance']:,.0f}\n\n"

                f"### 3. DBS (Direct Debit System):\n"
                f"- Usage Signal: Balance=₺{dbs_usage['balance']:,.0f}, Volume=₺{dbs_usage['volume']:,.0f}\n\n"

                f"### 4. SUPPLIER FINANCE (TFS/SCF):\n"
                f"- Usage Signal: Balance=₺{tfs_usage['balance']:,.0f}, Volume=₺{tfs_usage['volume']:,.0f}\n\n"

                f"### 5. CORPORATE CREDIT CARD:\n"
                f"- Usage Signal: Balance=₺{corporate_credit_card['balance']:,.0f}, Volume=₺{corporate_credit_card['volume']:,.0f}\n\n"

                f"### 6. VEHICLE FLEET & INSURANCE:\n"
                f"- Fleet Assets (254): ₺{vehicle_fleet_assets['balance']:,.0f}\n"
                f"- Insurance Expenses: ₺{insurance_expenses['balance']:,.0f}\n\n"

                f"### 7. CHECK PRODUCTS:\n"
                f"- Received Checks (101): Balance=₺{received_checks['balance']:,.0f}, Volume=₺{received_checks['volume']:,.0f}\n"
                f"- Issued Checks (103): Balance=₺{issued_checks['balance']:,.0f}, Volume=₺{issued_checks['volume']:,.0f}\n\n"

                f"### 8. TRADE FINANCE & LETTER OF CREDIT:\n"
                f"- Export Revenue (601): ₺{export_revenue['volume']:,.0f}\n"
                f"- Trade Finance Related Expenses: ₺{trade_finance_expenses['balance']:,.0f}\n\n"

                f"### 9. FX & INTERNATIONAL TRANSFERS:\n"
                f"- FX Net Impact: ₺{fx_volume['balance']:,.0f}, Total FX Activity: ₺{fx_volume['volume']:,.0f}\n"
                f"- SWIFT/Transfer Expenses: ₺{swift_transfer_expenses['balance']:,.0f}\n\n"

                f"### 10. PAYROLL & PERSONNEL:\n"
                f"- Total Payroll Volume: ₺{payroll_personnel_volume['volume']:,.0f}\n\n"

                f"### 11. SECTORAL INDICATORS:\n"
                f"- Construction Costs (170): ₺{construction_costs['balance']:,.0f}\n"
                f"- Progress Billings (350): ₺{progress_billings['balance']:,.0f}\n"
                f"- Machinery & Equipment (253): ₺{machinery_equipment['balance']:,.0f}\n"
                f"- Commercial Goods (153): ₺{commercial_goods['balance']:,.0f}\n"
                f"- Manufacturing Overhead (730): ₺{manufacturing_overhead['balance']:,.0f}\n\n"

                f"## INSTRUCTIONS FOR ANALYSIS:\n"
                f"For each product category with non-zero signals, you must:\n"
                f"1. State whether the company is currently using this product type\n"
                f"2. Quantify the opportunity size (annualized if period < 12M)\n"
                f"3. Recommend specific ING Bank products that match the signal\n"
                f"4. Estimate potential revenue impact for ING Bank\n"
                f"5. **CRITICAL**: Explicitly reference the exact Account Codes and Names used in the calculation (e.g. '101.01 - Alınan Çekler TL') so the Relationship Manager can easily track them in the Mizan.\n\n"
                f"Focus on HIGH-VOLUME signals first. Sort by revenue potential.\n"
                f"Output structured Markdown with clear product → opportunity mapping."
            )

            llm_text = invoke_llm(PRODUCT_ANALYST_SYSTEM_PROMPT, prompt, temperature=0.2, max_tokens=1500)
            self.metrics.record_llm_call(tokens=len(llm_text.split()))
            logger.info(f"✅ Product analyst LLM interpretation: {len(llm_text)} chars")
        except Exception as e:
            logger.warning(f"LLM skipped: {e}")
            llm_text = "LLM interpretation unavailable."

        product_signals["llm_interpretation"] = llm_text
        return {"product_signals": product_signals, "retry_count": retry_count + 1}


# Module-level callable for LangGraph
product_analyst_agent = ProductAnalystAgent()