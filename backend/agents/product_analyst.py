"""
Agent: Autonomous Product Analyst — Banking Product Signal Extraction
Uses Tool Calling to autonomously query the standardized Mizan data.
Finds cross-sell opportunities for ING Bank (loans, POS, DBS, trade finance, etc.)
by actively searching account codes and keywords.
 
Enhanced with:
- Selective few-shot prompting via agents/few_shot_library.py — only scenarios
  with detected product signals are injected (token-efficient)
- Current-usage awareness: products flagged as used in the local bank DB
  (db_product_flags) are excluded from recommendations
- All 5 columns: credit, debit, balance_debit, balance_credit, volume
- build_strategist_payload() data aggregator for strategist agent
- Sector-aware product suggestions
"""
import json
import logging
import pandas as pd
import re
from agents.base import BaseAgent
import warnings
warnings.filterwarnings("ignore")
from llm_config import invoke_llm, PRODUCT_ANALYST_SYSTEM_PROMPT
from agents.few_shot_library import (
    classify_product_opportunities,
    build_few_shot_injection,
    build_recommendation_catalog,
)
from agents.product_catalog import build_catalog_injection
logger = logging.getLogger("swarm.agents.product_analyst")
 
# Few-shot scenarios and the product classification/selection logic now
# live in agents/few_shot_library.py. Only scenarios whose product signals
# are present in the Mizan data are injected into the prompts.


def build_strategist_payload(quant_state: dict, product_state: dict) -> str:
    """
    Combines quant and product analyst outputs into a compressed JSON payload
    optimized for the strategist LLM. Includes ALL calculations organized by
    financial statement section for comprehensive strategy formulation.
    """
    ratios = quant_state.get("financial_ratios", {})
    signals = product_state.get("product_signals", {})
 
    donem_ctx = ratios.get("donem_context", {})
    period_days = donem_ctx.get("period_days", 360)
 
    # Helper to safely extract raw values from nested ratio dicts
    def rv(ratio_name: str, key: str):
        return ratios.get(ratio_name, {}).get("raw_values", {}).get(key, 0)
 
    cogs = rv("gross_margin", "cogs_62x")
    ccc_val = ratios.get("cash_conversion_cycle", {}).get("value", 0)
    estimated_wc = (ccc_val / period_days) * cogs if period_days > 0 and cogs > 0 else 0
 
    payload = {
        "context": {
            "period": donem_ctx.get("label", "Unknown"),
            "period_days": period_days,
            "period_months": donem_ctx.get("period_months", 12)
        },
        # ── 1. INCOME STATEMENT ──
        "income_statement": {
            "gross_revenue": rv("gross_margin", "gross_revenue") if rv("gross_margin", "gross_revenue") else rv("gross_margin", "net_revenue"),
            "sales_deductions": rv("gross_margin", "sales_deductions"),
            "net_revenue": rv("gross_margin", "net_revenue"),
            "cogs_62x": cogs,
            "gross_profit": rv("gross_margin", "gross_profit"),
            "gross_margin_pct": ratios.get("gross_margin", {}).get("value", 0),
            "op_expenses_63x": rv("operating_margin", "op_expenses_63x"),
            "operating_profit": rv("operating_margin", "operating_profit"),
            "operating_margin_pct": ratios.get("operating_margin", {}).get("value", 0),
            "ebitda_proxy": rv("operating_margin", "operating_profit") + abs(rv("quick_ratio", "inventory_total")) * 0  # placeholder; EBITDA stored in gross_profit proxy
        },
        # ── 2. BALANCE SHEET ──
        "balance_sheet": {
            "current_assets": rv("current_ratio", "current_assets") or rv("insider_lending_ratio", "current_assets"),
            "non_current_assets": rv("insider_lending_ratio", "non_current_assets"),
            "total_assets": rv("insider_lending_ratio", "total_assets"),
            "inventory": rv("quick_ratio", "inventory_total"),
            "liquid_assets": rv("quick_ratio", "liquid_assets"),
            "short_term_liabilities": rv("current_ratio", "short_term_liabilities") or rv("debt_to_equity", "short_term_liab"),
            "long_term_liabilities": rv("debt_to_equity", "long_term_liab"),
            "total_liabilities": rv("debt_to_equity", "total_liabilities"),
            "total_equity": rv("debt_to_equity", "total_equity"),
        },
        # ── 3. LEVERAGE ──
        "leverage": {
            "total_bank_loans": rv("bank_debt_ratio", "total_bank_loans"),
            "bank_loans_st_300": rv("bank_debt_ratio", "bank_loans_st_300"),
            "bank_loans_lt_400": rv("bank_debt_ratio", "bank_loans_lt_400"),
            "credit_cards_309": rv("bank_debt_ratio", "credit_cards_309"),
            "fin_expenses_780": rv("financial_expense_ratio", "finansman_giderleri_780"),
            "debt_to_equity": ratios.get("debt_to_equity", {}).get("value", 0),
            "bank_debt_ratio_pct": ratios.get("bank_debt_ratio", {}).get("value", 0),
            "financial_expense_ratio_pct": ratios.get("financial_expense_ratio", {}).get("value", 0),
        },
        # ── 4. WORKING CAPITAL ──
        "working_capital": {
            "trade_receivables": rv("collection_period", "trade_receivables"),
            "trade_payables": rv("payment_period", "trade_payables"),
            "collection_period_days": ratios.get("collection_period", {}).get("value", 0),
            "payment_period_days": ratios.get("payment_period", {}).get("value", 0),
            "inventory_period_days": ratios.get("inventory_period", {}).get("value", 0),
            "cash_conversion_cycle_days": ccc_val,
            "estimated_wc_need": estimated_wc,
            "current_ratio": ratios.get("current_ratio", {}).get("value", 0),
            "quick_ratio": ratios.get("quick_ratio", {}).get("value", 0),
        },
        # ── 5. CASH FLOW ──
        "cash_flow": ratios.get("cash_flow_summary", {}),
        # ── 6. RISKS ──
        "risks": {
            "insider_lending_131": rv("insider_lending_ratio", "insider_lending_131"),
            "insider_borrowing_331": rv("insider_lending_ratio", "insider_borrowing_331"),
            "insider_lending_ratio_pct": ratios.get("insider_lending_ratio", {}).get("value", 0),
            "given_checks_103": rv("check_risk_ratio", "given_checks_103"),
            "banks_102_total": rv("check_risk_ratio", "banks_102_total"),
            "check_risk_ratio": ratios.get("check_risk_ratio", {}).get("value", 0),
        },
        # ── 7. COMPETITOR BANKS ──
        "competitor_banks": {
            "deposits_102": ratios.get("competitor_banks", {}).get("102", []),
            "st_loans_300": ratios.get("competitor_banks", {}).get("300", []),
            "lt_loans_400": ratios.get("competitor_banks", {}).get("400", [])
        },
        # ── 8. PRODUCT SIGNALS ──
        "product_signals": {
            k: {
                "balance": v.get("balance"), "volume": v.get("volume"),
                "credit": v.get("credit", 0), "debit": v.get("debit", 0),
                "balance_debit": v.get("balance_debit", 0), "balance_credit": v.get("balance_credit", 0),
                "account_mapping": v.get("account_mapping", {})
            }
            for k, v in signals.items()
            if isinstance(v, dict) and (v.get("balance", 0) != 0 or v.get("volume", 0) != 0)
        }
       # # ── 9. ANALYST INSIGHTS ──
       # "analyst_insights": {
       #     "quant_summary": str(ratios.get("llm_interpretation", "")),
       #     "product_summary": str(signals.get("llm_interpretation", ""))
       # }
    }
 
    return json.dumps(payload, indent=2, ensure_ascii=False)
 
class ProductAnalystAgent(BaseAgent):
   name = "product_analyst"
   description = "Autonomously query Mizan to extract banking product signals using tools"
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
       logger.info(f"✅ Dynamic Period: {raw_donem} -> {period_months} Months ({period_days} days)")
       # Hesap kodlarını string'e çevir
       df["account_code"] = df["account_code"].astype(str).str.replace(",", ".").str.strip()
 
       # ══════════════════════════════════════════════════════════════
       # SIGNAL EXTRACTION ENGINE
       # ══════════════════════════════════════════════════════════════
       def get_signal_metrics(prefixes: list, keywords: list, account_type: str = "debit") -> dict:
            """
            Search for specified account codes and keywords.
            Prevents double-counting via leaf-node filtering.
 
            Returns dict with: credit, debit, balance_debit, balance_credit,
            balance (net), volume (gross flow), account_mapping {code: name}
            """
            prefix_tuple = tuple(prefixes)
            clean_codes = df["account_code"].astype(str).str.replace(",", ".").str.strip()
            mask_code = clean_codes.str.startswith(prefix_tuple)
 
            if keywords:
                escaped_keywords = [re.escape(k) for k in keywords]
                pattern = r'(?<!\w)(' + '|'.join(escaped_keywords) + r')(?!\w)'
                mask_word = df["account_name"].str.contains(pattern, case=False, na=False)
                filtered_df = df[mask_code & mask_word]
            else:
                filtered_df = df[mask_code]
 
            if filtered_df.empty:
                return {"credit": 0.0, "debit": 0.0, "balance_debit": 0.0,
                        "balance_credit": 0.0, "balance": 0.0, "volume": 0.0,
                        "account_mapping": {}}
 
            # Leaf-node filter to prevent double-counting parent/child rows
            codes = filtered_df["account_code"].astype(str).str.replace(",", ".").str.strip().tolist()
            leaf_codes = []
            for code in codes:
                is_parent = any(c.startswith(code) and c != code for c in codes)
                if not is_parent:
                    leaf_codes.append(code)
            final_df = filtered_df[clean_codes.isin(leaf_codes)]
            # account_mapping = dict(zip(final_df["account_code"].astype(str), final_df["account_name"].astype(str)))
 
            if account_type == "debit":
                volume_col = "debit"
                balance_col = "balance_debit"
            else:
                volume_col = "credit"
                balance_col = "balance_credit"
            # volume'a göre sırala
            final_df = final_df.sort_values(by=volume_col, ascending=False)
            # toplam volume
            total_volume = final_df[volume_col].sum()
            if total_volume > 0 and len(final_df)>10:
                final_df["cum_ratio"] = final_df[volume_col].cumsum() / total_volume
                cutoff_idx = final_df["cum_ratio"].searchsorted(0.8)
                final_df = final_df.iloc[:cutoff_idx+1]
                # en az 1 satır garanti
                if final_df.empty:
                    final_df = filtered_df.sort_values(by=volume_col, ascending=False).head(1)
                # max 10 satır
                final_df = final_df.head(10)
            # ---------------------------
            # mapping
            account_mapping = dict(zip(
                final_df["account_code"].astype(str) + " " + final_df["account_name"].astype(str),
                "volume: " + final_df[volume_col].astype(str)
            ))
            # All 5 columns
            total_credit = float(final_df["credit"].sum())
            total_debit = float(final_df["debit"].sum())
            total_bal_debit = float(final_df["balance_debit"].sum()) if "balance_debit" in final_df.columns else 0.0
            total_bal_credit = float(final_df["balance_credit"].sum()) if "balance_credit" in final_df.columns else 0.0
 
            if account_type == "debit":
                balance = total_debit - total_credit
                volume = total_debit
            else:
                balance = total_credit - total_debit
                volume = total_credit
 
            return {
                "credit": total_credit, "debit": total_debit,
                "balance_debit": total_bal_debit, "balance_credit": total_bal_credit,
                "balance": balance, "volume": volume,
                "account_mapping": account_mapping
            }
 
       def get_account_mapping(data):
            signal_lines = []
            mapping_str = ", ".join(f"'{c} - {n}'" for c, n in data.get("account_mapping", {}).items())
            signal_lines.append(
                    f": Balance=₺{data.get('balance', 0):,.0f}, Volume=₺{data.get('volume', 0):,.0f}, Accounts: {{{mapping_str}}}"
                )
            return mapping_str
        # ── 1. LOAN SIGNALS ──
       loan_300 = get_signal_metrics(["300"], [], "credit")
       loan_400 = get_signal_metrics(["400"], [], "credit")
       loan_metrics = {
             "credit": loan_300["credit"] + loan_400["credit"],
             "debit": loan_300["debit"] + loan_400["debit"],
             "balance_debit": loan_300["balance_debit"] + loan_400["balance_debit"],
             "balance_credit": loan_300["balance_credit"] + loan_400["balance_credit"],
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
             "credit": fx_profits["credit"] + fx_losses["credit"],
             "debit": fx_profits["debit"] + fx_losses["debit"],
             "balance_debit": fx_profits["balance_debit"] + fx_losses["balance_debit"],
             "balance_credit": fx_profits["balance_credit"] + fx_losses["balance_credit"],
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
       marketing_sales_expenses = get_signal_metrics(["760"], ['PAZARLAMA'], "debit")
        # ── 12. NEW PRODUCT SIGNALS (factoring, leasing, guarantees, cash mgmt, e-commerce) ──
       notes_receivable = get_signal_metrics(["121"], [], "debit")
       leasing_payables = get_signal_metrics(["301", "401"], [], "credit")
       guarantee_commissions = get_signal_metrics(["760", "770", "780"], ["TEMİNAT"], "debit")
       bank_transaction_volume = get_signal_metrics(["102"], [], "debit")
       ecommerce_revenue = get_signal_metrics(["600", "649"], ["E-TİCARET", "ETİCARET", "E TİCARET", "ONLINE", "PAZARYERİ", "SANAL"], "credit")
        # ══════════════════════════════════════════════════════════════
        # FINAL PRODUCT SIGNALS DICTIONARY
        # ══════════════════════════════════════════════════════════════
       product_signals = {
           "Bank Loans (300 + 400)": loan_metrics,
           "Total Financial Expenses (780)": financial_expenses,
           "POS Collection (108)": pos_collection,
           "POS / Virtual POS": pos_expenses,
           "DBS (Direct Debit System)": dbs_usage,
           "Supplier Finance (TFS/SCF)": tfs_usage,
           "Corporate Credit Card": corporate_credit_card,
           "Fleet Assets (254)": vehicle_fleet_assets,
           "Insurance Expenses (730/760/770)": insurance_expenses,
           "Received Checks (101)": received_checks,
           "Issued Checks (103)": issued_checks,
           "Export Revenue (601)": export_revenue,
           "FX Net Impact (646/656)": fx_volume,
           "SWIFT/Transfer Expenses": swift_transfer_expenses,
           "Payroll & Personnel": payroll_personnel_volume,
           "Construction Costs (170)": construction_costs,
           "Progress Billings (350)": progress_billings,
           "Machinery & Equipment (253)": machinery_equipment,
           "Direct Labor (720)": direct_labor_costs,
           "Manufacturing Overhead (730)": manufacturing_overhead,
           "Commercial Goods (153)": commercial_goods,
           "Marketing Sales": marketing_sales_expenses,
           "Trade Finance Signals (159/340)": trade_finance_expenses,
           "Notes Receivable (121)": notes_receivable,
           "Leasing Payables (301/401)": leasing_payables,
           "Guarantee Letter Commissions": guarantee_commissions,
           "Bank Transaction Volume (102)": bank_transaction_volume,
           "E-Commerce Revenue": ecommerce_revenue,
       }
       zero_volumes = [key for key, inner_dict in product_signals.items() if inner_dict.get('volume')==0]
       # Log non-zero signals
       active_signals = {k: v for k, v in product_signals.items()
                        if isinstance(v, dict) and (v.get("balance", 0) != 0 or v.get("volume", 0) != 0)}
       logger.info(f"📦 Product signals extracted: {len(active_signals)} active out of {len(product_signals)} total")
       for name, data in active_signals.items():
           logger.info(f"  - {name}: balance=₺{data['balance']:,.0f}, volume=₺{data['volume']:,.0f}")
       # ── LLM INTERPRETATION ──
       llm_text = ""
       #try:
       signal_lines = []
       for name, data in product_signals.items():
            if isinstance(data, dict) and (data.get("balance_credit", 0) != 0 or data.get("balance_debit", 0) != 0 or 
                                           data.get("credit", 0) != 0 or data.get("debit", 0) != 0):
                mapping_str = ", ".join(f"'{c} - {n}'" for c, n in data.get("account_mapping", {}).items())
                signal_lines.append(
                    f"- **{name}**" #: Balance=₺{data.get('balance', 0):,.0f}, Volume=₺{data.get('volume', 0):,.0f}, Accounts: {{{mapping_str}}}"
                )
 
       signal_summary = "\n".join(signal_lines) if signal_lines else "No significant product signals detected."
       annualization = round(12 / period_months, 2) if period_months else 1.0
       sector = state.get('sector', 'General')

       # ── DETAILED PRODUCT BREAKDOWN (token-optimized) ──
       # Active lines (non-zero signal) keep the full 5-column detail.
       # Zero-signal lines collapse to a single "no signal" line so they
       # still feed SECTION 2 (cross-sell gaps) without wasting ~1.8K tokens
       # of all-zero columns.
       def _has_signal(m: dict) -> bool:
           return any(abs(float(m.get(k, 0) or 0)) > 0 for k in
                      ("balance", "volume", "credit", "debit",
                       "balance_credit", "balance_debit"))

       def _fmt_breakdown_line(label: str, m: dict) -> str:
           if _has_signal(m):
               return (f"- {label}: Credit Balance=₺{m['balance_credit']:,.0f}, "
                       f"Debit Balance=₺{m['balance_debit']:,.0f}, "
                       f"Credit Volume=₺{m['credit']:,.0f}, "
                       f"Debit Volume=₺{m['debit']:,.0f}, "
                       f"Account Mapping={get_account_mapping(m)}")
           return f"- {label}: no signal (potential cross-sell gap)"

       breakdown_spec = [
           ("1. LOAN PRODUCTS (300 + 400 - Bank Loans)", [
               ("Short-Term Loans (300 - Bank Loans)", loan_300),
               ("Long-Term Loans (400 - Bank Loans)", loan_400),
               ("Total Financial Expenses (780 - Financial Expenses)", financial_expenses),
           ]),
           ("2. POS / VIRTUAL POS", [
               ("POS Collection (108 - Other Liquid Assets)", pos_collection),
               ("POS/VPOS Expenses (760/780 - Expenses)", pos_expenses),
           ]),
           ("3. DBS (Direct Debit System)", [
               ("Usage Signal (120/320/300)", dbs_usage),
           ]),
           ("4. SUPPLIER FINANCE (TFS/SCF)", [
               ("Usage Signal (320/300)", tfs_usage),
           ]),
           ("5. CORPORATE CREDIT CARD", [
               ("Usage Signal (309/336)", corporate_credit_card),
           ]),
           ("6. VEHICLE FLEET & INSURANCE", [
               ("Fleet Assets (254 - Vehicles)", vehicle_fleet_assets),
               ("Insurance Expenses (730/760/770)", insurance_expenses),
           ]),
           ("7. CHECK PRODUCTS", [
               ("Received Checks (101 - Received Checks)", received_checks),
               ("Issued Checks (103 - Given Checks)", issued_checks),
           ]),
           ("8. TRADE FINANCE & LETTER OF CREDIT", [
               ("Export Revenue (601 - Export Sales)", export_revenue),
               ("Trade Finance Related Expenses (159/340)", trade_finance_expenses),
           ]),
           ("9. FX & INTERNATIONAL TRANSFERS", [
               ("FX Net Impact (646/656)", fx_volume),
               ("SWIFT/Transfer Expenses", swift_transfer_expenses),
           ]),
           ("10. PAYROLL & PERSONNEL", [
               ("Total Payroll Volume (720/730/760/770 - Labor/Personnel)", payroll_personnel_volume),
           ]),
           ("11. SECTORAL INDICATORS", [
               ("Construction Costs (170 - Construction Costs)", construction_costs),
               ("Progress Billings (350 - Progress Billings)", progress_billings),
               ("Machinery & Equipment (253 - Machinery & Equipment)", machinery_equipment),
               ("Commercial Goods (153 - Commercial Goods)", commercial_goods),
               ("Manufacturing Overhead (730 - Manufacturing Overhead)", manufacturing_overhead),
           ]),
           ("12. FACTORING / LEASING / GUARANTEES / CASH MGMT / E-COMMERCE", [
               ("Notes Receivable (121 - Alacak Senetleri)", notes_receivable),
               ("Leasing Payables (301/401 - Finansal Kiralama Borçları)", leasing_payables),
               ("Guarantee Letter Commissions (760/770/780 - TEMİNAT)", guarantee_commissions),
               ("Bank Transaction Volume (102 - Bankalar)", bank_transaction_volume),
               ("E-Commerce Revenue (600/649 - E-TİCARET/ONLINE/PAZARYERİ)", ecommerce_revenue),
           ]),
       ]
       breakdown_lines = []
       for header, items in breakdown_spec:
           breakdown_lines.append(f"### {header}:")
           breakdown_lines.extend(_fmt_breakdown_line(lbl, m) for lbl, m in items)
           breakdown_lines.append("")
       detailed_breakdown = "\n".join(breakdown_lines)
       prompt = (
           f"⏱️ DATA PERIOD: {donem_label} ({period_days} days). "
           f"Annualization factor: {annualization}x\n\n"
           f"Analyze the banking product signals for **{state.get('company_name', 'Company')}** "
           f"(Sector: **{sector}**).\n\n"
           f"## SECTOR-AWARE PRODUCT PRIORITIZATION\n"
           f"The company operates in the **{sector}** sector. Prioritize product suggestions "
           f"relevant to this sector.\n\n"
           f"## ACTIVE PRODUCT SIGNALS:\n"
           f"{signal_summary}\n\n"
           "## DETAILED PRODUCT BREAKDOWN:\n"
           "(Lines marked 'no signal' have zero volume/balance — treat them as "
           "SECTION 2 cross-sell gaps.)\n"
           f"{detailed_breakdown}\n"

           f"## OUTPUT FORMAT INSTRUCTIONS:\n"
           f"Structure your output into EXACTLY these sections:\n\n"
           f"### SECTION 1: HIGH-PRIORITY SIGNALS (sorted by revenue potential)\n"
           f"For each product category with non-zero signals, use this EXACT format:\n\n"
           f"#### [Product Name] — Revenue Potential: [HIGH/MEDIUM/LOW]\n"
           f"- **IF**: [State the data condition that triggered the signal]\n"
           f"- **THINKING**: [Sector-aware interpretation — why this matters for {sector}]\n"
           f"- **DATA**: Debit=₺X, Credit=₺X, BalDebit=₺X, BalCredit=₺X, Net=₺X, Volume=₺X\n"
           f"- **ACCOUNTS**: [List exact sub-account codes from account mapping with **volume information** — DO NOT invent]\n"
           f"- **PROPOSAL**: [Specific ING Bank Turkey product] → Estimated annual revenue: ₺X - Use backed data(i.e. Credit card spending volume for Corporate Credit Card) for estimation.\n\n"
           f"### SECTION 2: CROSS-SELL GAPS\n"
           f"a. List products with ZERO signals where ING can create new opportunities.\n"
           f"b. List products with volume where ING can sell new products together with current products.\n"
           f"Format: - [Product] → ING Opportunity: [description]\n\n"
           f"### SECTION 3: REVENUE SUMMARY TABLE\n"
           f"You MUST explicitly include EVERY single product opportunity mentioned in Section 1 AND Section 2 in this table. Do not omit, group, or summarize any items. If an item is listed in Section 1 or Section 2, it MUST have its own row here.\n\n"
          #f"| Product Area/Need | Signal Type (Active/Cross-Sell) | Current Volume / Status | Proposed ING Product | Est. Annual Revenue | Priority |\n"
          #f"|-------------------|---------------------------------|-------------------------|----------------------|---------------------|----------|\n"
          #f"[Insert a row for EVERY active signal from Section 1]\n"
          #f"[Insert a row for EVERY cross-sell gap from Section 2]\n\n"
           #f"</output_instructions>\n\n"
           f"| Product | Current Volume | ING Product | Est. Annual Revenue | Priority |\n"
           f"|---------|---------------|-------------|--------------------|---------|\n"
           f"[Fill with data from Section 1 and Section 2, include ALL.]\n\n"
 
           #f"<critical_guardrails>\n"
           f"CRITICAL RULES:\n"
           f"1. PERSONA ENFORCEMENT: Tailor all insights through the lens of a corporate banker specializing in the {sector} sector.\n"
           f"2. ZERO HALLLUCINATIONS: Only reference sub-account codes explicitly listed in the account mapping dictionaries. If no code is present, omit it.\n"
           f"3. REVENUE SORTING: Order items in Section 1 strictly by the calculated annualized revenue potential from highest to lowest.\n"
           f"4. TOTAL ALIGNMENT CONSTRAINT: Before printing Section 3, verify that the total number of table rows equals exactly (Total Section 1 entries + Total Section 2 entries). Every identified product must be accounted for.\n"
           f"5. CURRENT USAGE EXCLUSION: Products listed under 'CURRENTLY USED PRODUCTS' (latest bank core data) MUST NOT appear as recommendations in ANY section or table — reference them only as existing relationship context.\n"
           #f"</critical_guardrails>\n"
           #f"CRITICAL RULES:\n"
           #f"- Only reference accounts explicitly provided. DO NOT invent sub-account codes.\n"
           #f"- Sort HIGH-PRIORITY signals by estimated revenue (highest first).\n"
           #f"- Use annualization factor {annualization}x for sub-annual periods.\n\n"
           #f"- TABLE COMPLETENESS STRICT RULE: The Section 3 table must be a 1:1 match with Sections 1 and 2. Count your proposals before writing the table to ensure absolutely no product is left out."
        )

       # ── SELECTIVE FEW-SHOT INJECTION (token-efficient) ──
       # Only scenarios whose product signals exist in this Mizan are injected.
       # Products already used per the local bank DB (db_product_flags == 1)
       # are excluded from recommendations and listed as current usage.
       db_product_flags = state.get("db_product_flags") or {}
       db_metrics = state.get("db_financial_metrics") or {}
       classification = classify_product_opportunities(
           product_signals, sector=sector, product_flags=db_product_flags
       )
       injection = build_few_shot_injection(classification, sector=sector)
       # Real ING product taxonomy (ANA ÜRÜN → ALT ÜRÜN) so suggestions map
       # to actual products: ANA ÜRÜN first, drill to ALT ÜRÜN with evidence,
       # YP (foreign-currency) only with FX/foreign-trade signals, and
       # respect AÇIKLAMA conditions (EXIMBANK/Reeskont/PARA TRANSFERLERİ).
       catalog_injection = build_catalog_injection(product_signals)
       if db_metrics:
           db_meta = state.get("db_meta") or {}
           metric_line = ", ".join(f"{k}={v:,.2f}" for k, v in db_metrics.items())
           prompt += (
               f"\n\n## LOCAL DB REFERENCE METRICS (bank core system, "
               f"period {db_meta.get('financial_period', '?')} — LATEST DATA):\n"
               f"{metric_line}\n"
               f"Use these as the authoritative latest view when sizing opportunities."
           )
       prompt += injection["user_addition"] + catalog_injection["user_addition"]
       system_prompt = (
           PRODUCT_ANALYST_SYSTEM_PROMPT
           + injection["system_addition"]
           + catalog_injection["system_addition"]
       )

       print("product_analyst_prompt: ", prompt)
       llm_text = invoke_llm(system_prompt, prompt, temperature=0.2, max_tokens=3000)
       self.metrics.record_llm_call(tokens=len(llm_text.split()))
       logger.info(f"✅ Product analyst LLM interpretation: {len(llm_text)} chars")
       #except Exception as e:
       #    logger.warning(f"LLM skipped: {e}")
       #    llm_text = "LLM interpretation unavailable."
       summary_rows = []
       detail_rows = []
       for product_name, metrics in product_signals.items():
           summary_rows.append({
               "product_name": product_name,
               "balance_debit": int(metrics.get("balance_debit", 0)),
               "balance_credit": int(metrics.get("balance_credit", 0)),
               "debit": int(metrics.get("debit", 0)),
               "credit": int(metrics.get("credit", 0)),
               "matched_account_count": len(metrics.get("account_mapping", {}))
           })
           for account_info, volume_info in metrics.get("account_mapping", {}).items():
               detail_rows.append({
                   "product_name": product_name,
                   "account_info": account_info,
                   "volume_info": float(volume_info.replace("volume: ", "").strip())
               })
       summary_df = pd.DataFrame(summary_rows)
       reference_df = pd.DataFrame(detail_rows)
       def format_tl(x):
           return f"{x:,.0f}"
       for col in ["balance_debit", "balance_credit", "debit", "credit"]:
           summary_df[col] = summary_df[col].apply(format_tl)
       for col in ["volume_info"]:
           reference_df[col] = reference_df[col].apply(format_tl)
       product_signals["llm_interpretation"] = llm_text
       product_signals["summary_df"]=summary_df
       product_signals["reference_df"]=reference_df
       # Classification result (selected/excluded products) for downstream agents
       product_signals["recommendation_classification"] = {
           "selected_keys": classification["selected_keys"],
           "excluded_existing": classification["excluded_existing"],
           "inactive_keys": classification["inactive_keys"],
       }
       # DETERMINISTIC recommendation catalog: fixes the strategist matrix
       # membership (all active signals + sector cross-sell gaps), so the
       # row count no longer varies run over run.
       product_signals["recommendation_catalog"] = build_recommendation_catalog(
           product_signals, sector=sector, product_flags=db_product_flags
       )
       return {"product_signals": product_signals, "retry_count": retry_count + 1}
# Module-level callable for LangGraph
product_analyst_agent = ProductAnalystAgent()
