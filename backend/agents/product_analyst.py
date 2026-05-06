"""
Agent: Autonomous Product Analyst — Banking Product Signal Extraction
Uses Tool Calling to autonomously query the standardized Mizan data.
Finds cross-sell opportunities for ING Bank (loans, POS, DBS, trade finance, etc.)
by actively searching account codes and keywords.

Enhanced with:
- Few-shot prompting guide (MIZAN_FEW_SHOT_GUIDE) for hallucination-free analysis
- All 5 columns: credit, debit, balance_debit, balance_credit, volume
- build_strategist_payload() data aggregator for strategist agent
- Sector-aware product suggestions
"""
import json
import logging
import pandas as pd
from agents.base import BaseAgent
from llm_config import invoke_llm, PRODUCT_ANALYST_SYSTEM_PROMPT
logger = logging.getLogger("swarm.agents.product_analyst")

# ══════════════════════════════════════════════════════════════
# FEW-SHOT PROMPTING GUIDE — ING TURKEY PRODUCT ANALYSIS
# ══════════════════════════════════════════════════════════════
MIZAN_FEW_SHOT_GUIDE = {
    "core_mizan_rules": [
        "DO NOT invent or search for account codes. Only use the exact balances, volumes, and sub-account names explicitly provided to you in the prompt by the Python system.",
        "Follow the IF -> THINKING -> ACTION -> PROPOSAL reasoning structure for your analysis.",
        "Map all proposals to specific ING Turkey products (e.g., 'ING DBS', 'ING e-Turuncu Kur', 'ING Bonus Business').",
        "Use the company's SECTOR information to prioritize relevant products. A manufacturing firm needs different products than a trading or services firm.",
        "Distinguish between BALANCE (stock at period-end) and VOLUME (flow during period). High volume with low balance = healthy turnover."
    ],
    "sector_product_priority": {
        "Manufacturing / Üretim": ["Leasing (253)", "Working Capital Loans", "Supply Chain Finance", "Fleet Insurance", "Payroll", "FX (if exporter)"],
        "Trading / Ticaret": ["POS/VPOS", "DBS", "Corporate Credit Card", "Check Products", "Commercial Loans", "FX/SWIFT"],
        "Construction / İnşaat": ["Letter of Guarantee (Teminat Mektubu)", "Progress Payment Finance (170/350)", "Leasing (253)", "Surety Bonds"],
        "Services / Hizmet": ["POS/VPOS", "Payroll", "Corporate Credit Card", "Cash Management", "Digital Banking"],
        "Export / İhracat": ["Trade Finance", "Letter of Credit", "FX/e-Turuncu Kur", "SWIFT", "Export Factoring"],
        "Import / İthalat": ["Trade Finance", "Letter of Credit", "FX/e-Turuncu Kur", "Import Loans", "Customs Guarantee"],
        "Retail / Perakende": ["POS/VPOS", "DBS", "Cash Management", "Payroll", "Corporate Credit Card"]
    },
    "few_shot_examples": [
        {
            "scenario": "Retail Collection & Explicit POS Opportunity",
            "input_signals": {
                "108 - Diğer Hazır Değerler": {"balance": 2000000, "volume": 85000000},
                "System Explicit Keyword Matches": ["780.01.002 - POS Komisyon Giderleri"]
            },
            "reasoning_process": {
                "IF": "108 Volume > 0 AND the Python system explicitly flagged 'POS' keywords in sub-accounts",
                "THINKING": "Company collects heavily via credit cards and the system confirmed they are actively paying POS commissions to competitor banks.",
                "ACTION": "Cite the system-provided sub-account directly as definitive proof.",
                "PROPOSAL": "Offer 'ING Fiziki POS', 'ING Sanal POS', or 'CebimPOS' with competitive commission rates."
            },
            "expected_output": "- **POS / Virtual POS**: Detected massive credit card collection volume (108: ₺85,000,000). Confirmed active POS usage via (780.01.002 - POS Komisyon Giderleri). Strong cross-sell for **ING Fiziki POS** or **ING Sanal POS**."
        },
        {
            "scenario": "DBS (Direct Debit) & Competitor Refinancing",
            "input_signals": {
                "120 - Alıcılar": {"balance": 30000000, "volume": 60000000},
                "System Explicit Keyword Matches": ["120.04.050 - B Bankası DBS Alacakları"]
            },
            "reasoning_process": {
                "IF": "120 volume is high AND the system flagged 'DBS' in sub-accounts",
                "THINKING": "Company has a B2B dealer network and the system found proof they use a competitor's DBS.",
                "ACTION": "Highlight the confirmed sub-account to target competitor wallet share.",
                "PROPOSAL": "Offer 'ING DBS' to refinance competitor collections."
            },
            "expected_output": "- **Collection/DBS**: Massive B2B collection volume. System confirms existing competitor usage (120.04.050 - B Bankası DBS Alacakları). Prime target for **ING DBS**."
        },
        {
            "scenario": "Fleet Assets & Confirmed Insurance Cross-Sell",
            "input_signals": {
                "254 - Taşıtlar": {"balance": 18000000, "volume": 2000000},
                "System Explicit Keyword Matches": ["770.03.005 - Araç Kasko Giderleri"]
            },
            "reasoning_process": {
                "IF": "254 > 0 AND system extracted 'KASKO/SİGORTA' from expense accounts",
                "THINKING": "Company owns a fleet and Python confirmed they actively pay insurance premiums.",
                "ACTION": "Use the extracted expense account to size the opportunity.",
                "PROPOSAL": "Cross-sell 'ING Kasko / Filo Sigortası'."
            },
            "expected_output": "- **Insurance / Fleet**: Vehicle assets (254: ₺18,000,000) with confirmed insurance payments (770.03.005 - Araç Kasko Giderleri). Refer to ING Insurance for **ING Kasko / Filo Sigortası**."
        },
        {
            "scenario": "Check Products — Received & Issued Checks",
            "input_signals": {
                "101 - Alınan Çekler": {"balance": 5000000, "volume": 40000000},
                "103 - Verilen Çekler": {"balance": 3000000, "volume": 25000000}
            },
            "reasoning_process": {
                "IF": "101 Volume > 0 OR 103 Volume > 0",
                "THINKING": "Company actively uses checks for both collections and payments. High volume indicates B2B trade dependency on check instruments.",
                "ACTION": "Size the check portfolio and propose ING check financing products.",
                "PROPOSAL": "Offer 'ING Çek Karnesi', 'ING Çek İskontosu/İştira' for received checks, check guarantee for issued checks."
            },
            "expected_output": "- **Check Products**: Active check usage — Received (101: Vol ₺40M, Bal ₺5M), Issued (103: Vol ₺25M, Bal ₺3M). Propose **ING Çek İskontosu** for receivables financing and **ING Çek Karnesi**."
        },
        {
            "scenario": "Trade Finance & Export Revenue (Sector: Export/Manufacturing)",
            "input_signals": {
                "601 - Yurtdışı Satışlar": {"balance": 0, "volume": 50000000},
                "System Explicit Keyword Matches": ["159.02 - İthalat Avansları", "780.05 - Akreditif Komisyonları"]
            },
            "reasoning_process": {
                "IF": "601 Volume > 0 AND system flagged 'İTHALAT/AKREDİTİF' keywords",
                "THINKING": "Company is an active exporter/importer. Export revenue confirms international trade. System found LC commission expenses proving they use trade finance elsewhere.",
                "ACTION": "Cite export revenue and LC expenses as proof of trade finance need.",
                "PROPOSAL": "Offer 'ING Akreditif', 'ING İhracat Faktoring', 'ING Döviz Kredisi'."
            },
            "expected_output": "- **Trade Finance**: Active exporter (601: ₺50M). Confirmed LC usage (780.05 - Akreditif Komisyonları). Propose **ING Akreditif**, **ING İhracat Faktoring**, and **ING Döviz Kredisi**."
        },
        {
            "scenario": "FX Activity & Currency Hedging",
            "input_signals": {
                "646 - Kambiyo Karları": {"balance": 0, "volume": 8000000},
                "656 - Kambiyo Zararları": {"balance": 0, "volume": 12000000}
            },
            "reasoning_process": {
                "IF": "646 + 656 total volume > 0",
                "THINKING": "Company has significant FX exposure. Net FX loss (656 > 646) indicates unhedged currency risk.",
                "ACTION": "Calculate net FX impact and recommend hedging products.",
                "PROPOSAL": "Offer 'ING e-Turuncu Kur' for FX trading, 'ING Forward/Opsiyon' for hedging."
            },
            "expected_output": "- **FX & Hedging**: Significant FX exposure (646: ₺8M gains, 656: ₺12M losses = Net ₺4M loss). Unhedged risk. Propose **ING e-Turuncu Kur** and **ING Forward/Opsiyon** for currency hedging."
        },
        {
            "scenario": "Payroll & Personnel (Sector-aware: Manufacturing/Services)",
            "input_signals": {
                "720/730/760/770 - Personnel Expenses": {"balance": 0, "volume": 15000000}
            },
            "reasoning_process": {
                "IF": "Personnel expense volume > 0",
                "THINKING": "Company has significant payroll. Manufacturing/services firms with large workforce = payroll banking opportunity.",
                "ACTION": "Estimate employee count from average salary and propose payroll package.",
                "PROPOSAL": "Offer 'ING Maaş Ödemesi Paketi', employee banking cross-sell."
            },
            "expected_output": "- **Payroll**: ₺15M personnel expenses. Estimate ~200 employees. Propose **ING Maaş Ödemesi Paketi** with employee banking cross-sell (ING Turuncu Hesap)."
        },
        {
            "scenario": "Corporate Credit Card Signal",
            "input_signals": {
                "309 - Diğer Mali Borçlar": {"balance": 2000000, "volume": 10000000},
                "System Explicit Keyword Matches": ["309.01 - Şirket Kredi Kartı Borçları"]
            },
            "reasoning_process": {
                "IF": "309 balance > 0 AND system flagged 'KREDİ KARTI' keywords",
                "THINKING": "Company uses corporate credit cards actively with competitor banks.",
                "ACTION": "Cite the flagged sub-account as proof of competitor card usage.",
                "PROPOSAL": "Offer 'ING Bonus Business Kart' with competitive limits and cashback."
            },
            "expected_output": "- **Corporate Credit Card**: Active card usage (309.01 - Şirket Kredi Kartı Borçları: ₺2M balance, ₺10M volume). Propose **ING Bonus Business Kart** with competitive limits."
        },
        {
            "scenario": "Supplier Finance (TFS/SCF) — Manufacturing Sector",
            "input_signals": {
                "320 - Satıcılar": {"balance": 20000000, "volume": 80000000},
                "System Explicit Keyword Matches": ["320.05 - TFS Borçları"]
            },
            "reasoning_process": {
                "IF": "320 volume is high AND system flagged 'TFS/TEDARİKÇİ' keywords AND sector is Manufacturing/Trading",
                "THINKING": "Company has large supplier payables and already uses supply chain finance. Sector confirms strong upstream dependency.",
                "ACTION": "Size the SCF opportunity from trade payables volume.",
                "PROPOSAL": "Offer 'ING Tedarik Zinciri Finansmanı' to capture supplier payments."
            },
            "expected_output": "- **Supplier Finance (TFS)**: Large supplier payables (320: ₺80M volume). Confirmed TFS usage (320.05 - TFS Borçları). Manufacturing sector = strong SCF fit. Propose **ING Tedarik Zinciri Finansmanı**."
        },
        {
            "scenario": "Deposit Capture Opportunity",
            "input_signals": {
                "102 - Bankalar": {"balance": 25000000, "volume": 150000000},
                "ING Not Present in 102 sub-accounts": True
            },
            "reasoning_process": {
                "IF": "102 balance > 0 AND ING is NOT present in 102 sub-accounts",
                "THINKING": "Company holds significant deposits at competitor banks. ING has zero share of deposits — this is a greenfield opportunity.",
                "ACTION": "Flag as priority deposit capture target.",
                "PROPOSAL": "Offer 'ING Turuncu Vadesiz', 'ING e-Turuncu Mevduat' with competitive rates to capture deposit flow."
            },
            "expected_output": "- **Deposit Capture**: ₺25M deposits at competitors, ING has 0% share. Priority acquisition target. Propose **ING Turuncu Vadesiz** and **ING e-Turuncu Mevduat** with competitive rates."
        }
    ]
}

FEW_SHOT_PROMPT_ADDITION = f"""
## REASONING HEURISTICS & ING TURKEY FEW-SHOT EXAMPLES
Internalize this logic. Rely ONLY on the data and keyword matches explicitly provided to you by the system. Do not guess sub-accounts.
{json.dumps(MIZAN_FEW_SHOT_GUIDE, indent=2, ensure_ascii=False)}
"""


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
        },
        # ── 9. ANALYST INSIGHTS ──
        "analyst_insights": {
            "quant_summary": str(ratios.get("llm_interpretation", ""))[:2500],
            "product_summary": str(signals.get("llm_interpretation", ""))[:2500]
        }
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
                pattern = "|".join(keywords)
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
            account_mapping = dict(zip(final_df["account_code"].astype(str), final_df["account_name"].astype(str)))

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
       marketing_sales_expenses = get_signal_metrics(["760"], [], "debit")
        
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
           signal_lines = []
           for name, data in product_signals.items():
               if isinstance(data, dict) and (data.get("balance", 0) != 0 or data.get("volume", 0) != 0):
                   mapping_str = ", ".join(f"'{c} - {n}'" for c, n in data.get("account_mapping", {}).items())
                   signal_lines.append(
                       f"- **{name}**: Debit=₺{data.get('debit', 0):,.0f}, Credit=₺{data.get('credit', 0):,.0f}, "
                       f"BalDebit=₺{data.get('balance_debit', 0):,.0f}, BalCredit=₺{data.get('balance_credit', 0):,.0f}, "
                       f"Net=₺{data.get('balance', 0):,.0f}, Volume=₺{data.get('volume', 0):,.0f}, "
                       f"Sub-Accounts: {{{mapping_str}}}"
                   )
           signal_summary = "\n".join(signal_lines) if signal_lines else "No significant product signals detected."
           annualization = round(12 / period_months, 2) if period_months else 1.0
           sector = state.get('sector', 'General')
           prompt = (
               f"⏱️ DATA PERIOD: {donem_label} ({period_days} days). "
               f"Annualization factor: {annualization}x\n\n"
               f"Analyze the banking product signals for **{state.get('company_name', 'Company')}** "
               f"(Sector: **{sector}**).\n\n"
               f"## SECTOR-AWARE PRODUCT PRIORITIZATION\n"
               f"The company operates in the **{sector}** sector. Prioritize product suggestions "
               f"relevant to this sector.\n\n"
               f"## ACTIVE PRODUCT SIGNALS (ALL 5 COLUMNS):\n"
               f"{signal_summary}\n\n"
               f"## DETAILED PRODUCT BREAKDOWN:\n"
               f"### 1. LOANS (300+400): ST=₺{loan_300['balance']:,.0f}, LT=₺{loan_400['balance']:,.0f}, FinExp=₺{financial_expenses['balance']:,.0f}\n"
               f"### 2. POS: Collection(108)=₺{pos_collection['volume']:,.0f}, Expenses=₺{pos_expenses['balance']:,.0f}\n"
               f"### 3. DBS: Bal=₺{dbs_usage['balance']:,.0f}, Vol=₺{dbs_usage['volume']:,.0f}\n"
               f"### 4. SCF/TFS: Bal=₺{tfs_usage['balance']:,.0f}, Vol=₺{tfs_usage['volume']:,.0f}\n"
               f"### 5. CORP CARD: Bal=₺{corporate_credit_card['balance']:,.0f}, Vol=₺{corporate_credit_card['volume']:,.0f}\n"
               f"### 6. FLEET/INSURANCE: Assets=₺{vehicle_fleet_assets['balance']:,.0f}, Insurance=₺{insurance_expenses['balance']:,.0f}\n"
               f"### 7. CHECKS: Recv(101)=₺{received_checks['balance']:,.0f}/₺{received_checks['volume']:,.0f}, Issued(103)=₺{issued_checks['balance']:,.0f}/₺{issued_checks['volume']:,.0f}\n"
               f"### 8. TRADE FINANCE: Export(601)=₺{export_revenue['volume']:,.0f}, Expenses=₺{trade_finance_expenses['balance']:,.0f}\n"
               f"### 9. FX/SWIFT: Net=₺{fx_volume['balance']:,.0f}, Activity=₺{fx_volume['volume']:,.0f}, SWIFT=₺{swift_transfer_expenses['balance']:,.0f}\n"
               f"### 10. PAYROLL: Vol=₺{payroll_personnel_volume['volume']:,.0f}\n"
               f"### 11. SECTORAL: Construction=₺{construction_costs['balance']:,.0f}, Machinery=₺{machinery_equipment['balance']:,.0f}, CommGoods=₺{commercial_goods['balance']:,.0f}\n\n"
               f"## OUTPUT FORMAT INSTRUCTIONS:\n"
               f"Structure your output into EXACTLY these sections:\n\n"
               f"### SECTION 1: HIGH-PRIORITY SIGNALS (sorted by revenue potential)\n"
               f"For each product category with non-zero signals, use this EXACT format:\n\n"
               f"#### [Product Name] — Revenue Potential: [HIGH/MEDIUM/LOW]\n"
               f"- **IF**: [State the data condition that triggered the signal]\n"
               f"- **THINKING**: [Sector-aware interpretation — why this matters for {sector}]\n"
               f"- **DATA**: Debit=₺X, Credit=₺X, BalDebit=₺X, BalCredit=₺X, Net=₺X, Volume=₺X\n"
               f"- **ACCOUNTS**: [List exact sub-account codes from mapping — DO NOT invent]\n"
               f"- **PROPOSAL**: [Specific ING Bank Turkey product] → Estimated annual revenue: ₺X\n\n"
               f"### SECTION 2: CROSS-SELL GAPS\n"
               f"List products with ZERO signals where ING can create new opportunities.\n"
               f"Format: - [Product] → ING Opportunity: [description]\n\n"
               f"### SECTION 3: REVENUE SUMMARY TABLE\n"
               f"| Product | Current Volume | ING Product | Est. Annual Revenue | Priority |\n"
               f"|---------|---------------|-------------|--------------------|---------|\n"
               f"[Fill with data from Section 1]\n\n"
               f"CRITICAL RULES:\n"
               f"- Only reference accounts explicitly provided. DO NOT invent sub-account codes.\n"
               f"- Sort HIGH-PRIORITY signals by estimated revenue (highest first).\n"
               f"- Use annualization factor {annualization}x for sub-annual periods.\n\n"
               + FEW_SHOT_PROMPT_ADDITION
           )
           llm_text = invoke_llm(PRODUCT_ANALYST_SYSTEM_PROMPT, prompt, temperature=0.2, max_tokens=3000)
           self.metrics.record_llm_call(tokens=len(llm_text.split()))
           logger.info(f"✅ Product analyst LLM interpretation: {len(llm_text)} chars")
       except Exception as e:
           logger.warning(f"LLM skipped: {e}")
           llm_text = "LLM interpretation unavailable."
       product_signals["llm_interpretation"] = llm_text
       return {"product_signals": product_signals, "retry_count": retry_count + 1}
 
# Module-level callable for LangGraph
product_analyst_agent = ProductAnalystAgent()