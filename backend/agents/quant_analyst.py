"""
Agent 2: Quantitative Analyst — Local calc + LLM interpretation

Refactored to use BaseAgent for tracing and error isolation.
All raw_values include hesap_kodu_aciklama (account code + Turkish description)
to ensure every number is properly cited.
"""

import logging
import pandas as pd
from agents.base import BaseAgent
from llm_config import invoke_llm, QUANT_ANALYST_SYSTEM_PROMPT

logger = logging.getLogger("swarm.agents.quant_analyst")

# ── Turkish Chart of Accounts (Tekdüzen Hesap Planı) ──
HESAP_KODU_ACIKLAMA = {
    "100": "KASA",
    "101": "ALINAN ÇEKLER",
    "102": "BANKALAR",
    "103": "VERİLEN ÇEKLER VE ÖDEME EMİRLERİ",
    "108": "DİĞER HAZIR DEĞERLER",
    "120": "ALICILAR",
    "121": "ALACAK SENETLERİ",
    "126": "VERİLEN DEPOZİTO VE TEMİNATLAR",
    "150": "İLK MADDE VE MALZEME",
    "151": "YARI MAMULLER - ÜRETİM",
    "152": "MAMULLER",
    "153": "TİCARİ MALLAR",
    "159": "VERİLEN SİPARİŞ AVANSLARI",
    "180": "GELECEK AYLARA AİT GİDERLER",
    "190": "İNDİRİLECEK KDV",
    "191": "İNDİRİLECEK KDV",
    "196": "PEŞİN ÖDENEN VERGİLER VE FONLAR",
    "240": "İŞTİRAKLERE SERMAYE TAAHHÜTLERİ",
    "242": "İŞTİRAKLER",
    "252": "BİNALAR",
    "253": "TESİS, MAKİNE VE CİHAZLAR",
    "254": "TAŞITLAR",
    "255": "DEMİRBAŞLAR",
    "257": "BİRİKMİŞ AMORTİSMANLAR (-)",
    "260": "HAKLAR",
    "264": "ÖZEL MALİYETLER",
    "268": "BİRİKMİŞ AMORTİSMANLAR (-)",
    "300": "BANKA KREDİLERİ",
    "303": "UZUN VADELİ KREDİLERİN ANAPARA TAKSİTLERİ VE FAİZLERİ",
    "309": "DİĞER MALİ BORÇLAR",
    "320": "SATICILAR",
    "321": "BORÇ SENETLERİ",
    "335": "PERSONELE BORÇLAR",
    "336": "DİĞER ÇEŞİTLİ BORÇLAR",
    "340": "ALINAN SİPARİŞ AVANSLARI",
    "360": "ÖDENECEK VERGİ VE FONLAR",
    "361": "ÖDENECEK SOSYAL GÜVENLİK KESİNTİLERİ",
    "380": "GELECEK AYLARA AİT GELİRLER",
    "391": "HESAPLANAN KDV",
    "400": "BANKA KREDİLERİ",
    "420": "SATICILAR",
    "429": "DİĞER TİCARİ BORÇLAR",
    "431": "ORTAKLARA BORÇLAR",
    "440": "ALINAN SİPARİŞ AVANSLARI (UV)",
    "472": "KIDEM TAZMİNATI KARŞILIĞI",
    "500": "SERMAYE",
    "520": "HİSSE SENETLERİ İHRAÇ PRİMLERİ",
    "540": "YASAL YEDEKLER",
    "570": "GEÇMİŞ YILLAR KÂRLARI",
    "580": "GEÇMİŞ YILLAR ZARARLARI (-)",
    "590": "DÖNEM NET KÂRI",
    "591": "DÖNEM NET ZARARI (-)",
    "600": "YURTİÇİ SATIŞLAR",
    "601": "YURTDIŞI SATIŞLAR",
    "610": "SATIŞ İNDİRİMLERİ (-)",
    "611": "SATIŞ İSKONTOLARI (-)",
    "612": "DİĞER İNDİRİMLER (-)",
    "620": "SATILAN MALIN MALİYETİ (-)",
    "621": "SATILAN HİZMET MALİYETİ (-)",
    "630": "ARAŞTIRMA VE GELİŞTİRME GİDERLERİ (-)",
    "631": "PAZARLAMA, SATIŞ VE DAĞITIM GİDERLERİ (-)",
    "632": "GENEL YÖNETİM GİDERLERİ (-)",
    "640": "İŞTİRAKLERDEN TEMETTÜ GELİRLERİ",
    "642": "FAİZ GELİRLERİ",
    "644": "KONUSUKALMAMIŞPROVİZYON KARŞILIĞI",
    "646": "KAMBİYO KÂRLARI",
    "647": "REESKONT FAİZ GELİRLERİ",
    "648": "ENFLASYON DÜZELTME KÂRLARI",
    "649": "DİĞER OLAĞAN GELİR VE KÂRLAR",
    "653": "KOMİSYON GİDERLERİ (-)",
    "654": "KARŞILIK GİDERLERİ (-)",
    "655": "TÜM DİĞER İNDİRİM VE GİDERLER (-)",
    "656": "KAMBİYO ZARARLARI (-)",
    "657": "REESKONT FAİZ GİDERLERİ (-)",
    "658": "ENFLASYON DÜZELTME ZARARLARI (-)",
    "659": "DİĞER OLAĞAN GİDER VE ZARARLAR (-)",
    "660": "KISA VADELİ BORÇLANMA GİDERLERİ (-)",
    "661": "UZUN VADELİ BORÇLANMA GİDERLERİ (-)",
    "679": "DİĞER OLAĞANDIŞI GELİR VE KÂRLAR",
    "680": "ÇALIŞILMAYAN KISIM GİDER VE ZARARLARI (-)",
    "681": "ÖNCEKİ DÖNEM GİDER VE ZARARLARI (-)",
    "689": "DİĞER OLAĞANDIŞI GİDER VE ZARARLAR (-)",
    "691": "DÖNEM KÂRI VEYA ZARARI (-)",
    "780": "FİNANSMAN GİDERLERİ",
    "780.01": "POS KOMİSYON GİDERLERİ",
}


def _get_aciklama(code: str) -> str:
    """Get the Turkish description for an account code."""
    if code in HESAP_KODU_ACIKLAMA:
        return HESAP_KODU_ACIKLAMA[code]
    # Try prefix matching for compound codes
    prefix = code.split(".")[0]
    if prefix in HESAP_KODU_ACIKLAMA:
        return HESAP_KODU_ACIKLAMA[prefix]
    return f"HESAP {code}"


class QuantAnalystAgent(BaseAgent):
    name = "quant_analyst"
    description = "Calculate financial ratios from standardized Mizan data"
    required_inputs = ["standardized_mizan"]
    output_keys = ["financial_ratios"]

    def execute(self, state: dict) -> dict:
        retry_count = state.get("retry_count", 0)
        verification_errors = state.get("verification_errors", "")

        logger.info(f"Calculating ratios (attempt #{retry_count + 1})...")
        if verification_errors:
            logger.info(f"Verification feedback: {verification_errors}")

        standardized = state.get("standardized_mizan", [])
        if not standardized:
            return {"financial_ratios": {"error": "No standardized mizan data"}, "retry_count": retry_count + 1}

        df = pd.DataFrame(standardized)

        def bal(code: str) -> float:
            """Get net balance for an account code (exact match or prefix sum)."""
            # Try exact match first
            m = df[df["account_code"] == code]
            if not m.empty:
                return float(m.iloc[0]["debit"] - m.iloc[0]["credit"])
            # Fall back to prefix matching (sum all sub-accounts)
            m = df[df["account_code"].str.startswith(code)]
            if not m.empty:
                return float(m["debit"].sum() - m["credit"].sum())
            return 0.0

        def bal_named(code: str) -> dict:
            """Get balance with hesap kodu açıklama citation."""
            return {
                "value": abs(bal(code)),
                "hesap_kodu": code,
                "hesap_kodu_aciklama": _get_aciklama(code),
            }

        # ── LOCAL CALCULATIONS ──
        # Profitability Metrics
        revenue_600 = abs(bal("600"))
        cogs_620 = abs(bal("620"))
        gross_profit = revenue_600 - cogs_620
        gross_margin = ((gross_profit) / revenue_600 * 100) if revenue_600 else 0

        op_exp_630 = abs(bal("630"))
        op_exp_631 = abs(bal("631"))
        op_exp_632 = abs(bal("632"))
        op_expenses = op_exp_630 + op_exp_631 + op_exp_632
        operating_profit = gross_profit - op_expenses
        operating_margin = (operating_profit / revenue_600 * 100) if revenue_600 else 0

        # Liquidity Metrics
        ca_codes = df[df["category"] == "Current Assets"]["account_code"].tolist()
        current_assets = float(df[df["category"] == "Current Assets"]["debit"].sum())
        stl_codes = df[df["category"] == "Short-Term Liabilities"]["account_code"].tolist()
        short_term_liab = float(df[df["category"] == "Short-Term Liabilities"]["credit"].sum())

        current_ratio = (current_assets / short_term_liab) if short_term_liab else 0

        inv_150 = abs(bal("150"))
        inv_151 = abs(bal("151"))
        inv_152 = abs(bal("152"))
        inv_153 = abs(bal("153"))
        inventory = inv_150 + inv_151 + inv_152 + inv_153
        quick_ratio = ((current_assets - inventory) / short_term_liab) if short_term_liab else 0

        # Checks & Liquid Instruments
        received_checks_101 = abs(bal("101"))
        given_checks_103 = abs(bal("103"))

        # ── COMPETITOR BANK ANALYSIS (102, 300, 400) ──
        def get_bank_breakdown(parent_code: str) -> list:
            """Extracts sub-account balances to identify competitor bank shares."""
            shares = []
            total_bal = 0
            for _, row in df.iterrows():
                code = str(row["account_code"])
                # Identify sub-accounts (e.g., '102.01', '102 01')
                if code.startswith(parent_code) and code != parent_code:
                    net_bal = abs(float(row["debit"]) - float(row["credit"]))
                    if net_bal > 0:
                        shares.append({
                            "name": str(row.get("account_name", code)),
                            "balance": net_bal
                        })
                        total_bal += net_bal
            
            # Calculate percentages
            for s in shares:
                s["share_pct"] = (s["balance"] / total_bal * 100) if total_bal else 0
            
            # Sort highest balance first
            shares.sort(key=lambda x: x["balance"], reverse=True)
            return shares

        banks_102 = get_bank_breakdown("102")
        banks_300 = get_bank_breakdown("300")
        banks_400 = get_bank_breakdown("400")

        def fmt_shares(shares):
            if not shares:
                return "No detailed sub-account data available."
            return ", ".join([f"{s['name']} (₺{s['balance']:,.0f} - %{s['share_pct']:.1f})" for s in shares[:5]])

        # Leverage & Debt Metrics
        total_liab = float(
            df[df["category"] == "Short-Term Liabilities"]["credit"].sum() +
            df[df["category"] == "Long-Term Liabilities"]["credit"].sum()
        )
        total_equity = float(df[df["category"] == "Equity"]["credit"].sum())
        debt_to_equity = (total_liab / total_equity) if total_equity else 0

        short_term_loans_300 = abs(bal("300"))
        credit_card_expenses_309 = abs(bal("309"))
        long_term_loans_400 = abs(bal("400"))
        total_bank_loans = short_term_loans_300 + credit_card_expenses_309 + long_term_loans_400
        bank_debt_ratio = (total_bank_loans / total_liab * 100) if total_liab else 0

        fin_exp_780 = abs(bal("780"))
        fin_expense_ratio = (fin_exp_780 / revenue_600 * 100) if revenue_600 else 0

        # Efficiency & Working Capital
        trade_recv_120 = abs(bal("120"))
        trade_recv_121 = abs(bal("121"))
        trade_receivables = trade_recv_120 + trade_recv_121
        collection_period = (trade_receivables / revenue_600 * 365) if revenue_600 else 0

        trade_pay_320 = abs(bal("320"))
        trade_pay_321 = abs(bal("321"))
        trade_payables = trade_pay_320 + trade_pay_321
        payment_period = (trade_payables / cogs_620 * 365) if cogs_620 else 0

        # Transactional / Behavioral Metrics
        pos_780_01 = abs(bal("780.01"))
        pos_ratio = (pos_780_01 / revenue_600 * 100) if revenue_600 else 0

        ratios = {
            "gross_margin": {
                "value": round(gross_margin, 2), "unit": "%",
                "formula": "(Revenue[600] - COGS[620]) / Revenue[600] × 100",
                "accounts_used": ["600", "620"],
                "raw_values": {
                    "revenue_600": bal_named("600"),
                    "cogs_620": bal_named("620"),
                    "gross_profit": gross_profit,
                }
            },
            "operating_margin": {
                "value": round(operating_margin, 2), "unit": "%",
                "formula": "(Gross Profit - Operating Expenses[630+631+632]) / Revenue[600] × 100",
                "accounts_used": ["600", "620", "630", "631", "632"],
                "raw_values": {
                    "revenue_600": bal_named("600"),
                    "cogs_620": bal_named("620"),
                    "op_exp_630": bal_named("630"),
                    "op_exp_631": bal_named("631"),
                    "op_exp_632": bal_named("632"),
                    "operating_profit": operating_profit,
                }
            },
            "current_ratio": {
                "value": round(current_ratio, 2), "unit": "x",
                "formula": "Current Assets [1xx] / Short-Term Liabilities [3xx]",
                "accounts_used": ca_codes,
                "raw_values": {
                    "current_assets": current_assets,
                    "short_term_liabilities": short_term_liab,
                }
            },
            "quick_ratio": {
                "value": round(quick_ratio, 2), "unit": "x",
                "formula": "(Current Assets[1xx] - Inventory[15x]) / Short-Term Liabilities[3xx]",
                "accounts_used": ca_codes + ["150", "151", "152", "153"],
                "raw_values": {
                    "liquid_assets": current_assets - inventory,
                    "short_term_liabilities": short_term_liab,
                    "received_checks_101": bal_named("101"),
                    "given_checks_103": bal_named("103"),
                    "inventory_150": bal_named("150"),
                    "inventory_151": bal_named("151"),
                    "inventory_152": bal_named("152"),
                    "inventory_153": bal_named("153"),
                }
            },
            "collection_period": {
                "value": round(collection_period, 0), "unit": "days",
                "formula": "Trade Receivables[120+121] / Revenue[600] × 365",
                "accounts_used": ["120", "121", "600"],
                "raw_values": {
                    "trade_receivables_120": bal_named("120"),
                    "trade_receivables_121": bal_named("121"),
                    "trade_receivables": trade_receivables,
                    "revenue_600": bal_named("600"),
                }
            },
            "payment_period": {
                "value": round(payment_period, 0), "unit": "days",
                "formula": "Trade Payables[320+321] / COGS[620] × 365",
                "accounts_used": ["320", "321", "620"],
                "raw_values": {
                    "trade_payables_320": bal_named("320"),
                    "trade_payables_321": bal_named("321"),
                    "trade_payables": trade_payables,
                    "cogs_620": bal_named("620"),
                }
            },
            "debt_to_equity": {
                "value": round(debt_to_equity, 2), "unit": "x",
                "formula": "Total Liabilities [3xx+4xx] / Equity [5xx]",
                "accounts_used": stl_codes + ["400", "500", "570"],
                "raw_values": {
                    "total_liabilities": total_liab,
                    "total_equity": total_equity,
                    "sermaye_500": bal_named("500"),
                    "gecmis_yil_karlari_570": bal_named("570"),
                }
            },
            "bank_debt_ratio": {
                "value": round(bank_debt_ratio, 2), "unit": "%",
                "formula": "Bank Loans[300+309+400] / Total Liabilities × 100",
                "accounts_used": ["300", "309", "400"],
                "raw_values": {
                    "banka_kredileri_kv_300": bal_named("300"),
                    "diger_mali_borclar_309": bal_named("309"),
                    "banka_kredileri_uv_400": bal_named("400"),
                    "total_bank_loans": total_bank_loans,
                    "total_liabilities": total_liab,
                }
            },
            "financial_expense_ratio": {
                "value": round(fin_expense_ratio, 2), "unit": "%",
                "formula": "Financial Expenses [780] / Revenue [600] × 100",
                "accounts_used": ["780", "600"],
                "raw_values": {
                    "finansman_giderleri_780": bal_named("780"),
                    "revenue_600": bal_named("600"),
                }
            },
            "pos_commission_ratio": {
                "value": round(pos_ratio, 2), "unit": "%",
                "formula": "POS Commission [780.01] / Revenue [600] × 100",
                "accounts_used": ["780.01", "600"],
                "raw_values": {
                    "pos_komisyon_780_01": bal_named("780.01"),
                    "revenue_600": bal_named("600"),
                }
            },
            # Save competitor data in the dict so downstream Strategist can access it too
            "competitor_banks": {
                "102": banks_102,
                "300": banks_300,
                "400": banks_400
            }
        }

        for n, d in ratios.items():
            if isinstance(d, dict) and "value" in d:
                logger.info(f"  - {n}: {d['value']}{d['unit']}")

        # ── LLM INTERPRETATION ──
        llm_text = ""
        try:
            summary = "\n".join(
                f"- {n}: {d['value']}{d['unit']} (accounts: {d['accounts_used']})"
                for n, d in ratios.items() if isinstance(d, dict) and "value" in d
            )
            
            prompt = (
                f"Analyze these financial ratios for {state.get('company_name', 'Company')} ({state.get('sector', 'General')}):\n\n"
                f"{summary}\n\n"
                f"### RAW DATA FOR ANALYSIS (with Hesap Kodu Açıklama):\n"
                f"- **600-YURTİÇİ SATIŞLAR:** ₺{revenue_600:,.0f} | **620-SATILAN MALIN MALİYETİ:** ₺{cogs_620:,.0f}\n"
                f"- **630-ARAŞTIRMA GELİŞTİRME GİD.:** ₺{op_exp_630:,.0f} | **631-PAZARLAMA SATIŞ GİD.:** ₺{op_exp_631:,.0f} | **632-GENEL YÖNETİM GİD.:** ₺{op_exp_632:,.0f}\n"
                f"- **Dönen Varlıklar (1xx):** ₺{current_assets:,.0f} | **150-153 STOKLAR:** ₺{inventory:,.0f} | **Kısa Vadeli Borçlar (3xx):** ₺{short_term_liab:,.0f}\n"
                f"- **120-ALICILAR:** ₺{trade_recv_120:,.0f} | **121-ALACAK SENETLERİ:** ₺{trade_recv_121:,.0f}\n"
                f"- **320-SATICILAR:** ₺{trade_pay_320:,.0f} | **321-BORÇ SENETLERİ:** ₺{trade_pay_321:,.0f}\n"
                f"- **101-ALINAN ÇEKLER:** ₺{received_checks_101:,.0f} | **103-VERİLEN ÇEKLER:** ₺{given_checks_103:,.0f}\n"
                f"- **Total Liabilities:** ₺{total_liab:,.0f} | **300-BANKA KREDİLERİ (KV):** ₺{short_term_loans_300:,.0f} | **309-DİĞER MALİ BORÇLAR:** ₺{credit_card_expenses_309:,.0f} | **400-BANKA KREDİLERİ (UV):** ₺{long_term_loans_400:,.0f}\n"
                f"- **500-SERMAYE:** ₺{abs(bal('500')):,.0f} | **570-GEÇMİŞ YIL KÂRLARI:** ₺{abs(bal('570')):,.0f} | **Total Equity:** ₺{total_equity:,.0f}\n"
                f"- **780-FİNANSMAN GİDERLERİ:** ₺{fin_exp_780:,.0f} | **780.01-POS KOMİSYON GİDERLERİ:** ₺{pos_780_01:,.0f}\n\n"
                f"### COMPETITOR BANK DISTRIBUTION:\n"
                f"- **102-BANKALAR (Deposits):** {fmt_shares(banks_102)}\n"
                f"- **300-BANKA KREDİLERİ KV (ST Loans):** {fmt_shares(banks_300)}\n"
                f"- **400-BANKA KREDİLERİ UV (LT Loans):** {fmt_shares(banks_400)}\n\n"
                f"Based on your system instructions, structure your analysis into these four exact pillars:\n"
                f"1. PROFITABILITY\n"
                f"2. LIQUIDITY & WORKING CAPITAL\n"
                f"3. LEVERAGE & DEPENDENCY\n"
                f"4. TRANSACTIONAL COST\n\n"
                f"IMPORTANT: Cite every value with its hesap kodu and açıklama (e.g., '101-ALINAN ÇEKLER: ₺X').\n"
                f"Identify red flags and map the mathematical groundwork for downstream cross-selling, explicitly utilizing the competitor bank distribution."
            )
            llm_text = invoke_llm(QUANT_ANALYST_SYSTEM_PROMPT, prompt, temperature=0.2, max_tokens=1500)
            self.metrics.record_llm_call(tokens=len(llm_text.split()))
            logger.info(f"✅ LLM interpretation: {len(llm_text)} chars")
        except Exception as e:
            logger.warning(f"LLM skipped: {e}")
            llm_text = "LLM interpretation unavailable."

        ratios["llm_interpretation"] = llm_text
        return {"financial_ratios": ratios, "retry_count": retry_count + 1}


# Module-level callable for LangGraph
quant_analyst_agent = QuantAnalystAgent()