"""
Few-Shot Scenario Library & Product Classification
====================================================
Central library of ING Turkey product few-shot scenarios for the
product_analyst and strategist agents.

Design goals:
1. Each scenario is keyed by a `product_key` and mapped to the exact
   product_signal dictionary keys produced by product_analyst.
2. `classify_product_opportunities()` selects ONLY the scenarios whose
   signals are actually present in the Mizan data (token-efficient:
   irrelevant scenarios are never injected into the prompt).
3. Products already used by the customer (local DB flags == 1) are
   EXCLUDED from recommendation scenarios and returned separately so
   the prompts can forbid re-recommending them.
4. `build_few_shot_injection()` returns separate system-prompt and
   user-prompt additions.
"""

import json
import logging
import re
import unicodedata

logger = logging.getLogger("swarm.agents.few_shot_library")

# ══════════════════════════════════════════════════════════════
# CORE RULES — always injected into the SYSTEM prompt when at
# least one scenario is selected. Kept short for token efficiency.
# ══════════════════════════════════════════════════════════════
CORE_MIZAN_RULES = [
    "DO NOT invent or search for account codes. Only use the exact balances, volumes, and sub-account names explicitly provided in the prompt by the Python system.",
    "Follow the IF -> THINKING -> ACTION -> PROPOSAL reasoning structure for your analysis.",
    "Map all proposals to specific ING Turkey products (e.g., 'ING DBS', 'ING e-Turuncu Kur', 'ING e-Turuncu Mevduat', 'ING Kredi Limit Çalışması', 'ING Bonus Business').",
    "Use the company's SECTOR information to prioritize relevant products. A manufacturing firm needs different products than a trading or services firm.",
    "Distinguish between BALANCE (stock at period-end) and VOLUME (flow during period). High volume with low balance = healthy turnover.",
    "You MUST cite exact sub-account names and volume or balance information used in your reasoning for proposed product type.",
    "If a product is listed as CURRENTLY USED (bank core system flag), DO NOT recommend it as a new cross-sell opportunity.",
]

# Aligned 1:1 with the sector keys in data/tcmb_sector_benchmarks.json
# (Manufacturing, Trading, Retail, Construction, Services, Textile,
#  Food & Beverage, Automotive, Energy, Transportation & Logistics,
#  Technology, Agriculture, Tourism, Chemicals, General) plus the
# Export/Import cross-sector modifiers used by predict_sector
# (e.g., 'Retail + Export'). Tested in tests/test_db_and_fewshot.py.
SECTOR_PRODUCT_PRIORITY = {
    "Manufacturing / Üretim": ["Leasing (253)", "Working Capital Loans", "Supply Chain Finance", "Fleet Insurance", "Payroll", "FX (if exporter)"],
    "Trading / Ticaret": ["POS/VPOS", "DBS", "Corporate Credit Card", "Check Products", "Commercial Loans", "FX/SWIFT"],
    "Retail / Perakende": ["POS/VPOS", "DBS", "Cash Management", "Payroll", "Corporate Credit Card"],
    "Construction / İnşaat": ["Letter of Guarantee (Teminat Mektubu)", "Progress Payment Finance (170/350)", "Leasing (253)", "Surety Bonds"],
    "Services / Hizmet": ["POS/VPOS", "Payroll", "Corporate Credit Card", "Cash Management", "Digital Banking"],
    "Textile / Tekstil": ["Export Factoring", "FX Hedging (Forward/Opsiyon)", "Working Capital Loans (inventory cycle)", "Trade Finance (Akreditif)", "Leasing (253)", "Payroll"],
    "Food & Beverage / Gıda": ["DBS (dealer/distributor collections)", "Seasonal Working Capital", "Supply Chain Finance", "Inventory Finance", "POS/VPOS"],
    "Automotive / Otomotiv": ["Floor-Plan/Stock Financing", "Supplier Finance (TFS)", "Fleet Leasing", "Fleet Insurance (Kasko)", "DBS"],
    "Energy / Enerji": ["Project Finance", "FX Hedging", "Long-Term Investment Loans", "Letter of Guarantee", "Loan Refinancing"],
    "Transportation & Logistics / Lojistik": ["Fleet Leasing (254)", "Fleet Insurance (Kasko)", "Fuel Card Programs", "FX/SWIFT", "Cash Management"],
    "Technology / Bilişim": ["Payroll", "Virtual POS / E-Commerce", "Cash Management (API banking)", "FX (license payments)", "Corporate Credit Card"],
    "Agriculture / Tarım": ["Seasonal Working Capital", "Harvest-Cycle Inventory Finance", "Equipment Leasing", "DBS"],
    "Tourism / Turizm": ["POS/VPOS Acquiring", "Seasonal Credit Lines", "FX Products (natural hedge)", "Investment Loan Refinancing"],
    "Chemicals / Kimya": ["Import LC (Akreditif)", "FX Hedging", "Inventory Working Capital", "Trade Finance"],
    "Export / İhracat": ["Trade Finance", "Letter of Credit", "FX/e-Turuncu Kur", "SWIFT", "Export Factoring"],
    "Import / İthalat": ["Trade Finance", "Letter of Credit", "FX/e-Turuncu Kur", "Import Loans", "Customs Guarantee"],
    "General / Genel": ["Cash Management", "POS/VPOS", "Working Capital Loans", "Deposit Products"],
}

# ══════════════════════════════════════════════════════════════
# FEW-SHOT SCENARIOS
# `signal_keys` must match the keys of the product_signals dict
# produced by ProductAnalystAgent. `product_key` must match the
# product flag column names in local_db.PRODUCT_FLAG_COLUMNS.
# ══════════════════════════════════════════════════════════════
FEW_SHOT_SCENARIOS = [
    {
        "product_key": "pos",
        "scenario": "Retail Collection & Explicit POS Opportunity",
        "signal_keys": ["POS Collection (108)", "POS / Virtual POS"],
        "sector_affinity": ["Retail", "Trading", "Services", "Food", "Perakende", "Ticaret"],
        "input_signals": {
            "108 - Diğer Hazır Değerler": {"balance": 2000000, "volume": 85000000},
            "System Explicit Keyword Matches": ["780.XX.YYY - POS Komisyon Giderleri"],
        },
        "reasoning_process": {
            "IF": "108 Volume > 0 AND the Python system explicitly flagged 'POS' keywords in sub-accounts",
            "THINKING": "Company collects heavily via credit cards and the system confirmed they are actively paying POS commissions to competitor banks.",
            "ACTION": "Cite the system-provided sub-account directly as definitive proof.",
            "PROPOSAL": "Offer 'ING Fiziki POS', 'ING Sanal POS' with competitive commission rates.",
        },
        "expected_output": "- **POS / Virtual POS**: Detected massive credit card collection volume (108: ₺85,000,000). Confirmed active POS usage via (780. - POS Komisyon Giderleri). Strong cross-sell for **ING Fiziki POS** or **ING Sanal POS**.",
    },
    {
        "product_key": "dbs",
        "scenario": "DBS (Direct Debit) & Competitor Refinancing",
        "signal_keys": ["DBS (Direct Debit System)"],
        "sector_affinity": ["Trading", "Manufacturing", "Distribution", "Ticaret", "Üretim"],
        "input_signals": {
            "120 - Alıcılar": {"balance": 30000000, "volume": 60000000},
            "System Explicit Keyword Matches": ["120.04.050 - B Bankası DBS Alacakları"],
        },
        "reasoning_process": {
            "IF": "120 volume is high AND the system flagged 'DBS' in sub-accounts",
            "THINKING": "Company has a B2B dealer network and the system found proof they use a competitor's DBS.",
            "ACTION": "Highlight the confirmed sub-account to target competitor wallet share.",
            "PROPOSAL": "Offer 'ING DGÖS' to refinance competitor collections.",
        },
        "expected_output": "- **Collection/DBS**: Massive B2B collection volume. System confirms existing competitor usage (120.04.050 - B Bankası DBS Alacakları). Prime target for **ING DGÖS**.",
    },
    {
        "product_key": "insurance",
        "scenario": "Fleet Assets & Confirmed Insurance Cross-Sell",
        "signal_keys": ["Fleet Assets (254)", "Insurance Expenses (730/760/770)"],
        "sector_affinity": ["Manufacturing", "Logistics", "Construction", "Transportation", "Üretim", "İnşaat"],
        "input_signals": {
            "254 - Taşıtlar": {"balance": 18000000, "volume": 2000000},
            "System Explicit Keyword Matches": ["770.03.005 - Araç Kasko Giderleri"],
        },
        "reasoning_process": {
            "IF": "254 > 0 AND system extracted 'KASKO/SİGORTA' from expense accounts",
            "THINKING": "Company owns a fleet and Python confirmed they actively pay insurance premiums.",
            "ACTION": "Use the extracted expense account to size the opportunity.",
            "PROPOSAL": "Cross-sell 'ING Kasko / Filo Sigortası'.",
        },
        "expected_output": "- **Insurance / Fleet**: Vehicle assets (254: ₺18,000,000) with confirmed insurance payments (770.03.005 - Araç Kasko Giderleri). Refer to ING Insurance for **ING Kasko / Filo Sigortası**.",
    },
    {
        "product_key": "checks",
        "scenario": "Check Products — Received & Issued Checks",
        "signal_keys": ["Received Checks (101)", "Issued Checks (103)"],
        "sector_affinity": ["Trading", "Manufacturing", "Construction", "Ticaret", "Üretim", "İnşaat"],
        "input_signals": {
            "101 - Alınan Çekler": {"balance": 5000000, "volume": 40000000},
            "103 - Verilen Çekler": {"balance": 3000000, "volume": 25000000},
        },
        "reasoning_process": {
            "IF": "101 Volume > 0 OR 103 Volume > 0",
            "THINKING": "Company actively uses checks for both collections and payments. High volume indicates B2B trade dependency on check instruments.",
            "ACTION": "Size the check portfolio and propose ING check financing products.",
            "PROPOSAL": "Offer 'ING Çek Karnesi', 'ING Çek İskontosu/İştira' for received checks, check guarantee for issued checks.",
        },
        "expected_output": "- **Check Products**: Active check usage — Received (101: Vol ₺40M, Bal ₺5M), Issued (103: Vol ₺25M, Bal ₺3M). Propose **ING Çek İskontosu** for receivables financing and **ING Çek Karnesi**.",
    },
    {
        "product_key": "trade_finance",
        "scenario": "Trade Finance & Export Revenue (Sector: Export/Manufacturing)",
        "signal_keys": ["Export Revenue (601)", "Trade Finance Signals (159/340)"],
        "sector_affinity": ["Export", "Import", "Manufacturing", "İhracat", "İthalat", "Üretim"],
        "input_signals": {
            "601 - Yurtdışı Satışlar": {"balance": 0, "volume": 50000000},
            "System Explicit Keyword Matches": ["159.02 - İthalat Avansları", "780.05 - Akreditif Komisyonları"],
        },
        "reasoning_process": {
            "IF": "601 Volume > 0 AND system flagged 'İTHALAT/AKREDİTİF' keywords",
            "THINKING": "Company is an active exporter/importer. Export revenue confirms international trade. System found LC commission expenses proving they use trade finance elsewhere.",
            "ACTION": "Cite export revenue and LC expenses as proof of trade finance need.",
            "PROPOSAL": "Offer 'ING Akreditif', 'ING İhracat Faktoring', 'ING Döviz Kredisi'.",
        },
        "expected_output": "- **Trade Finance**: Active exporter (601: ₺50M). Confirmed LC usage (780.05 - Akreditif Komisyonları). Propose **ING Akreditif**, **ING İhracat Faktoring**, and **ING Döviz Kredisi**.",
    },
    {
        "product_key": "fx",
        "scenario": "FX Activity & Currency Hedging",
        "signal_keys": ["FX Net Impact (646/656)", "SWIFT/Transfer Expenses"],
        "sector_affinity": ["Export", "Import", "Manufacturing", "Energy", "İhracat", "İthalat"],
        "input_signals": {
            "646 - Kambiyo Karları": {"balance": 0, "volume": 8000000},
            "656 - Kambiyo Zararları": {"balance": 0, "volume": 12000000},
        },
        "reasoning_process": {
            "IF": "646 + 656 total volume > 0",
            "THINKING": "Company has significant FX exposure. Net FX loss (656 > 646) indicates unhedged currency risk.",
            "ACTION": "Calculate net FX impact and recommend hedging products.",
            "PROPOSAL": "Offer 'ING e-Turuncu Kur' for FX trading, 'ING Forward/Opsiyon' for hedging.",
        },
        "expected_output": "- **FX & Hedging**: Significant FX exposure (646: ₺8M gains, 656: ₺12M losses = Net ₺4M loss). Unhedged risk. Propose **ING e-Turuncu Kur** and **ING Forward/Opsiyon** for currency hedging.",
    },
    {
        "product_key": "payroll",
        "scenario": "Payroll & Personnel (Sector-aware: Manufacturing/Services)",
        "signal_keys": ["Payroll & Personnel"],
        "sector_affinity": ["Manufacturing", "Services", "Construction", "Üretim", "Hizmet", "İnşaat"],
        "input_signals": {
            "720/730/760/770 - Personnel Expenses": {"balance": 0, "volume": 15000000},
        },
        "reasoning_process": {
            "IF": "Personnel expense volume > 0",
            "THINKING": "Company has significant payroll. Manufacturing/services firms with large workforce = payroll banking opportunity.",
            "ACTION": "Estimate employee count from average salary and propose payroll package.",
            "PROPOSAL": "Offer 'ING Maaş Ödemesi Paketi', employee banking cross-sell.",
        },
        "expected_output": "- **Payroll**: ₺15M personnel expenses. Estimate ~200 employees. Propose **ING Maaş Ödemesi Paketi** with employee banking cross-sell (ING Turuncu Hesap).",
    },
    {
        "product_key": "credit_card",
        "scenario": "Corporate Credit Card Signal",
        "signal_keys": ["Corporate Credit Card"],
        "sector_affinity": ["Trading", "Services", "Retail", "Ticaret", "Hizmet", "Perakende"],
        "input_signals": {
            "309 - Diğer Mali Borçlar": {"balance": 2000000, "volume": 10000000},
            "System Explicit Keyword Matches": ["309.01 - Şirket Kredi Kartı Borçları"],
        },
        "reasoning_process": {
            "IF": "309 balance > 0 AND system flagged 'KREDİ KARTI' keywords",
            "THINKING": "Company uses corporate credit cards actively with competitor banks.",
            "ACTION": "Cite the flagged sub-account as proof of competitor card usage.",
            "PROPOSAL": "Offer 'ING Bonus Business Kart' with competitive limits and cashback.",
        },
        "expected_output": "- **Corporate Credit Card**: Active card usage (309.01 - Şirket Kredi Kartı Borçları: ₺2M balance, ₺10M volume). Propose **ING Bonus Business Kart** with competitive limits.",
    },
    {
        "product_key": "supplier_finance",
        "scenario": "Supplier Finance (TFS/SCF) — Manufacturing Sector",
        "signal_keys": ["Supplier Finance (TFS/SCF)"],
        "sector_affinity": ["Manufacturing", "Trading", "Automotive", "Üretim", "Ticaret"],
        "input_signals": {
            "320 - Satıcılar": {"balance": 20000000, "volume": 80000000},
            "System Explicit Keyword Matches": ["320.05 - TFS Borçları"],
        },
        "reasoning_process": {
            "IF": "320 volume is high AND system flagged 'TFS/TEDARİKÇİ' keywords AND sector is Manufacturing/Trading",
            "THINKING": "Company has large supplier payables and already uses supply chain finance. Sector confirms strong upstream dependency.",
            "ACTION": "Size the SCF opportunity from trade payables volume.",
            "PROPOSAL": "Offer 'ING Tedarik Zinciri Finansmanı' to capture supplier payments.",
        },
        "expected_output": "- **Supplier Finance (TFS)**: Large supplier payables (320: ₺80M volume). Confirmed TFS usage (320.05 - TFS Borçları). Manufacturing sector = strong SCF fit. Propose **ING Tedarik Zinciri Finansmanı**.",
    },
    {
        "product_key": "deposit",
        "scenario": "Deposit Capture Opportunity",
        "signal_keys": ["Bank Transaction Volume (102)"],
        "sector_affinity": ["Trading", "Services", "Manufacturing", "Retail"],
        "input_signals": {
            "102 - Bankalar": {"balance": 25000000, "volume": 150000000},
            "ING Not Present in 102 sub-accounts": True,
        },
        "reasoning_process": {
            "IF": "102 balance > 0 AND ING is NOT present in 102 sub-accounts",
            "THINKING": "Company holds significant deposits at competitor banks. ING has zero share of deposits — this is a greenfield opportunity.",
            "ACTION": "Flag as priority deposit capture target.",
            "PROPOSAL": "Offer 'ING Turuncu Vadesiz', 'ING e-Turuncu Mevduat' with competitive rates to capture deposit flow.",
        },
        "expected_output": "- **Deposit Capture**: ₺25M deposits at competitors, ING has 0% share. Priority acquisition target. Propose **ING Turuncu Vadesiz** and **ING e-Turuncu Mevduat** with competitive rates.",
    },
    # ── NEW SCENARIOS ──
    {
        "product_key": "leasing",
        "scenario": "Leasing — CAPEX-Heavy Machinery Park & Competitor Leasing Refinance",
        "signal_keys": ["Machinery & Equipment (253)", "Leasing Payables (301/401)"],
        "sector_affinity": ["Manufacturing", "Construction", "Logistics", "Üretim", "İnşaat"],
        "input_signals": {
            "253 - Tesis, Makine ve Cihazlar": {"balance": 45000000, "volume": 12000000},
            "301 - Finansal Kiralama İşlemlerinden Borçlar": {"balance": 6000000, "volume": 9000000},
        },
        "reasoning_process": {
            "IF": "253 balance is high (capex-heavy asset base) AND/OR 301/401 balances > 0 (existing leasing with competitor)",
            "THINKING": "Company runs a machinery-intensive operation. Recent 253 volume signals ongoing investment; existing 301/401 leasing payables prove they already finance equipment via a competitor lessor.",
            "ACTION": "Size the equipment park from 253 balance and cite 301/401 sub-accounts as competitor-leasing evidence.",
            "PROPOSAL": "Offer 'ING Leasing (Finansal Kiralama)' for new machinery investment and refinancing of competitor lease contracts; consider sale-and-leaseback for liquidity.",
        },
        "expected_output": "- **Leasing**: Machinery park of ₺45M (253) with active investment flow (₺12M) and existing competitor leasing debt (301: ₺6M). Propose **ING Leasing** for new capex and lease refinancing / sale-and-leaseback.",
    },
    {
        "product_key": "letter_of_guarantee",
        "scenario": "Letter of Guarantee & Progress Payment Finance (Construction)",
        "signal_keys": ["Construction Costs (170)", "Progress Billings (350)", "Guarantee Letter Commissions"],
        "sector_affinity": ["Construction", "İnşaat", "Energy", "Infrastructure"],
        "input_signals": {
            "170 - Yıllara Yaygın İnşaat Maliyetleri": {"balance": 30000000, "volume": 30000000},
            "350 - Hakediş Bedelleri": {"balance": 25000000, "volume": 25000000},
            "System Explicit Keyword Matches": ["780.07 - Teminat Mektubu Komisyonları"],
        },
        "reasoning_process": {
            "IF": "170 or 350 balances > 0 (multi-year contract work) AND/OR system flagged 'TEMİNAT' commission expenses",
            "THINKING": "Company executes long-term contracts requiring bid/performance bonds. Guarantee commission expenses prove they obtain letters of guarantee from competitor banks.",
            "ACTION": "Cite 170/350 balances to size the contract pipeline and the flagged commission account as competitor-usage proof.",
            "PROPOSAL": "Offer 'ING Teminat Mektubu' (non-cash limit), surety bonds for tenders, and progress payment (hakediş) finance against 350.",
        },
        "expected_output": "- **Letter of Guarantee / Non-Cash**: Active multi-year contracts (170: ₺30M, 350: ₺25M) with confirmed guarantee commissions (780.07). Propose **ING Teminat Mektubu** limit and **Hakediş Finansmanı**. Ask RM to prepare a non-cash credit limit analysis.",
    },
    {
        "product_key": "factoring",
        "scenario": "Factoring — Notes Receivable & Long Collection Cycle",
        "signal_keys": ["Notes Receivable (121)"],
        "sector_affinity": ["Trading", "Manufacturing", "Textile", "Ticaret", "Üretim"],
        "input_signals": {
            "121 - Alacak Senetleri": {"balance": 12000000, "volume": 35000000},
        },
        "reasoning_process": {
            "IF": "121 volume > 0 (active promissory note portfolio), especially when collection period is long",
            "THINKING": "Company sells on term against notes. A rotating note portfolio of this size ties up working capital and carries collection risk.",
            "ACTION": "Size the discountable portfolio from 121 volume and balance.",
            "PROPOSAL": "Offer 'ING Faktoring' (with/without recourse) and 'Senet İskontosu' to convert the note portfolio into immediate liquidity.",
        },
        "expected_output": "- **Factoring**: Promissory note portfolio (121: Bal ₺12M, Vol ₺35M) locking working capital. Propose **ING Faktoring** and **Senet İskontosu** for receivables financing.",
    },
    {
        "product_key": "cash_loan",
        "scenario": "Working Capital Loan & Competitor Loan Refinancing",
        "signal_keys": ["Bank Loans (300 + 400)", "Total Financial Expenses (780)"],
        "sector_affinity": ["Manufacturing", "Trading", "Construction", "Üretim", "Ticaret", "İnşaat"],
        "input_signals": {
            "300 - Banka Kredileri (KV)": {"balance": 35000000, "volume": 90000000},
            "780 - Finansman Giderleri": {"balance": 0, "volume": 9000000},
        },
        "reasoning_process": {
            "IF": "300/400 volume is high (frequent loan rollovers at competitors) AND 780 expense burden is significant relative to revenue",
            "THINKING": "Company continuously rolls short-term debt at competitor banks and pays heavy financing costs — a price-led refinancing window for ING.",
            "ACTION": "Cite competitor sub-accounts under 300/400 and the 780 expense as the cost-saving argument.",
            "PROPOSAL": "Offer 'ING Spot Kredi' / 'BCH Rotatif' refinancing. Instruct RM: 'Prepare a credit limit analysis' — do not state a limit.",
        },
        "expected_output": "- **Working Capital / Refinancing**: ₺90M ST loan turnover at competitors (300) with ₺9M financing cost (780). Strong refinancing target. Propose **ING Spot Kredi / BCH Rotatif**; RM to prepare a credit limit analysis.",
    },
    {
        "product_key": "cash_management",
        "scenario": "Cash Management — Fragmented Multi-Bank Transaction Traffic",
        "signal_keys": ["Bank Transaction Volume (102)", "SWIFT/Transfer Expenses"],
        "sector_affinity": ["Services", "Trading", "Retail", "Hizmet", "Ticaret", "Perakende"],
        "input_signals": {
            "102 - Bankalar": {"balance": 8000000, "volume": 250000000},
            "System Explicit Keyword Matches": ["780.09 - EFT/Havale Komisyon Giderleri"],
        },
        "reasoning_process": {
            "IF": "102 volume is very high relative to balance (heavy payment/collection traffic) AND transfer commission expenses flagged",
            "THINKING": "Company routes massive transaction traffic across several banks and pays per-transaction fees. Consolidating flows is a fee-income and deposit-float opportunity for ING.",
            "ACTION": "Quantify the 102 turnover and cite EFT/havale commission sub-accounts.",
            "PROPOSAL": "Offer 'ING Nakit Yönetimi' (bulk payments/collections, API banking), salary+vendor payment bundling, and 'ING e-Turuncu Mevduat' for idle float.",
        },
        "expected_output": "- **Cash Management**: ₺250M annual bank transaction volume (102) vs only ₺8M balance, with confirmed EFT/havale fees (780.09). Propose **ING Nakit Yönetimi** bundle to consolidate flows and capture float.",
    },
    {
        "product_key": "virtual_pos",
        "scenario": "E-Commerce / Virtual POS — Online Sales Channel",
        # Triggers ONLY on explicit e-commerce keyword evidence (600/649 sub-accounts);
        # plain 108 card-collection volume belongs to the physical "pos" scenario.
        "signal_keys": ["E-Commerce Revenue"],
        "sector_affinity": ["Retail", "Trading", "Services", "Perakende", "Ticaret", "E-Commerce"],
        "input_signals": {
            "600.03 - E-Ticaret / Pazaryeri Satışları": {"balance": 0, "volume": 30000000},
            "108 - Diğer Hazır Değerler": {"balance": 1500000, "volume": 28000000},
        },
        "reasoning_process": {
            "IF": "system flagged 'E-TİCARET/PAZARYERİ/ONLINE' keywords in 600/649 sub-accounts AND 108 collection volume confirms card-based settlement",
            "THINKING": "Company runs an online sales channel settled via card schemes/marketplaces — virtual POS and marketplace collection products apply.",
            "ACTION": "Cite the flagged e-commerce revenue sub-account and 108 settlement volume.",
            "PROPOSAL": "Offer 'ING Sanal POS' with competitive blended rates and marketplace/PSP collection solutions.",
        },
        "expected_output": "- **Virtual POS / E-Commerce**: Online channel revenue (600.03: ₺30M) settled through card collections (108: ₺28M). Propose **ING Sanal POS** and marketplace collection solutions.",
    },
]

# Quick lookup by product_key
SCENARIOS_BY_KEY = {sc["product_key"]: sc for sc in FEW_SHOT_SCENARIOS}

# ══════════════════════════════════════════════════════════════
# RECOMMENDATION TEMPLATES — deterministic Client Need / ING product
# wording per product_key, used to build the PRODUCT RECOMMENDATION
# CATALOG that stabilizes the strategist's matrix (fixed membership,
# fixed order — the LLM only polishes the Reasoning wording).
# ══════════════════════════════════════════════════════════════
RECOMMENDATION_TEMPLATES = {
    "pos": {
        "client_need": "Card-based collection infrastructure",
        "ing_products": ["ING Fiziki POS", "ING Sanal POS"],
    },
    "virtual_pos": {
        "client_need": "E-commerce / online collection",
        "ing_products": ["ING Sanal POS", "Marketplace collection solutions"],
    },
    "dbs": {
        "client_need": "B2B dealer collection automation",
        "ing_products": ["ING DGÖS (Direct Debit System)"],
    },
    "supplier_finance": {
        "client_need": "Supplier payment financing",
        "ing_products": ["ING Tedarik Zinciri Finansmanı"],
    },
    "credit_card": {
        "client_need": "Corporate purchasing & expense management",
        "ing_products": ["ING Bonus Business Kart"],
    },
    "insurance": {
        "client_need": "Fleet & asset insurance coverage",
        "ing_products": ["ING Kasko / Filo Sigortası"],
    },
    "checks": {
        "client_need": "Check-based collection & payment financing",
        "ing_products": ["ING Çek İskontosu", "ING Çek Karnesi"],
    },
    "trade_finance": {
        "client_need": "Cross-border trade financing",
        "ing_products": ["ING Akreditif", "ING İhracat Faktoring", "ING Döviz Kredisi"],
    },
    "fx": {
        "client_need": "FX risk management & hedging",
        "ing_products": ["ING e-Turuncu Kur", "ING Forward/Opsiyon"],
    },
    "payroll": {
        "client_need": "Payroll & employee banking",
        "ing_products": ["ING Maaş Ödemesi Paketi"],
    },
    "deposit": {
        "client_need": "Deposit & liquidity placement",
        "ing_products": ["ING Turuncu Vadesiz", "ING e-Turuncu Mevduat"],
    },
    "leasing": {
        "client_need": "Equipment & machinery financing",
        "ing_products": ["ING Leasing (Finansal Kiralama)"],
    },
    "letter_of_guarantee": {
        "client_need": "Non-cash guarantee limits & progress payment finance",
        "ing_products": ["ING Teminat Mektubu", "Hakediş Finansmanı"],
    },
    "factoring": {
        "client_need": "Receivables (notes) financing",
        "ing_products": ["ING Faktoring", "Senet İskontosu"],
    },
    "cash_loan": {
        "client_need": "Working capital & loan refinancing",
        "ing_products": ["ING Spot Kredi", "BCH Rotatif"],
    },
    "cash_management": {
        "client_need": "Payment/collection consolidation & float capture",
        "ing_products": ["ING Nakit Yönetimi"],
    },
}

# Human-readable product names used in exclusion messages
PRODUCT_DISPLAY_NAMES = {
    "pos": "POS (Physical)",
    "virtual_pos": "Virtual POS / E-Commerce",
    "dbs": "DBS (Direct Debit System)",
    "supplier_finance": "Supplier Finance (TFS/SCF)",
    "credit_card": "Corporate Credit Card",
    "insurance": "Fleet/Asset Insurance",
    "checks": "Check Products",
    "trade_finance": "Trade Finance / Letter of Credit",
    "fx": "FX & Hedging",
    "payroll": "Payroll Package",
    "deposit": "Deposit Products",
    "leasing": "Leasing",
    "letter_of_guarantee": "Letter of Guarantee (Non-Cash)",
    "factoring": "Factoring / Note Discounting",
    "cash_loan": "Cash Loans (Spot/Revolving)",
    "cash_management": "Cash Management",
}


def _signal_strength(product_signals: dict, signal_keys: list) -> tuple:
    """Return (is_active, total_strength) for the given signal keys."""
    active = False
    total = 0.0
    for key in signal_keys:
        data = product_signals.get(key)
        if not isinstance(data, dict):
            continue
        vol = abs(float(data.get("volume") or 0))
        bal = abs(float(data.get("balance") or 0))
        if vol > 0 or bal > 0:
            active = True
        total += vol + bal
    return active, total


def _norm(text: str) -> str:
    """Lowercase + strip Turkish diacritics so 'Üretim' matches 'uretim'."""
    s = (text or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _sector_tokens(sector: str) -> list:
    """
    Split a (possibly multi-sector) prediction into normalized tokens.
    'Retail + Export' → ['retail', 'export'];
    'Textile Manufacturing + Export' → ['textile manufacturing', 'export'].
    """
    return [t.strip() for t in re.split(r"[+/,;]", _norm(sector)) if t.strip()]


def _sector_matches(sector: str, affinities: list) -> bool:
    """Multi-sector aware: any token of the prediction vs any part of any affinity."""
    tokens = _sector_tokens(sector)
    if not tokens:
        return False
    for affinity in affinities:
        parts = [p.strip() for p in re.split(r"[+/,;]", _norm(affinity)) if p.strip()]
        for part in parts:
            if any(part in tok or tok in part for tok in tokens):
                return True
    return False


def classify_product_opportunities(
    product_signals: dict,
    sector: str = "General",
    product_flags: dict = None,
    max_scenarios: int = 6,
) -> dict:
    """
    Classify which few-shot scenarios should be injected into prompts.

    Selection rules:
    - A scenario is a candidate only if at least one of its mapped
      product_signal keys has non-zero volume or balance (data evidence).
    - Scenarios whose product is already used by the customer
      (local DB flag == 1) are EXCLUDED and reported separately.
    - Candidates are ranked by signal strength (volume+balance), with a
      1.5x boost when the company sector matches the scenario's affinity.
    - At most `max_scenarios` are selected (token budget control).

    Returns dict with keys:
      selected            -> list of scenario dicts to inject
      selected_keys       -> list of product_keys selected
      excluded_existing   -> list of {product_key, product_name, reason}
      inactive_keys       -> product_keys with no signal evidence
    """
    product_flags = product_flags or {}
    candidates = []
    excluded_existing = []
    inactive_keys = []

    for sc in FEW_SHOT_SCENARIOS:
        key = sc["product_key"]
        active, strength = _signal_strength(product_signals or {}, sc["signal_keys"])

        flag_val = product_flags.get(key)
        try:
            already_used = int(flag_val) == 1
        except (TypeError, ValueError):
            already_used = False

        if already_used:
            excluded_existing.append({
                "product_key": key,
                "product_name": PRODUCT_DISPLAY_NAMES.get(key, key),
                "reason": "Customer already uses this product per bank core system (flag=1, latest data).",
            })
            continue

        if not active:
            inactive_keys.append(key)
            continue

        score = strength * (1.5 if _sector_matches(sector, sc.get("sector_affinity", [])) else 1.0)
        candidates.append((score, sc))

    candidates.sort(key=lambda t: t[0], reverse=True)
    selected = [sc for _, sc in candidates[:max_scenarios]]

    logger.info(
        f"🎯 Few-shot selection: {len(selected)} injected "
        f"({[sc['product_key'] for sc in selected]}), "
        f"{len(excluded_existing)} excluded as already-used, "
        f"{len(inactive_keys)} inactive"
    )

    return {
        "selected": selected,
        "selected_keys": [sc["product_key"] for sc in selected],
        "excluded_existing": excluded_existing,
        "inactive_keys": inactive_keys,
    }


def _render_scenario(sc: dict) -> str:
    """Compact single-scenario rendering (token-efficient)."""
    body = {
        "input_signals": sc["input_signals"],
        "reasoning_process": sc["reasoning_process"],
        "expected_output": sc["expected_output"],
    }
    return (
        f"### Scenario [{sc['product_key']}]: {sc['scenario']}\n"
        + json.dumps(body, indent=1, ensure_ascii=False)
    )


def build_few_shot_injection(classification: dict, sector: str = "General") -> dict:
    """
    Build prompt additions from a classification result.

    Returns:
      {"system_addition": str, "user_addition": str}

    - system_addition: core anti-hallucination rules + sector priority line.
      Injected into the SYSTEM prompt only when at least one scenario or
      exclusion exists.
    - user_addition: only the selected scenarios + a hard exclusion block
      for currently-used products. Injected into the USER prompt.
    """
    selected = classification.get("selected", [])
    excluded = classification.get("excluded_existing", [])

    if not selected and not excluded:
        return {"system_addition": "", "user_addition": ""}

    # ── System prompt addition ──
    rules = "\n".join(f"- {r}" for r in CORE_MIZAN_RULES)
    # Multi-sector aware: a company can match several priority entries
    # (e.g., 'Retail + Export' → Retail AND Export priorities).
    sector_lines = []
    for sec_name, products in SECTOR_PRODUCT_PRIORITY.items():
        if _sector_matches(sector, [sec_name]):
            sector_lines.append(f"\nSECTOR PRIORITY ({sec_name}): {', '.join(products)}")
            if len(sector_lines) >= 3:
                break
    if not sector_lines:
        # FALLBACK: sector prediction absent or not covered by the
        # TCMB-aligned priority map — use the all-sector General entry.
        logger.warning(
            f"⚠️ Sector prediction absent/unmatched ('{sector}') — "
            f"no SECTOR_PRODUCT_PRIORITY match; falling back to General priorities. "
            f"Report will be generated without sector-specific product weighting."
        )
        general = SECTOR_PRODUCT_PRIORITY.get("General / Genel", [])
        sector_lines.append(
            f"\nSECTOR PRIORITY (General fallback — sector unmatched): {', '.join(general)}"
        )
    sector_line = "".join(sector_lines)
    system_addition = (
        "\n\n## MIZAN ANALYSIS CORE RULES\n"
        f"{rules}{sector_line}\n"
        "Few-shot scenarios in the user message are selected ONLY for products "
        "with detected data evidence — treat absent products accordingly."
    )

    # ── User prompt addition ──
    parts = []
    if excluded:
        lines = "\n".join(
            f"- {e['product_name']}: {e['reason']}" for e in excluded
        )
        parts.append(
            "## ⛔ CURRENTLY USED PRODUCTS — DO NOT RECOMMEND (latest bank data)\n"
            f"{lines}\n"
            "These products are ALREADY in use by this customer. You MUST NOT "
            "include them as new cross-sell recommendations in any section or table. "
            "You may only reference them as existing relationship context."
        )
    if selected:
        scenario_text = "\n\n".join(_render_scenario(sc) for sc in selected)
        parts.append(
            "## REASONING HEURISTICS — FEW-SHOT EXAMPLES (selected for detected signals)\n"
            "Internalize this logic. Rely ONLY on the data and keyword matches "
            "explicitly provided by the system. Do not guess sub-accounts.\n\n"
            f"{scenario_text}"
        )

    return {
        "system_addition": system_addition,
        "user_addition": "\n\n" + "\n\n".join(parts),
    }


# ══════════════════════════════════════════════════════════════
# PRODUCT RECOMMENDATION CATALOG — deterministic matrix membership
# ══════════════════════════════════════════════════════════════

def _build_evidence(product_signals: dict, signal_keys: list, max_accounts: int = 3) -> str:
    """Render real signal values + top sub-accounts as bulleted evidence."""
    lines = []
    for key in signal_keys:
        data = product_signals.get(key)
        if not isinstance(data, dict):
            continue
        vol = float(data.get("volume") or 0)
        bal = float(data.get("balance") or 0)
        if vol == 0 and bal == 0:
            continue
        lines.append(f"• {key}: Volume=₺{vol:,.0f}, Balance=₺{bal:,.0f}")
        for acc, vol_info in list(data.get("account_mapping", {}).items())[:max_accounts]:
            lines.append(f"  - {acc} ({vol_info})")
    return "\n".join(lines)


def build_recommendation_catalog(
    product_signals: dict,
    sector: str = "General",
    product_flags: dict = None,
    max_cross_sell: int = 4,
) -> dict:
    """
    Build the DETERMINISTIC product recommendation catalog that fixes the
    matrix membership across runs. Unlike the few-shot prompt injection
    (capped scenarios for token budget), the catalog includes:

    - "active": ONE entry per scenario with real signal evidence (no cap),
      sorted by signal strength descending. Evidence cites the actual
      balances/volumes/sub-accounts from this Mizan — never example data.
    - "cross_sell": sector-affinity products with NO signal evidence
      (capped at max_cross_sell), tagged as zero-volume opportunities.
    - "excluded_existing": products already used per the bank DB (flag=1)
      — never allowed into the matrix.

    The strategist renders the matrix 1:1 from this catalog, which makes
    the row count stable run over run.
    """
    cls = classify_product_opportunities(
        product_signals, sector=sector, product_flags=product_flags,
        max_scenarios=len(FEW_SHOT_SCENARIOS),  # no cap for the catalog
    )

    active = []
    for sc in cls["selected"]:  # already sorted by boosted signal strength
        key = sc["product_key"]
        tpl = RECOMMENDATION_TEMPLATES.get(key, {})
        _, strength = _signal_strength(product_signals or {}, sc["signal_keys"])
        active.append({
            "product_key": key,
            "signal_type": "Active Signal",
            "client_need": tpl.get("client_need", PRODUCT_DISPLAY_NAMES.get(key, key)),
            "ing_products": tpl.get("ing_products", []),
            "data_evidence": _build_evidence(product_signals or {}, sc["signal_keys"]),
            "reasoning": sc["reasoning_process"]["THINKING"],
            "signal_strength": strength,
        })

    cross_sell = []
    for key in cls["inactive_keys"]:
        sc = SCENARIOS_BY_KEY.get(key)
        if sc is None or not _sector_matches(sector, sc.get("sector_affinity", [])):
            continue
        tpl = RECOMMENDATION_TEMPLATES.get(key, {})
        cross_sell.append({
            "product_key": key,
            "signal_type": "Cross-Sell",
            "client_need": tpl.get("client_need", PRODUCT_DISPLAY_NAMES.get(key, key)),
            "ing_products": tpl.get("ing_products", []),
            "data_evidence": "• No current volume detected in Mizan — sector-priority product",
            "reasoning": (
                f"Sector-aligned greenfield opportunity for the {sector} sector; "
                f"no usage footprint in the Mizan data."
            ),
            "signal_strength": 0.0,
        })
        if len(cross_sell) >= max_cross_sell:
            break

    catalog = {
        "active": active,
        "cross_sell": cross_sell,
        "excluded_existing": cls["excluded_existing"],
        "total_rows": len(active) + len(cross_sell),
    }
    logger.info(
        f"📋 Recommendation catalog: {len(active)} active + "
        f"{len(cross_sell)} cross-sell = {catalog['total_rows']} matrix rows "
        f"({len(cls['excluded_existing'])} excluded as already-used)"
    )
    return catalog


def render_catalog_for_prompt(catalog: dict) -> dict:
    """
    Render the catalog as strategist prompt additions.
    Returns {"system_addition": str, "user_addition": str}.
    """
    entries = catalog.get("active", []) + catalog.get("cross_sell", [])
    if not entries:
        return {"system_addition": "", "user_addition": ""}
    n = len(entries)

    lines = []
    for i, e in enumerate(entries, start=1):
        evidence = e["data_evidence"].replace("\n", " <br> ") if e["data_evidence"] else "—"
        lines.append(
            f"{i}. [{e['signal_type']}] Client Need: {e['client_need']} | "
            f"Product: {', '.join(e['ing_products'])} | "
            f"Data Evidence: {evidence} | "
            f"Reasoning: {e['reasoning']}"
        )
    excluded = catalog.get("excluded_existing", [])
    excluded_line = ""
    if excluded:
        names = ", ".join(e["product_name"] for e in excluded)
        excluded_line = (
            f"\nEXCLUDED (already used per bank core data — DO NOT add as rows): {names}"
        )

    user_addition = (
        f"\n\n### 📋 PRODUCT RECOMMENDATION CATALOG (AUTHORITATIVE — {n} rows)\n"
        f"The PRODUCT RECOMMENDATION MATRIX TABLE must contain EXACTLY these {n} "
        f"rows, in this exact order:\n"
        + "\n".join(lines)
        + excluded_line
    )
    system_addition = (
        f"\n\nMATRIX STABILITY RULE (CRITICAL): The user message contains a "
        f"PRODUCT RECOMMENDATION CATALOG with exactly {n} numbered entries. "
        f"The PRODUCT RECOMMENDATION MATRIX TABLE MUST contain exactly {n} data "
        f"rows — one per catalog entry, in catalog order. Copy 'Client Need', "
        f"'Product' and 'Data Evidence' from the catalog (you may compact "
        f"formatting); you may only refine the 'Reasoning' wording. "
        f"NEVER add, drop, merge, split, or reorder rows, and NEVER include "
        f"products listed as EXCLUDED."
    )
    return {"system_addition": system_addition, "user_addition": user_addition}
