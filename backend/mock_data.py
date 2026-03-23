"""
Mock Data Generation Module
============================
Generates realistic Turkish financial data for the Financial Intelligence Platform.
- Mizan (Trial Balance) with Turkish Chart of Accounts codes
- 6-month transaction history with 120+ rows
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# ============================================================================
# CONSTANTS - Company & Counterparty Definitions
# ============================================================================

TARGET_COMPANY = "X Tekstil A.S."
TARGET_TAX_ID = "1234567890"

# Mock counterparty companies (buyers / customers)
CUSTOMERS = [
    {"code": "120.01", "name": "Anadolu Giyim Ltd.", "tax_id": "1111111111"},
    {"code": "120.02", "name": "Marmara Moda A.S.", "tax_id": "2222222222"},
    {"code": "120.03", "name": "Ege Tekstil San.", "tax_id": "3333333333"},
    {"code": "120.04", "name": "Karadeniz Konfeksiyon", "tax_id": "4444444444"},
    {"code": "120.05", "name": "Istanbul Hazir Giyim", "tax_id": "5555555555"},
]

# Mock counterparty companies (suppliers)
SUPPLIERS = [
    {"code": "320.01", "name": "Bursa Iplik A.S.", "tax_id": "6666666666"},
    {"code": "320.02", "name": "Denizli Dokuma San.", "tax_id": "7777777777"},
    {"code": "320.03", "name": "Gaziantep Boya Kim.", "tax_id": "8888888888"},
    {"code": "320.04", "name": "Adana Pamuk Tic.", "tax_id": "9999999999"},
    {"code": "320.05", "name": "Kayseri Aksesuar Ltd.", "tax_id": "1010101010"},
]

# Mock banks
BANKS = [
    {"code": "102.01", "name": "Garanti BBVA"},
    {"code": "102.02", "name": "Is Bankasi"},
    {"code": "102.03", "name": "Yapi Kredi"},
]


def get_mizan_df() -> pd.DataFrame:
    """
    Generate a mock Mizan (Trial Balance) DataFrame.
    
    Uses standard Turkish Uniform Chart of Accounts (Tekdüzen Hesap Planı).
    Account ranges:
      - 1xx: Current Assets
      - 2xx: Non-Current Assets
      - 3xx: Short-Term Liabilities
      - 4xx: Long-Term Liabilities
      - 5xx: Equity
      - 6xx: Revenue
      - 7xx: Operating & Financial Expenses
    
    Returns:
        pd.DataFrame with columns: account_code, account_name, debit, credit
    """
    
    rows = []
    
    # ---- 100: Cash (Kasa) ----
    rows.append({
        "account_code": "100",
        "account_name": "Kasa (Cash)",
        "debit": 125_000.00,
        "credit": 0.00,
    })
    
    # ---- 102: Banks (Bankalar) - Sub-accounts per bank ----
    bank_balances = [
        ("102.01", "Bankalar - Garanti BBVA", 1_450_000.00, 0.00),
        ("102.02", "Bankalar - Is Bankasi", 870_000.00, 0.00),
        ("102.03", "Bankalar - Yapi Kredi", 320_000.00, 0.00),
    ]
    for code, name, debit, credit in bank_balances:
        rows.append({
            "account_code": code,
            "account_name": name,
            "debit": debit,
            "credit": credit,
        })
    
    # ---- 120: Trade Receivables (Alicilar) - Sub-accounts per customer ----
    receivable_amounts = [780_000, 560_000, 430_000, 290_000, 185_000]
    for i, cust in enumerate(CUSTOMERS):
        rows.append({
            "account_code": cust["code"],
            "account_name": f"Alicilar - {cust['name']}",
            "debit": float(receivable_amounts[i]),
            "credit": 0.00,
        })
    
    # ---- 153: Inventory (Ticari Mallar) ----
    rows.append({
        "account_code": "153",
        "account_name": "Ticari Mallar (Inventory)",
        "debit": 2_100_000.00,
        "credit": 0.00,
    })
    
    # ---- 255: Fixed Assets (Demirbaslar) ----
    rows.append({
        "account_code": "255",
        "account_name": "Demirbaslar (Fixed Assets)",
        "debit": 950_000.00,
        "credit": 0.00,
    })
    
    # ---- 257: Accumulated Depreciation ----
    rows.append({
        "account_code": "257",
        "account_name": "Birikmis Amortismanlar (-)",
        "debit": 0.00,
        "credit": 380_000.00,
    })
    
    # ---- 300: Short-Term Bank Loans ----
    rows.append({
        "account_code": "300",
        "account_name": "Banka Kredileri (Short-Term)",
        "debit": 0.00,
        "credit": 1_200_000.00,
    })
    
    # ---- 320: Trade Payables (Saticilar) - Sub-accounts per supplier ----
    payable_amounts = [620_000, 480_000, 350_000, 270_000, 140_000]
    for i, supp in enumerate(SUPPLIERS):
        rows.append({
            "account_code": supp["code"],
            "account_name": f"Saticilar - {supp['name']}",
            "debit": 0.00,
            "credit": float(payable_amounts[i]),
        })
    
    # ---- 360: Tax Payables ----
    rows.append({
        "account_code": "360",
        "account_name": "Odenecek Vergi ve Fonlar",
        "debit": 0.00,
        "credit": 185_000.00,
    })
    
    # ---- 400: Long-Term Bank Loans ----
    rows.append({
        "account_code": "400",
        "account_name": "Banka Kredileri (Long-Term)",
        "debit": 0.00,
        "credit": 750_000.00,
    })
    
    # ---- 500: Equity (Sermaye) ----
    rows.append({
        "account_code": "500",
        "account_name": "Sermaye (Equity Capital)",
        "debit": 0.00,
        "credit": 2_000_000.00,
    })
    
    # ---- 570: Retained Earnings ----
    rows.append({
        "account_code": "570",
        "account_name": "Gecmis Yillar Karlari",
        "debit": 0.00,
        "credit": 450_000.00,
    })
    
    # ---- 600: Gross Sales (Yurt Ici Satislar) ----
    rows.append({
        "account_code": "600",
        "account_name": "Yurt Ici Satislar (Domestic Sales)",
        "debit": 0.00,
        "credit": 12_500_000.00,
    })
    
    # ---- 610: Sales Returns ----
    rows.append({
        "account_code": "610",
        "account_name": "Satis Iadeleri (-)",
        "debit": 350_000.00,
        "credit": 0.00,
    })
    
    # ---- 620: Cost of Goods Sold (Satilan Malin Maliyeti) ----
    rows.append({
        "account_code": "620",
        "account_name": "Satilan Malin Maliyeti (COGS)",
        "debit": 8_750_000.00,
        "credit": 0.00,
    })
    
    # ---- 630: General Administrative Expenses ----
    rows.append({
        "account_code": "630",
        "account_name": "Genel Yonetim Giderleri",
        "debit": 1_100_000.00,
        "credit": 0.00,
    })
    
    # ---- 760: Marketing & Selling Expenses ----
    rows.append({
        "account_code": "760",
        "account_name": "Pazarlama Satis Dagitim Giderleri",
        "debit": 680_000.00,
        "credit": 0.00,
    })
    
    # ---- 780: Financial Expenses (Finansman Giderleri) ----
    # Includes POS commission costs and bank interest
    rows.append({
        "account_code": "780",
        "account_name": "Finansman Giderleri (Financial Expenses)",
        "debit": 420_000.00,
        "credit": 0.00,
    })
    
    # ---- 780.01: POS Commission Sub-account ----
    rows.append({
        "account_code": "780.01",
        "account_name": "POS Komisyon Giderleri",
        "debit": 185_000.00,
        "credit": 0.00,
    })
    
    # ---- 780.02: Bank Interest Expense ----
    rows.append({
        "account_code": "780.02",
        "account_name": "Banka Faiz Giderleri",
        "debit": 235_000.00,
        "credit": 0.00,
    })
    
    df = pd.DataFrame(rows)
    return df


def get_transactions_df() -> pd.DataFrame:
    """
    Generate mock transaction history for the past 6 months.
    
    Creates 120+ realistic transaction rows covering:
    - POS collections from customers
    - Supplier invoice payments
    - Bank loan repayments
    - Salary payments
    - Tax payments
    - Utility bills
    - Raw material purchases
    
    Returns:
        pd.DataFrame with columns: Transaction_ID, Date, Amount, Type,
                                   Counterparty_Name, Description
    """
    
    random.seed(42)
    np.random.seed(42)
    
    transactions = []
    tx_id = 1000
    
    # Date range: past 6 months
    end_date = datetime(2026, 2, 23)
    start_date = end_date - timedelta(days=180)
    
    def rand_date():
        """Generate a random date within the 6-month window."""
        delta = (end_date - start_date).days
        return start_date + timedelta(days=random.randint(0, delta))
    
    # --- POS Collections (Incoming) from customers ---
    # ~30 transactions: POS card collections
    for _ in range(30):
        cust = random.choice(CUSTOMERS)
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(15_000, 120_000), 2),
            "Type": "Incoming",
            "Counterparty_Name": cust["name"],
            "Description": "POS collection - credit card payment",
        })
    
    # --- Invoice Payments from customers (Bank Transfer) ---
    # ~15 transactions
    for _ in range(15):
        cust = random.choice(CUSTOMERS)
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(50_000, 200_000), 2),
            "Type": "Incoming",
            "Counterparty_Name": cust["name"],
            "Description": "Invoice payment - bank transfer",
        })
    
    # --- Supplier Payments (Outgoing) ---
    # ~25 transactions
    for _ in range(25):
        supp = random.choice(SUPPLIERS)
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(30_000, 180_000), 2),
            "Type": "Outgoing",
            "Counterparty_Name": supp["name"],
            "Description": "Supplier invoice payment",
        })
    
    # --- Bank Loan Repayments (Outgoing) ---
    # ~12 transactions (2 per month)
    for month_offset in range(6):
        for bank in random.sample(BANKS, 2):
            tx_id += 1
            payment_date = start_date + timedelta(days=30 * month_offset + random.randint(1, 5))
            transactions.append({
                "Transaction_ID": f"TX-{tx_id}",
                "Date": payment_date,
                "Amount": round(random.uniform(40_000, 95_000), 2),
                "Type": "Outgoing",
                "Counterparty_Name": bank["name"],
                "Description": "Loan repayment - monthly installment",
            })
    
    # --- Salary Payments (Outgoing) ---
    # ~6 transactions (monthly)
    for month_offset in range(6):
        tx_id += 1
        pay_date = start_date + timedelta(days=30 * month_offset + 25)
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": pay_date,
            "Amount": round(random.uniform(280_000, 350_000), 2),
            "Type": "Outgoing",
            "Counterparty_Name": "Personel Maas Odemesi",
            "Description": "Monthly salary payment",
        })
    
    # --- Tax Payments (Outgoing) ---
    # ~6 transactions
    for month_offset in range(6):
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": start_date + timedelta(days=30 * month_offset + 20),
            "Amount": round(random.uniform(25_000, 60_000), 2),
            "Type": "Outgoing",
            "Counterparty_Name": "Vergi Dairesi",
            "Description": "Tax payment - VAT / withholding",
        })
    
    # --- Utility & Rent Payments (Outgoing) ---
    # ~12 transactions
    utilities = ["Elektrik Faturasi", "Dogalgaz Faturasi", "Su Faturasi", "Kira Odemesi"]
    for month_offset in range(6):
        for util in random.sample(utilities, 2):
            tx_id += 1
            transactions.append({
                "Transaction_ID": f"TX-{tx_id}",
                "Date": start_date + timedelta(days=30 * month_offset + random.randint(10, 18)),
                "Amount": round(random.uniform(8_000, 45_000), 2),
                "Type": "Outgoing",
                "Counterparty_Name": util,
                "Description": f"Utility/Rent payment - {util}",
            })
    
    # --- Competitor Bank Incoming Transfers (to detect multi-banking) ---
    # ~10 transactions from competitor banks
    competitor_banks = ["Akbank", "Ziraat Bankasi", "Halkbank"]
    for _ in range(10):
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(20_000, 150_000), 2),
            "Type": "Incoming",
            "Counterparty_Name": random.choice(competitor_banks),
            "Description": "Transfer from competitor bank account",
        })
    
    # --- Raw Material Purchases (Outgoing) ---
    # ~8 transactions
    for _ in range(8):
        supp = random.choice(SUPPLIERS)
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(80_000, 250_000), 2),
            "Type": "Outgoing",
            "Counterparty_Name": supp["name"],
            "Description": "Raw material purchase - bulk order",
        })
    
    df = pd.DataFrame(transactions)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df


# ============================================================================
# Quick test when run directly
# ============================================================================
if __name__ == "__main__":
    mizan = get_mizan_df()
    txns = get_transactions_df()
    print(f"Mizan rows: {len(mizan)}")
    print(mizan.to_string(index=False))
    print(f"\nTransactions rows: {len(txns)}")
    print(txns.head(20).to_string(index=False))
