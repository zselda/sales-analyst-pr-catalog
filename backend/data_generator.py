"""
Context-Aware Transaction Generator
======================================
Generates synthetic but consistent transactions based on entities
extracted from the uploaded Mizan Excel file.
 
Consistency rules:
  - Incoming payments: payer names come from the 120 (Customer) list
  - Outgoing payments: receiver names come from the 320 (Supplier) list
  - Operational expenses (Rent, Tax, Salaries) are independent
"""
 
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
 
import logging
 
logger = logging.getLogger("swarm.data_generator")
 
 
def generate_transactions(
    customers: list[dict],
    suppliers: list[dict],
    num_rows: int = 200,
    months: int = 6,
) -> pd.DataFrame:
    """
    Generate context-aware synthetic transactions.
 
    Args:
        customers: List of customer dicts (from extract_entities). Each has 'name', 'balance'.
        suppliers: List of supplier dicts (from extract_entities). Each has 'name', 'balance'.
        num_rows: Target number of transactions (~200).
        months: Number of months to cover (6).
 
    Returns:
        pd.DataFrame with columns:
            Transaction_ID, Date, Amount, Type, Counterparty_Name, Description
    """
    random.seed(42)
    np.random.seed(42)
 
    transactions = []
    tx_id = 1000
 
    end_date = datetime(2026, 2, 27)
    start_date = end_date - timedelta(days=30 * months)
 
    def rand_date():
        delta = (end_date - start_date).days
        return start_date + timedelta(days=random.randint(0, delta))
 
    # Fallback if lists are empty
    if not customers:
        customers = [{"name": "Generic Customer", "balance": 100_000}]
    if not suppliers:
        suppliers = [{"name": "Generic Supplier", "balance": 80_000}]
 
    # Weight customers/suppliers by balance for realistic distribution
    cust_weights = [max(c.get("balance", 1), 1) for c in customers]
    supp_weights = [max(s.get("balance", 1), 1) for s in suppliers]
 
    def pick_customer():
        return random.choices(customers, weights=cust_weights, k=1)[0]["name"]
 
    def pick_supplier():
        return random.choices(suppliers, weights=supp_weights, k=1)[0]["name"]
 
    # --- 1. POS Collections (Incoming) from customers ---
    for _ in range(35):
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(15_000, 150_000), 2),
            "Type": "Incoming",
            "Counterparty_Name": pick_customer(),
            "Description": "POS collection - credit card payment",
        })
 
    # --- 2. Invoice Payments from customers (Bank Transfer) ---
    for _ in range(20):
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(50_000, 300_000), 2),
            "Type": "Incoming",
            "Counterparty_Name": pick_customer(),
            "Description": "Invoice payment - bank transfer",
        })
 
    # --- 3. Supplier Payments (Outgoing) ---
    for _ in range(30):
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(30_000, 250_000), 2),
            "Type": "Outgoing",
            "Counterparty_Name": pick_supplier(),
            "Description": "Supplier invoice payment",
        })
 
    # --- 4. Raw Material Purchases (Outgoing) ---
    for _ in range(20):
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(80_000, 350_000), 2),
            "Type": "Outgoing",
            "Counterparty_Name": pick_supplier(),
            "Description": "Raw material purchase - bulk order",
        })
 
    # --- 5. Bank Loan Repayments (Outgoing) ---
    banks = ["Garanti BBVA", "İş Bankası", "Yapı Kredi", "Akbank", "Ziraat Bankası"]
    for month_offset in range(months):
        for bank in random.sample(banks[:3], 2):
            tx_id += 1
            payment_date = start_date + timedelta(days=30 * month_offset + random.randint(1, 5))
            transactions.append({
                "Transaction_ID": f"TX-{tx_id}",
                "Date": payment_date,
                "Amount": round(random.uniform(40_000, 120_000), 2),
                "Type": "Outgoing",
                "Counterparty_Name": bank,
                "Description": "Loan repayment - monthly installment",
            })
 
    # --- 6. Salary Payments (Outgoing) ---
    for month_offset in range(months):
        tx_id += 1
        pay_date = start_date + timedelta(days=30 * month_offset + 25)
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": pay_date,
            "Amount": round(random.uniform(350_000, 550_000), 2),
            "Type": "Outgoing",
            "Counterparty_Name": "Personel Maaş Ödemesi",
            "Description": "Monthly salary payment",
        })
 
    # --- 7. Tax Payments (Outgoing) ---
    for month_offset in range(months):
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": start_date + timedelta(days=30 * month_offset + 20),
            "Amount": round(random.uniform(30_000, 80_000), 2),
            "Type": "Outgoing",
            "Counterparty_Name": "Vergi Dairesi",
            "Description": "Tax payment - VAT / withholding",
        })
 
    # --- 8. Utility & Rent Payments (Outgoing) ---
    utilities = ["Elektrik Faturası", "Doğalgaz Faturası", "Su Faturası", "Kira Ödemesi"]
    for month_offset in range(months):
        for util in random.sample(utilities, 2):
            tx_id += 1
            transactions.append({
                "Transaction_ID": f"TX-{tx_id}",
                "Date": start_date + timedelta(days=30 * month_offset + random.randint(10, 18)),
                "Amount": round(random.uniform(10_000, 60_000), 2),
                "Type": "Outgoing",
                "Counterparty_Name": util,
                "Description": f"Utility/Rent payment - {util}",
            })
 
    # --- 9. Competitor Bank Incoming Transfers ---
    competitor_banks = ["Akbank", "Ziraat Bankası", "Halkbank"]
    for _ in range(12):
        tx_id += 1
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(25_000, 200_000), 2),
            "Type": "Incoming",
            "Counterparty_Name": random.choice(competitor_banks),
            "Description": "Transfer from competitor bank account",
        })
 
    # --- 10. Additional customer payments to reach ~200 ---
    remaining = max(0, num_rows - len(transactions))
    for _ in range(remaining):
        tx_id += 1
        is_incoming = random.random() < 0.6
        transactions.append({
            "Transaction_ID": f"TX-{tx_id}",
            "Date": rand_date(),
            "Amount": round(random.uniform(20_000, 180_000), 2),
            "Type": "Incoming" if is_incoming else "Outgoing",
            "Counterparty_Name": pick_customer() if is_incoming else pick_supplier(),
            "Description": "Invoice payment" if is_incoming else "Material/service purchase",
        })
 
    df = pd.DataFrame(transactions)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
 
    logger.info(f"Generated {len(df)} transactions over {months} months")
    logger.info(f"  Incoming: {len(df[df['Type'] == 'Incoming'])}")
    logger.info(f"  Outgoing: {len(df[df['Type'] == 'Outgoing'])}")
 
    return df
