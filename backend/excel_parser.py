"""
Excel Parser Module
=====================
Parses uploaded Mizan (Trial Balance) Excel files (.xlsx).
Extracts entities (customers, suppliers) from 120/320 account codes.
Assigns random 10-digit Tax IDs (VKN) to each extracted entity.
Extracts company name from filename and predicts sector via LLM.
"""

import re
import pandas as pd
import random
import logging

logger = logging.getLogger("swarm.excel_parser")

# Column name mappings: Turkish → internal English names
COLUMN_MAP = {
    "Hesap Kodu": "account_code",
    "Hesap Adı": "account_name",
    "Borç": "debit",
    "Alacak": "credit",
    "Bakiye Borç": "balance_debit",
    "Bakiye Alacak": "balance_credit",
    "Dönem": "donem",
    "Donem": "donem",
}


def parse_mizan_excel(file_bytes: bytes) -> pd.DataFrame:
    """
    Parse an uploaded Mizan Excel file into a standardized DataFrame.

    Args:
        file_bytes: Raw bytes of the .xlsx file.

    Returns:
        pd.DataFrame with columns:
            account_code, account_name, debit, credit, balance_debit, balance_credit
    """
    from io import BytesIO

    df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")

    # Try to match columns by Turkish names
    rename_map = {}
    for turkish, english in COLUMN_MAP.items():
        for col in df.columns:
            if turkish.lower() in str(col).lower():
                rename_map[col] = english
                break

    if rename_map:
        df = df.rename(columns=rename_map)
    else:
        # Fallback: assume columns are already in English or positional
        expected = ["account_code", "account_name", "debit", "credit", "balance_debit", "balance_credit"]
        if len(df.columns) >= 6:
            df.columns = expected[:len(df.columns)]

    # Ensure account_code is string (preserve leading zeros / dots)
    df["account_code"] = df["account_code"].astype(str).str.strip()

    # Drop rows where account_code is NaN/empty/None
    df = df[df["account_code"].notna() & (df["account_code"] != "") & (df["account_code"] != "nan")]

    # Fill NaN numeric values with 0
    for col in ["debit", "credit", "balance_debit", "balance_credit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Drop header-like rows that got mixed in (e.g. row where account_code == "Hesap Kodu")
    df = df[~df["account_code"].str.contains("Hesap", case=False, na=False)]

    df = df.reset_index(drop=True)
    logger.info(f"Parsed {len(df)} accounts from Excel")
    return df


def extract_donem(df: pd.DataFrame) -> dict:
    """
    Extract and parse the Dönem (time period) from a Mizan DataFrame.

    The Dönem column uses YYYYMM format:
        202412 → covers all 12 months of 2024 (360 days)
        202503 → covers first 3 months of 2025 (90 days)

    Uses 360-day year convention (month × 30) per Turkish banking standards.

    Returns:
        dict with keys: raw, year, month, period_months, period_days, label
    Falls back to full-year (360 days) if column is missing or unparseable.
    """
    DEFAULT = {
        "raw": "unknown",
        "year": None,
        "month": 12,
        "period_months": 12,
        "period_days": 360,
        "label": "Annual (12M) — default, no Dönem column found",
    }

    if "donem" not in df.columns:
        logger.warning("No 'Dönem' column found in Mizan — defaulting to full-year (360 days)")
        return DEFAULT

    # Get the first non-null Dönem value
    donem_series = df["donem"].dropna()
    if donem_series.empty:
        logger.warning("Dönem column is empty — defaulting to full-year (360 days)")
        return DEFAULT

    raw_value = str(donem_series.iloc[0]).strip()

    # Parse YYYYMM
    try:
        # Handle potential float conversion (e.g., 202503.0)
        raw_value = str(int(float(raw_value)))

        if len(raw_value) != 6:
            raise ValueError(f"Dönem value '{raw_value}' is not 6 digits (YYYYMM)")

        year = int(raw_value[:4])
        month = int(raw_value[4:6])

        if not (1 <= month <= 12):
            raise ValueError(f"Dönem month '{month}' out of range [1-12]")
        if not (2000 <= year <= 2099):
            raise ValueError(f"Dönem year '{year}' out of expected range [2000-2099]")

        period_months = month
        period_days = month * 30  # 360-day year convention

        # Human-readable label
        if period_months == 12:
            label = f"Annual (12M, {year})"
        elif period_months % 3 == 0:
            quarter = period_months // 3
            label = f"Q{quarter} ({period_months}M, {year})"
        else:
            label = f"{period_months}M ({year})"

        result = {
            "raw": raw_value,
            "year": year,
            "month": month,
            "period_months": period_months,
            "period_days": period_days,
            "label": label,
        }
        logger.info(f"Parsed Dönem: {raw_value} → {label} ({period_days} days)")
        return result

    except (ValueError, TypeError) as e:
        logger.warning(f"Failed to parse Dönem '{raw_value}': {e} — defaulting to full-year (360 days)")
        return DEFAULT


def _generate_vkn() -> str:
    """Generate a random 10-digit Turkish Tax ID (VKN)."""
    return "".join([str(random.randint(0, 9)) for _ in range(10)])


def extract_entities(mizan_df: pd.DataFrame) -> dict:
    """
    Extract customers (120.xxx) and suppliers (320.xxx) from Mizan data.
    Assigns a random VKN to each unique entity.

    Args:
        mizan_df: Parsed Mizan DataFrame.

    Returns:
        dict with keys:
            - customers: list of {name, account_code, tax_id, balance}
            - suppliers: list of {name, account_code, tax_id, balance}
    """
    random.seed(42)  # Reproducible VKNs for consistency

    def _extract_group(prefix: str, balance_col: str, entity_type: str) -> list:
        """Extract leaf-level entities for a given account prefix."""
        mask = mizan_df["account_code"].str.startswith(prefix)
        group_df = mizan_df[mask].copy()

        if group_df.empty:
            return []

        # Filter to leaf accounts only (longest codes — those with no children)
        # A leaf is an account where no other account code starts with it
        all_codes = group_df["account_code"].tolist()
        leaf_codes = []
        for code in all_codes:
            is_parent = any(
                other.startswith(code) and len(other) > len(code)
                for other in all_codes
            )
            if not is_parent:
                leaf_codes.append(code)

        leaf_df = group_df[group_df["account_code"].isin(leaf_codes)]

        entities = []
        for _, row in leaf_df.iterrows():
            balance = float(row.get(balance_col, 0))
            if balance == 0:
                # Try alternative: debit for customers, credit for suppliers
                if entity_type == "customer":
                    balance = float(row.get("debit", 0))
                else:
                    balance = float(row.get("credit", 0))

            entities.append({
                "name": str(row["account_name"]).strip(),
                "account_code": str(row["account_code"]),
                "tax_id": _generate_vkn(),
                "balance": balance,
                "type": entity_type,
            })

        return entities

    customers = _extract_group("120", "balance_debit", "customer")
    suppliers = _extract_group("320", "balance_credit", "supplier")

    logger.info(f"Extracted {len(customers)} customers, {len(suppliers)} suppliers")

    return {
        "customers": customers,
        "suppliers": suppliers,
    }


def extract_company_name(filename: str) -> str:
    """
    Extract company name from the uploaded Mizan filename.

    Examples:
        mizan_ABC_Holding.xlsx      → ABC Holding
        mizan_best.xlsx             → Best
        ABC_Tekstil_mizan.xlsx      → ABC Tekstil
        my_company_report.xlsx      → My Company Report
    """
    # Remove extension
    name = re.sub(r'\.xlsx?$', '', filename, flags=re.IGNORECASE).strip()

    # Remove 'mizan' prefix/suffix (case-insensitive)
    name = re.sub(r'^mizan[_\-\s]*', '', name, flags=re.IGNORECASE).strip()
    name = re.sub(r'[_\-\s]*mizan$', '', name, flags=re.IGNORECASE).strip()

    # Replace underscores, hyphens with spaces
    name = re.sub(r'[_\-]+', ' ', name).strip()

    if not name:
        return "Company"

    # Title-case each word
    return name.title()


def predict_sector(customers: list, suppliers: list) -> str:
    """
    Use the LLM to predict the corporate sector from customer/supplier names.

    Returns a short sector string like "Textile", "Food & Beverage", "Automotive", etc.
    Falls back to "General" if LLM is unavailable.
    """
    try:
        from llm_config import invoke_llm
    except ImportError:
        logger.warning("LLM not available for sector prediction")
        return "General"

    # Gather top entity names for context
    cust_names = [c["name"] for c in customers[:10]]
    supp_names = [s["name"] for s in suppliers[:10]]

    entity_text = ""
    if cust_names:
        entity_text += f"Customers: {', '.join(cust_names)}\n"
    if supp_names:
        entity_text += f"Suppliers: {', '.join(supp_names)}\n"

    if not entity_text:
        return "General"

    system_prompt = (
        "You are a business analyst. Given a list of customer and supplier names, "
        "predict the primary industry/sector of the company. "
        "Reply with ONLY the sector name in 1-3 words (e.g. 'Textile', 'Food & Beverage', "
        "'Automotive', 'Construction', 'Retail', 'Technology', 'Healthcare'). "
        "Do not include any explanation."
    )

    prompt = f"Based on these business partners, what sector does this company operate in?\n\n{entity_text}"

    try:
        sector = invoke_llm(system_prompt, prompt, temperature=0.1, max_tokens=256)
        # Clean: take first line, strip quotes/punctuation
        sector = sector.strip().split("\n")[0].strip().strip('"').strip("'").strip(".")
        if len(sector) > 40:
            sector = sector[:40]
        logger.info(f"Predicted sector: {sector}")
        return sector or "General"
    except Exception as e:
        logger.warning(f"Sector prediction failed: {e}")
        return "General"


# ──────────────────────────────────────────────────────────────
# Transaction File Parsing
# ──────────────────────────────────────────────────────────────

# Column name mappings for transactions: Turkish / common variants → internal
TXN_COLUMN_MAP = {
    # Date
    "tarih": "Date", "date": "Date", "işlem tarihi": "Date",
    "transaction_date": "Date", "transaction date": "Date",
    # Amount
    "tutar": "Amount", "amount": "Amount", "miktar": "Amount",
    "işlem tutarı": "Amount", "transaction_amount": "Amount",
    # Type
    "tür": "Type", "type": "Type", "tip": "Type", "yön": "Type",
    "işlem türü": "Type", "transaction_type": "Type",
    # Counterparty
    "karşı taraf": "Counterparty_Name", "counterparty": "Counterparty_Name",
    "counterparty_name": "Counterparty_Name", "müşteri": "Counterparty_Name",
    "firma": "Counterparty_Name", "unvan": "Counterparty_Name",
    "alıcı/gönderen": "Counterparty_Name", "counterparty name": "Counterparty_Name",
    # Description
    "açıklama": "Description", "description": "Description",
    "not": "Description", "detay": "Description",
    # Transaction ID
    "işlem no": "Transaction_ID", "transaction_id": "Transaction_ID",
    "referans": "Transaction_ID", "ref": "Transaction_ID",
}

# Type value normalization: Turkish / common → Incoming / Outgoing
TYPE_INCOMING = {"incoming", "gelen", "alacak", "giriş", "gelir", "tahsilat", "in", "credit"}
TYPE_OUTGOING = {"outgoing", "giden", "borç", "çıkış", "gider", "ödeme", "out", "debit"}


def parse_transactions_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Parse an uploaded transaction file (.xlsx or .csv) into a standardized DataFrame.

    Flexibly maps column names (Turkish/English) to the expected format:
        Date, Amount, Type (Incoming/Outgoing), Counterparty_Name, Description, Transaction_ID

    Args:
        file_bytes: Raw bytes of the file.
        filename: Original filename (used to detect format).

    Returns:
        pd.DataFrame with standardized columns.
    """
    from io import BytesIO

    fname_lower = filename.lower()
    if fname_lower.endswith(".csv"):
        # Try UTF-8 first, then common Turkish encoding
        try:
            df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(file_bytes), encoding="latin-1")
    elif fname_lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(BytesIO(file_bytes), engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file format: {filename}. Use .xlsx or .csv")

    # Map columns
    rename_map = {}
    for col in df.columns:
        col_lower = str(col).lower().strip()
        if col_lower in TXN_COLUMN_MAP:
            rename_map[col] = TXN_COLUMN_MAP[col_lower]

    if rename_map:
        df = df.rename(columns=rename_map)

    # Ensure required columns exist
    required = ["Date", "Amount"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'. Available: {list(df.columns)}")

    # Parse dates
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])

    # Parse amounts
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0).abs()

    # Normalize Type column
    if "Type" in df.columns:
        df["Type"] = df["Type"].astype(str).str.lower().str.strip()
        df["Type"] = df["Type"].apply(
            lambda x: "Incoming" if x in TYPE_INCOMING else ("Outgoing" if x in TYPE_OUTGOING else "Outgoing")
        )
    else:
        # Infer from sign if Amount had negatives before abs()
        df["Type"] = "Outgoing"
        logger.warning("No 'Type' column found — defaulting all transactions to 'Outgoing'")

    # Fill optional columns
    if "Counterparty_Name" not in df.columns:
        df["Counterparty_Name"] = "Unknown"
    if "Description" not in df.columns:
        df["Description"] = ""
    if "Transaction_ID" not in df.columns:
        df["Transaction_ID"] = [f"TX-{i+1}" for i in range(len(df))]

    # Clean strings
    df["Counterparty_Name"] = df["Counterparty_Name"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()

    df = df.sort_values("Date").reset_index(drop=True)

    incoming = len(df[df["Type"] == "Incoming"])
    outgoing = len(df[df["Type"] == "Outgoing"])
    logger.info(f"Parsed {len(df)} transactions ({incoming} incoming, {outgoing} outgoing)")

    return df
