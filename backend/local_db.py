"""
Local Customer Database Layer (SQLite dev backend + Oracle EDW backend)
========================================================================
Stores / reads the bank's own latest view of each customer:
1. customer_financials — financial metrics in LONG format: one row per
   metric with columns (tax_id, period, tr_description, value). The
   tr_description values are the Turkish metric names used by the bank's
   core system (e.g., 'Asit Test Oranı', 'Cari Oran', 'VAFÖK Marjı (%)').
   After every SQL read the DataFrame is preprocessed with
   add_english_descriptions(), which appends an `en_description` column
   via TR_EN_METRIC_MAP so the (English) LLM pipeline can consume it.
2. customer_products  — current product usage flags
   (e.g., pos=1, credit_card=0, checks=1) plus the customer's NACE code.

BACKENDS (selected via CUSTOMER_DB_BACKEND env var):
- "sqlite" (default): local file at data/customer_db.sqlite — used for
  development, tests, and demo runs. Supports writes + demo seeding.
- "oracle": the bank's EDW via SQLAlchemy + python-oracledb, using the
  standard connection pattern (credentials file + init_oracle_client +
  DESCRIPTION DSN). READ-ONLY — warehouse rows are managed by ETL, so
  upserts/seeding raise. On query failure the readers return EMPTY
  frames (never demo data) and the pipeline degrades to Mizan-only
  evidence.

Oracle configuration (env vars, with bank defaults):
  CUSTOMER_DB_BACKEND       sqlite | oracle
  ORACLE_CREDENTIALS_PATH   /home/athena/credentials.txt (username\\npassword)
  ORACLE_DSN                (DESCRIPTION=...EDWDB:4525...edwprd)
  ORACLE_FINANCIALS_TABLE   CUSTOMER_FINANCIALS
  ORACLE_PRODUCTS_TABLE     CUSTOMER_PRODUCTS

The db_enrichment agent reads this data and exposes it to the pipeline
as DataFrames + plain dictionaries so quant_analyst / product_analyst /
strategist prompts are grounded in the LATEST bank-side data.

CLI:
    python local_db.py --seed     # create sqlite DB and load demo rows
    python local_db.py --show     # dump current contents (any backend)
"""

import argparse
import logging
import os
import re
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger("swarm.local_db")

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "customer_db.sqlite"

# ══════════════════════════════════════════════════════════════
# BACKEND CONFIGURATION
# ══════════════════════════════════════════════════════════════
ORACLE_CREDENTIALS_PATH = os.environ.get(
    "ORACLE_CREDENTIALS_PATH", "/home/athena/credentials.txt"
)
ORACLE_DSN = os.environ.get(
    "ORACLE_DSN",
    "(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=EDWDB)(PORT=4525))"
    "(CONNECT_DATA=(SERVICE_NAME=edwprd)))",
)
ORACLE_FINANCIALS_TABLE = os.environ.get("ORACLE_FINANCIALS_TABLE", "CUSTOMER_FINANCIALS")
ORACLE_PRODUCTS_TABLE = os.environ.get("ORACLE_PRODUCTS_TABLE", "CUSTOMER_PRODUCTS")

_oracle_engine = None


def get_backend() -> str:
    """Active backend: 'sqlite' (default) or 'oracle'. Read per-call so
    tests and deployments can switch via the environment."""
    return os.environ.get("CUSTOMER_DB_BACKEND", "sqlite").strip().lower()


def _get_oracle_engine():
    """
    Lazy singleton SQLAlchemy engine for the bank EDW, built with the
    standard bank connection pattern:

        with open('/home/athena/credentials.txt') as f:
            credentials = f.read().split()
        oracledb.init_oracle_client()
        engine = sa.create_engine('oracle+oracledb://user:pass@(DESCRIPTION=...)')

    sqlalchemy/oracledb are imported here (not module-level) so the
    pipeline stays importable on machines without the Oracle client.
    """
    global _oracle_engine
    if _oracle_engine is None:
        import sqlalchemy as sa
        import oracledb

        with open(ORACLE_CREDENTIALS_PATH) as f:
            credentials = f.read().split()
        db_username = credentials[0]
        db_password = credentials[1]
        oracledb.init_oracle_client()
        _oracle_engine = sa.create_engine(
            "oracle+oracledb://" + db_username + ":" + db_password + "@" + ORACLE_DSN
        )
        logger.info("[DB] Oracle EDW engine created (DSN host EDWDB, service edwprd)")
    return _oracle_engine


def _read_oracle(query: str, params: dict = None) -> pd.DataFrame:
    """Run a read query against the EDW; lowercase column names so the
    rest of the pipeline (tr_description, value, ...) works unchanged —
    Oracle returns UPPERCASE identifiers by default."""
    import sqlalchemy as sa

    engine = _get_oracle_engine()
    with engine.connect() as connection:
        df = pd.read_sql(sa.text(query), con=connection, params=params or {})
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _assert_writable():
    """Writes are only supported on the sqlite dev backend."""
    if get_backend() == "oracle":
        raise NotImplementedError(
            "Oracle EDW backend is READ-ONLY for this pipeline — "
            "customer_financials / customer_products rows are maintained by "
            "the warehouse ETL. Use CUSTOMER_DB_BACKEND=sqlite for local writes."
        )

# Product flag columns. Keys MUST match few_shot_library product_keys.
PRODUCT_FLAG_COLUMNS = [
    "pos", "virtual_pos", "credit_card", "checks", "dbs",
    "supplier_finance", "leasing", "factoring", "trade_finance",
    "fx", "payroll", "insurance", "deposit", "cash_loan",
    "cash_management", "letter_of_guarantee",
]

# ══════════════════════════════════════════════════════════════
# TURKISH → ENGLISH METRIC MAPPING
# Keys are the bank core system's tr_description values (normalized via
# _norm_metric before lookup, so trailing/double spaces in the source
# data — e.g. 'Nakit  Döngüsü (Gün)', 'Brüt Faiz ' — are tolerated).
# ══════════════════════════════════════════════════════════════
TR_EN_METRIC_MAP = {
    "Asit Test Oranı": "Acid-Test Ratio (Quick Ratio)",
    "Net Satışlar / İşletme Sermayesi": "Net Sales / Working Capital",
    "VAFÖK/Finasman Gideri": "EBITDA / Financial Expenses",   # source typo kept
    "VFÖK/Finansman Gideri": "EBIT / Financial Expenses",
    "Faaliyetlerinden elde edilen nakit/Toplam Net Borç": "Cash Flow from Operations (CFO) / Total Net Debt",
    "Serbest Nakit Akımı/Toplam Net Borç": "Free Cash Flow / Total Net Debt",
    "VAFÖK / KV Finansal Borç": "EBITDA / Short-Term Financial Debt",
    "Faaliyetlerden elde edilen nakit (CFO)/Toplam KV Finansal Borç": "CFO / Total Short-Term Financial Debt",
    "Free Cash Flow": "Free Cash Flow",
    "Brüt Kar marjı (%)": "Gross Profit Margin (%)",
    "VAFÖK Marjı (%)": "EBITDA Margin (%)",
    "Net Faaliyet (FVÖK) Karı (%)": "Net Operating Profit (EBIT) Margin (%)",
    "Vergi öncesi kar marjı (%)": "Pre-Tax Profit Margin (%)",
    "Net kar marjı (%)": "Net Profit Margin (%)",
    "Temettü Ödeme Oranı (%)": "Dividend Payout Ratio (%)",
    "Vergi Öncesi Kar / Maddi Özsermaye": "Pre-Tax Profit / Tangible Net Worth",
    "Vergi Öncesi Kar / Özkaynak": "Pre-Tax Profit / Equity",
    "Aktif Devir Hızı": "Asset Turnover",
    "Net Kar/Maddi Özsermaye": "Net Profit / Tangible Net Worth",
    "Özkaynak Devir Hızı": "Equity Turnover",
    "Faaliyetlerden Elde Edilen Nakit (İşletme Sermayesinden önce)": "Cash Flow from Operations (before Working Capital)",
    "Alacak Devir Süresi (Gün)": "Receivables Collection Period (Days)",
    "Stok Devir Süresi (Gün)": "Inventory Period (Days)",
    "Borç Devir Süresi (Gün)": "Payables Period (Days)",
    "Nakit Döngüsü (Gün)": "Cash Conversion Cycle (Days)",
    "Takipteki Alacaklar (NPL)": "Non-Performing Loans (NPL)",
    "Takipteki Alacaklar / VAFÖK": "NPL / EBITDA",
    "Takipteki Alacaklar (NPL)/Özkaynak": "NPL / Equity",
    "Net Satış / Toplam Aktif": "Net Sales / Total Assets",
    "Net Satışlar / Maddi Özsermaye": "Net Sales / Tangible Net Worth",
    "Aktif Büyüme Oranı": "Asset Growth Rate",
    "Net Satışlar Büyüme Oranı": "Net Sales Growth Rate",
    "VAFÖK Büyüme Oranı": "EBITDA Growth Rate",
    "FVÖK Büyüme Oranı": "EBIT Growth Rate",
    "Net Kar Büyüme Oranı": "Net Profit Growth Rate",
    "Takipteki Alacaklar (NPL) Büyüme Oranı": "NPL Growth Rate",
    "KV Finansal Borç": "Short-Term Financial Debt",
    "Uzun Vadeli Banka Kredilerin Kısa Vadeye Düşen Kısmı": "Current Portion of Long-Term Bank Loans",
    "UV Finansal Borç": "Long-Term Financial Debt",
    "Faizli Borç Toplamı (IBD)": "Total Interest-Bearing Debt (IBD)",
    "Toplam Net Borç": "Total Net Debt",
    "Kısa Vadeli Faizli Borç": "Short-Term Interest-Bearing Debt",
    "Uzun Vadeli Faizli Borç": "Long-Term Interest-Bearing Debt",
    "Toplam Öncelikli Faizli Borç": "Total Senior Interest-Bearing Debt",
    "Toplam Net Faizli Borç": "Total Net Interest-Bearing Debt",
    "Brüt Faiz": "Gross Interest",
    "Maddi Özsermaye": "Tangible Net Worth",
    "Toplam Net Borç / Özkaynak": "Total Net Debt / Equity",
    "Toplam Net Borç / Maddi Özsermaye": "Total Net Debt / Tangible Net Worth",
    "Toplam Net Borç / VAFÖK": "Total Net Debt / EBITDA",
    "Toplam Net Borç / (Toplam Net Borç + Özkaynak)": "Total Net Debt / (Total Net Debt + Equity)",
    "Toplam Finansal Borç / Toplam Yükümlülükler": "Total Financial Debt / Total Liabilities",
    "Toplam Finansal Borç / VAFÖK": "Total Financial Debt / EBITDA",
    "Borç Servisi Karşılama Oranı (DSCR)": "Debt Service Coverage Ratio (DSCR)",
    "Toplam Yükümlülük / Toplam Aktif": "Total Liabilities / Total Assets",
    "Finansal Kaldıraç": "Financial Leverage",
    "Özkaynak / Toplam Aktif (%)": "Equity / Total Assets (%)",
    "Toplam Aktif / Özkaynak": "Total Assets / Equity",
    "İşletme Sermayesi": "Working Capital",
    "Cari Oran": "Current Ratio",
    "Toplam Borç Servisi": "Total Debt Service",
    "Toplam Finansal Borç / Net Satışlar": "Total Financial Debt / Net Sales",
    "Düzeltilmiş ödenmiş sermaye": "Adjusted Paid-In Capital",
    "Vergi Öncesi Kar / Toplam Aktif": "Pre-Tax Profit / Total Assets",
}


def _norm_metric(text: str) -> str:
    """Normalize a tr_description for lookup: strip + collapse whitespace."""
    return re.sub(r"\s+", " ", str(text or "")).strip()


# Normalized-key lookup built once at import
_TR_EN_NORMALIZED = {_norm_metric(k): v for k, v in TR_EN_METRIC_MAP.items()}


def add_english_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a customer_financials DataFrame fetched via SQL:
    append an `en_description` column mapped from `tr_description`
    (whitespace-tolerant). Unmapped Turkish names fall back to the
    original tr_description and are logged once per call.
    """
    if df is None or df.empty or "tr_description" not in df.columns:
        return df
    df = df.copy()
    normalized = df["tr_description"].map(_norm_metric)
    df["en_description"] = normalized.map(_TR_EN_NORMALIZED)
    unmapped = sorted(normalized[df["en_description"].isna()].unique())
    if unmapped:
        logger.warning(
            f"⚠️ {len(unmapped)} tr_description value(s) missing from "
            f"TR_EN_METRIC_MAP (kept as Turkish): {unmapped}"
        )
        df["en_description"] = df["en_description"].fillna(df["tr_description"])
    return df


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS customer_financials (
    tax_id          TEXT NOT NULL,
    company_name    TEXT,
    period          TEXT NOT NULL,
    tr_description  TEXT NOT NULL,
    value           REAL,
    updated_at      TEXT DEFAULT (date('now')),
    PRIMARY KEY (tax_id, period, tr_description)
);

CREATE TABLE IF NOT EXISTS customer_products (
    tax_id          TEXT NOT NULL PRIMARY KEY,
    company_name    TEXT,
    nace_code       TEXT,
    {', '.join(f'{c} INTEGER DEFAULT 0' for c in PRODUCT_FLAG_COLUMNS)},
    updated_at      TEXT DEFAULT (date('now'))
);
"""


def _migrate(conn: sqlite3.Connection):
    """Schema migrations (idempotent)."""
    # customer_products: nace_code column added after initial deployments
    cols = {row[1] for row in conn.execute("PRAGMA table_info(customer_products)")}
    if cols and "nace_code" not in cols:
        conn.execute("ALTER TABLE customer_products ADD COLUMN nace_code TEXT")
    # customer_financials: legacy WIDE schema (one column per metric) is
    # replaced by the LONG (tr_description, value) schema. Demo-only data,
    # so the old table is dropped and recreated.
    fin_cols = {row[1] for row in conn.execute("PRAGMA table_info(customer_financials)")}
    if fin_cols and "tr_description" not in fin_cols:
        logger.warning(
            "Migrating customer_financials from legacy wide schema to "
            "long (tr_description, value) schema — old rows dropped"
        )
        conn.execute("DROP TABLE customer_financials")
        conn.executescript(_SCHEMA)


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def upsert_financials(tax_id: str, period: str, metrics: dict,
                      company_name: str = None, db_path: Path = DB_PATH):
    """
    Insert or update financial metric rows for (tax_id, period).
    `metrics` maps tr_description → value, e.g. {'Asit Test Oranı': 0.95}.
    sqlite backend only — the Oracle EDW is read-only.
    """
    _assert_writable()
    rows = [
        (tax_id, company_name, period, _norm_metric(tr), float(val))
        for tr, val in metrics.items() if val is not None
    ]
    with get_connection(db_path) as conn:
        conn.executemany(
            "INSERT INTO customer_financials "
            "(tax_id, company_name, period, tr_description, value) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(tax_id, period, tr_description) DO UPDATE SET "
            "value=excluded.value, company_name=excluded.company_name",
            rows,
        )


def upsert_product_flags(tax_id: str, flags: dict, company_name: str = None,
                         nace_code: str = None, db_path: Path = DB_PATH):
    """Insert or update product usage flags (and NACE code) for a customer.
    sqlite backend only — the Oracle EDW is read-only."""
    _assert_writable()
    cols = [c for c in PRODUCT_FLAG_COLUMNS if c in flags]
    with get_connection(db_path) as conn:
        conn.execute(
            f"INSERT INTO customer_products (tax_id, company_name, nace_code, {', '.join(cols)}) "
            f"VALUES (?, ?, ?, {', '.join('?' for _ in cols)}) "
            f"ON CONFLICT(tax_id) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in cols + ["company_name", "nace_code"]),
            [tax_id, company_name, nace_code] + [int(flags[c]) for c in cols],
        )


def get_financials_df(tax_id: str = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    """
    Financial metric rows in LONG format — columns: tax_id, company_name,
    period, tr_description, value — preprocessed with an additional
    `en_description` column (English metric names) so the downstream
    English-language pipeline can consume the data directly.
    Newest period first. Backend-dispatched (sqlite dev / Oracle EDW).
    """
    if get_backend() == "oracle":
        try:
            if tax_id:
                df = _read_oracle(
                    f"SELECT * FROM {ORACLE_FINANCIALS_TABLE} "
                    f"WHERE tax_id = :tax_id "
                    f"ORDER BY period DESC, tr_description",
                    {"tax_id": tax_id},
                )
            else:
                df = _read_oracle(
                    f"SELECT * FROM {ORACLE_FINANCIALS_TABLE} "
                    f"ORDER BY tax_id, period DESC, tr_description"
                )
        except Exception as e:
            logger.error(
                f"[DB] Oracle financials query failed: {e} — "
                f"returning empty frame (no demo fallback on EDW backend)"
            )
            return pd.DataFrame()
        return add_english_descriptions(df)

    with get_connection(db_path) as conn:
        if tax_id:
            df = pd.read_sql_query(
                "SELECT * FROM customer_financials WHERE tax_id = ? "
                "ORDER BY period DESC, tr_description",
                conn, params=[tax_id],
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM customer_financials "
                "ORDER BY tax_id, period DESC, tr_description", conn
            )
    return add_english_descriptions(df)


def get_product_flags_df(tax_id: str = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    if get_backend() == "oracle":
        try:
            if tax_id:
                return _read_oracle(
                    f"SELECT * FROM {ORACLE_PRODUCTS_TABLE} WHERE tax_id = :tax_id",
                    {"tax_id": tax_id},
                )
            return _read_oracle(f"SELECT * FROM {ORACLE_PRODUCTS_TABLE}")
        except Exception as e:
            logger.error(
                f"[DB] Oracle product flags query failed: {e} — "
                f"returning empty frame (no demo fallback on EDW backend)"
            )
            return pd.DataFrame()

    with get_connection(db_path) as conn:
        if tax_id:
            return pd.read_sql_query(
                "SELECT * FROM customer_products WHERE tax_id = ?", conn, params=[tax_id]
            )
        return pd.read_sql_query("SELECT * FROM customer_products", conn)


def _lookup_by_company_name(table: str, company_name: str, conn) -> pd.DataFrame:
    """sqlite fuzzy lookup when tax_id is unknown (e.g., standalone runs)."""
    if not company_name:
        return pd.DataFrame()
    return pd.read_sql_query(
        f"SELECT * FROM {table} WHERE UPPER(company_name) LIKE ?",
        conn, params=[f"%{company_name.strip().upper()}%"],
    )


def _financials_by_company(company_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Backend-dispatched company-name fallback for financial metrics."""
    if not company_name:
        return pd.DataFrame()
    if get_backend() == "oracle":
        try:
            df = _read_oracle(
                f"SELECT * FROM {ORACLE_FINANCIALS_TABLE} "
                f"WHERE UPPER(company_name) LIKE :pattern "
                f"ORDER BY period DESC, tr_description",
                {"pattern": f"%{company_name.strip().upper()}%"},
            )
        except Exception as e:
            logger.error(f"[DB] Oracle company-name financials lookup failed: {e}")
            return pd.DataFrame()
        return add_english_descriptions(df)
    with get_connection(db_path) as conn:
        return add_english_descriptions(
            _lookup_by_company_name("customer_financials", company_name, conn)
        )


def _products_by_company(company_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Backend-dispatched company-name fallback for product flags."""
    if not company_name:
        return pd.DataFrame()
    if get_backend() == "oracle":
        try:
            return _read_oracle(
                f"SELECT * FROM {ORACLE_PRODUCTS_TABLE} "
                f"WHERE UPPER(company_name) LIKE :pattern",
                {"pattern": f"%{company_name.strip().upper()}%"},
            )
        except Exception as e:
            logger.error(f"[DB] Oracle company-name products lookup failed: {e}")
            return pd.DataFrame()
    with get_connection(db_path) as conn:
        return _lookup_by_company_name("customer_products", company_name, conn)


def get_customer_snapshot(tax_id: str, company_name: str = None,
                          db_path: Path = DB_PATH) -> dict:
    """
    Return the bank's latest view of a customer as plain dict + DataFrames.

    Returns:
      {
        "found": bool,
        "matched_by": "tax_id" | "company_name" | None,
        "financial_metrics": {en_description: value} for the LATEST period,
        "financial_period": "YYYYMM" or None,
        "product_flags": {product_key: 0/1},
        "nace_code": "47.11.02" or None,
        "financials_df": LONG DataFrame (all periods, tr + en descriptions),
        "product_flags_df": DataFrame,
      }
    """
    snapshot = {
        "found": False, "matched_by": None,
        "financial_metrics": {}, "financial_period": None,
        "product_flags": {}, "nace_code": None,
        "financials_df": pd.DataFrame(), "product_flags_df": pd.DataFrame(),
    }
    try:
        fin_df = get_financials_df(tax_id, db_path) if tax_id else pd.DataFrame()
        prod_df = get_product_flags_df(tax_id, db_path) if tax_id else pd.DataFrame()
        matched_by = "tax_id" if (not fin_df.empty or not prod_df.empty) else None

        if matched_by is None and company_name:
            fin_df = _financials_by_company(company_name, db_path)
            prod_df = _products_by_company(company_name, db_path)
            if not fin_df.empty or not prod_df.empty:
                matched_by = "company_name"

        if matched_by is None:
            logger.warning(
                f"Local DB: no record for tax_id='{tax_id}' / company='{company_name}'"
            )
            return snapshot

        snapshot["found"] = True
        snapshot["matched_by"] = matched_by
        snapshot["financials_df"] = fin_df
        snapshot["product_flags_df"] = prod_df

        if not fin_df.empty:
            latest_period = fin_df["period"].astype(str).max()
            latest = fin_df[fin_df["period"].astype(str) == latest_period]
            snapshot["financial_period"] = latest_period
            snapshot["financial_metrics"] = {
                row["en_description"]: float(row["value"])
                for _, row in latest.iterrows() if pd.notna(row["value"])
            }

        if not prod_df.empty:
            prow = prod_df.iloc[0]
            snapshot["product_flags"] = {
                c: int(prow[c]) for c in PRODUCT_FLAG_COLUMNS
                if c in prod_df.columns and pd.notna(prow[c])
            }
            if "nace_code" in prod_df.columns and pd.notna(prow["nace_code"]):
                snapshot["nace_code"] = str(prow["nace_code"]).strip() or None

        logger.info(
            f"Local DB: matched by {matched_by} — "
            f"{len(snapshot['financial_metrics'])} metrics "
            f"(period {snapshot['financial_period']}), "
            f"{sum(v == 1 for v in snapshot['product_flags'].values())} active product flags"
        )
        return snapshot
    except Exception as e:
        logger.error(f"Local DB read failed: {e}")
        return snapshot


def seed_demo_data(db_path: Path = DB_PATH):
    """Load demonstration rows so the pipeline can run end-to-end."""
    today_period = date.today().strftime("%Y%m")
    demo = [
        {
            "tax_id": "1234567890",
            "company_name": "MIZAN BEST",
            "nace_code": "47.11.02",  # Süpermarket perakende ticareti → Retail
            "metrics": {
                "Asit Test Oranı": 0.95,
                "Cari Oran": 1.40,
                "Brüt Kar marjı (%)": 20.1,
                "VAFÖK Marjı (%)": 13.4,
                "Net kar marjı (%)": 6.2,
                "Alacak Devir Süresi (Gün)": 78,
                "Borç Devir Süresi (Gün)": 64,
                "Stok Devir Süresi (Gün)": 60,
                "Nakit  Döngüsü (Gün)": 74,        # double space as in core system
                "İşletme Sermayesi": 14_000_000,
                "Maddi Özsermaye": 30_000_000,
                "KV Finansal Borç": 18_000_000,
                "UV Finansal Borç": 9_000_000,
                "Toplam Net Borç": 21_000_000,
                "Toplam Net Borç / VAFÖK ": 2.1,    # trailing space as in core system
                "Free Cash Flow": 5_200_000,
                "Finansal Kaldıraç": 1.8,
                "Özkaynak / Toplam Aktif (%)": 29.1,
                "Borç Servisi Karşılama Oranı (DSCR)": 1.3,
            },
            "flags": {
                "pos": 1, "virtual_pos": 0, "credit_card": 0, "checks": 1,
                "dbs": 0, "supplier_finance": 0, "leasing": 0, "factoring": 0,
                "trade_finance": 0, "fx": 0, "payroll": 0, "insurance": 0,
                "deposit": 1, "cash_loan": 0, "cash_management": 0,
                "letter_of_guarantee": 0,
            },
        },
        {
            "tax_id": "standalone",
            "company_name": "STANDALONE DEMO",
            "nace_code": "46.19.01",  # Çeşitli malların toptan ticareti → Trading
            "metrics": {
                "Asit Test Oranı": 1.05,
                "Cari Oran": 1.62,
                "Brüt Kar marjı (%)": 18.1,
                "VAFÖK Marjı (%)": 10.9,
                "Net kar marjı (%)": 4.8,
                "Alacak Devir Süresi (Gün)": 65,
                "Borç Devir Süresi (Gün)": 58,
                "Stok Devir Süresi (Gün)": 49,
                "Nakit  Döngüsü (Gün)": 56,
                "İşletme Sermayesi": 9_500_000,
                "Maddi Özsermaye": 26_000_000,
                "KV Finansal Borç": 11_000_000,
                "UV Finansal Borç": 4_000_000,
                "Toplam Net Borç": 12_500_000,
                "Toplam Net Borç / VAFÖK ": 1.6,
                "Free Cash Flow": 3_100_000,
                "Finansal Kaldıraç": 1.2,
                "Özkaynak / Toplam Aktif (%)": 40.0,
                "Borç Servisi Karşılama Oranı (DSCR)": 1.7,
            },
            "flags": {
                "pos": 0, "virtual_pos": 0, "credit_card": 1, "checks": 0,
                "dbs": 0, "supplier_finance": 0, "leasing": 0, "factoring": 0,
                "trade_finance": 0, "fx": 0, "payroll": 1, "insurance": 0,
                "deposit": 0, "cash_loan": 0, "cash_management": 0,
                "letter_of_guarantee": 0,
            },
        },
    ]
    for row in demo:
        upsert_financials(row["tax_id"], today_period, row["metrics"],
                          row["company_name"], db_path)
        upsert_product_flags(row["tax_id"], row["flags"],
                             row["company_name"], row.get("nace_code"), db_path)
    logger.info(f"Seeded {len(demo)} demo customers into {db_path}")


def ensure_db(db_path: Path = DB_PATH, seed_if_missing: bool = True):
    """Create the sqlite DB (and demo data) on first use so the pipeline
    never crashes. On the Oracle EDW backend this is a no-op — the
    warehouse schema and rows are managed by ETL, and demo data must
    never be seeded there."""
    if get_backend() == "oracle":
        logger.info("[DB] Oracle EDW backend active — skipping sqlite bootstrap/seeding")
        return
    first_time = not db_path.exists()
    get_connection(db_path).close()
    if first_time and seed_if_missing:
        seed_demo_data(db_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ap = argparse.ArgumentParser(description="Local customer DB utility")
    ap.add_argument("--seed", action="store_true", help="Create DB and load demo rows")
    ap.add_argument("--show", action="store_true", help="Print current DB contents")
    args = ap.parse_args()
    if args.seed:
        ensure_db(seed_if_missing=False)
        seed_demo_data()
    if args.show:
        print("── customer_financials (long format, TR + EN) ──")
        print(get_financials_df().to_string(index=False))
        print("\n── customer_products ──")
        print(get_product_flags_df().to_string(index=False))
    if not args.seed and not args.show:
        ap.print_help()
