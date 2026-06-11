"""
Local Customer Database Layer (SQLite)
=======================================
Stores the bank's own latest view of each customer:
1. customer_financials — pre-computed financial metrics per period
   (acid-test ratio, gross profit, etc.) from the bank's core systems.
2. customer_products  — current product usage flags
   (e.g., pos=1, credit_card=0, checks=1).

The db_enrichment agent reads this data and exposes it to the pipeline
as DataFrames + plain dictionaries so quant_analyst / product_analyst /
strategist prompts can be grounded in the LATEST bank-side data.

CLI:
    python local_db.py --seed     # create DB and load demo rows
    python local_db.py --show     # dump current contents
"""

import argparse
import logging
import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger("swarm.local_db")

DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "customer_db.sqlite"

# Product flag columns. Keys MUST match few_shot_library product_keys.
PRODUCT_FLAG_COLUMNS = [
    "pos", "virtual_pos", "credit_card", "checks", "dbs",
    "supplier_finance", "leasing", "factoring", "trade_finance",
    "fx", "payroll", "insurance", "deposit", "cash_loan",
    "cash_management", "letter_of_guarantee",
]

# Financial metric columns stored per customer/period
FINANCIAL_METRIC_COLUMNS = [
    "acid_test_ratio", "current_ratio", "gross_profit", "net_revenue",
    "gross_margin_pct", "operating_margin_pct", "debt_to_equity",
    "total_assets", "total_equity", "working_capital", "ebitda",
    "collection_period_days", "payment_period_days",
]

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS customer_financials (
    tax_id          TEXT NOT NULL,
    company_name    TEXT,
    period          TEXT NOT NULL,
    {', '.join(f'{c} REAL' for c in FINANCIAL_METRIC_COLUMNS)},
    updated_at      TEXT DEFAULT (date('now')),
    PRIMARY KEY (tax_id, period)
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
    """Add columns introduced after initial deployments (idempotent)."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(customer_products)")}
    if cols and "nace_code" not in cols:
        conn.execute("ALTER TABLE customer_products ADD COLUMN nace_code TEXT")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    _migrate(conn)
    return conn


def upsert_financials(tax_id: str, period: str, metrics: dict,
                      company_name: str = None, db_path: Path = DB_PATH):
    """Insert or update a financial metrics row for (tax_id, period)."""
    cols = [c for c in FINANCIAL_METRIC_COLUMNS if c in metrics]
    with get_connection(db_path) as conn:
        conn.execute(
            f"INSERT INTO customer_financials (tax_id, company_name, period, {', '.join(cols)}) "
            f"VALUES (?, ?, ?, {', '.join('?' for _ in cols)}) "
            f"ON CONFLICT(tax_id, period) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in cols + ["company_name"]),
            [tax_id, company_name, period] + [metrics[c] for c in cols],
        )


def upsert_product_flags(tax_id: str, flags: dict, company_name: str = None,
                         nace_code: str = None, db_path: Path = DB_PATH):
    """Insert or update product usage flags (and NACE code) for a customer."""
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
    """All financial metric rows (optionally for one customer), newest period first."""
    with get_connection(db_path) as conn:
        if tax_id:
            return pd.read_sql_query(
                "SELECT * FROM customer_financials WHERE tax_id = ? ORDER BY period DESC",
                conn, params=[tax_id],
            )
        return pd.read_sql_query(
            "SELECT * FROM customer_financials ORDER BY tax_id, period DESC", conn
        )


def get_product_flags_df(tax_id: str = None, db_path: Path = DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        if tax_id:
            return pd.read_sql_query(
                "SELECT * FROM customer_products WHERE tax_id = ?", conn, params=[tax_id]
            )
        return pd.read_sql_query("SELECT * FROM customer_products", conn)


def _lookup_by_company_name(table: str, company_name: str, conn) -> pd.DataFrame:
    """Fallback fuzzy lookup when tax_id is unknown (e.g., standalone runs)."""
    if not company_name:
        return pd.DataFrame()
    return pd.read_sql_query(
        f"SELECT * FROM {table} WHERE UPPER(company_name) LIKE ?",
        conn, params=[f"%{company_name.strip().upper()}%"],
    )


def get_customer_snapshot(tax_id: str, company_name: str = None,
                          db_path: Path = DB_PATH) -> dict:
    """
    Return the bank's latest view of a customer as plain dict + DataFrames.

    Returns:
      {
        "found": bool,
        "matched_by": "tax_id" | "company_name" | None,
        "financial_metrics": {metric: value} for the LATEST period,
        "financial_period": "YYYYMM" or None,
        "product_flags": {product_key: 0/1},
        "nace_code": "47.11.02" or None,
        "financials_df": DataFrame (all periods),
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
        with get_connection(db_path) as conn:
            fin_df = get_financials_df(tax_id, db_path) if tax_id else pd.DataFrame()
            prod_df = get_product_flags_df(tax_id, db_path) if tax_id else pd.DataFrame()
            matched_by = "tax_id" if (not fin_df.empty or not prod_df.empty) else None

            if matched_by is None and company_name:
                fin_df = _lookup_by_company_name("customer_financials", company_name, conn)
                prod_df = _lookup_by_company_name("customer_products", company_name, conn)
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
            fin_df = fin_df.sort_values("period", ascending=False)
            latest = fin_df.iloc[0]
            snapshot["financial_period"] = str(latest["period"])
            snapshot["financial_metrics"] = {
                c: float(latest[c]) for c in FINANCIAL_METRIC_COLUMNS
                if c in fin_df.columns and pd.notna(latest[c])
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
                "acid_test_ratio": 0.95, "current_ratio": 1.40,
                "gross_profit": 18_500_000, "net_revenue": 92_000_000,
                "gross_margin_pct": 20.1, "operating_margin_pct": 9.4,
                "debt_to_equity": 1.8, "total_assets": 110_000_000,
                "total_equity": 32_000_000, "working_capital": 14_000_000,
                "ebitda": 12_300_000,
                "collection_period_days": 78, "payment_period_days": 64,
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
                "acid_test_ratio": 1.05, "current_ratio": 1.62,
                "gross_profit": 9_800_000, "net_revenue": 54_000_000,
                "gross_margin_pct": 18.1, "operating_margin_pct": 7.2,
                "debt_to_equity": 1.2, "total_assets": 70_000_000,
                "total_equity": 28_000_000, "working_capital": 9_500_000,
                "ebitda": 6_900_000,
                "collection_period_days": 65, "payment_period_days": 58,
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
    """Create DB (and demo data) on first use so the pipeline never crashes."""
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
        print("── customer_financials ──")
        print(get_financials_df().to_string(index=False))
        print("\n── customer_products ──")
        print(get_product_flags_df().to_string(index=False))
    if not args.seed and not args.show:
        ap.print_help()
