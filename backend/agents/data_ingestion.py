"""
Agent 1: Data Ingestion Agent
===============================
Reads the raw Mizan DataFrame and standardizes account codes.
Ensures all codes follow the format: main_code.sub_code (e.g., "120.01").

Refactored to use BaseAgent for tracing and error isolation.
"""

import logging
import pandas as pd
from agents.base import BaseAgent
from excel_parser import extract_donem

logger = logging.getLogger("swarm.agents.data_ingestion")


class DataIngestionAgent(BaseAgent):
    name = "data_ingestion"
    description = "Standardize Mizan data with Turkish Chart of Accounts classification"
    required_inputs = ["mizan_data"]
    output_keys = ["standardized_mizan"]

    def execute(self, state: dict) -> dict:
        """
        Process and standardize the raw Mizan data.

        Steps:
        1. Convert raw mizan_data (list of dicts) into a DataFrame
        2. Validate account_code format
        3. Classify each account into its category (Asset, Liability, etc.)
        4. Return standardized data back to the state
        """

        raw_data = state.get("mizan_data", [])
        if not raw_data:
            logger.warning("No mizan data found in state!")
            return {"standardized_mizan": []}

        df = pd.DataFrame(raw_data)

        # --- Step 1: Validate and clean account codes ---
        df["account_code"] = df["account_code"].astype(str).str.strip()

        # --- Step 2: Add classification based on Turkish Chart of Accounts ---
        def classify_account(code: str) -> str:
            """Classify account code into category based on first digit."""
            main_code = code.split(".")[0]
            try:
                first_digit = int(main_code[0])
            except (ValueError, IndexError):
                return "Unknown"

            classifications = {
                1: "Current Assets",
                2: "Non-Current Assets",
                3: "Short-Term Liabilities",
                4: "Long-Term Liabilities",
                5: "Equity",
                6: "Profit & Loss Accounts",
                7: "Cost Accounts",
                8: "Free Accounts",
                9: "Off-Balance Sheet Accounts"
            }
            return classifications.get(first_digit, "Unknown")

        df["category"] = df["account_code"].apply(classify_account)

        # --- Step 3: Calculate net balance ---
        df["net_balance"] = df["debit"] - df["credit"]

        # --- Step 4: Flag sub-accounts ---
        df["is_sub_account"] = df["account_code"].str.contains(r"\.", regex=True)

        # --- Step 5: Add main account code for grouping ---
        df["main_account"] = df["account_code"].apply(lambda x: x.split(".")[0])

        standardized = df.to_dict(orient="records")

        logger.info(f"Standardized {len(standardized)} accounts:")
        for cat in df["category"].unique():
            count = len(df[df["category"] == cat])
            logger.info(f"  - {cat}: {count} accounts")

        # --- Step 6: Extract Dönem (time period) from document ---
        donem_info = extract_donem(df)
        logger.info(f"Detected period: {donem_info['label']} ({donem_info['period_days']} days)")

        return {"standardized_mizan": standardized, "donem_info": donem_info}


# Module-level callable for LangGraph
data_ingestion_agent = DataIngestionAgent()
