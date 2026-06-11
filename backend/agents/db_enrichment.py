"""
Agent: Local DB Enrichment
============================
Reads the bank's local customer database (SQLite) and injects the
LATEST bank-side view of the customer into the pipeline state:

- db_financial_metrics: dict of pre-computed metrics
  (acid_test_ratio, gross_profit, current_ratio, ...) for the latest period
- db_product_flags: dict of current product usage flags
  (e.g., {"pos": 1, "credit_card": 0, "checks": 1})
- db_meta: lookup metadata (found / matched_by / period) plus NACE-based
  sector verification (see below)

NACE SECTOR VERIFICATION:
The customer's NACE code (İTO 6-digit classification) stored in the DB is
mapped to a TCMB benchmark sector via sector_analysis.sector_from_nace and
compared against the LLM sector prediction. The predictor itself stays
unchanged — this agent only verifies, logs mismatches, and fills the
sector in when the prediction is absent.

Downstream consumers:
- quant_analyst   → cross-checks Mizan-derived ratios against DB metrics
- product_analyst → suppresses recommendations for products already used
- strategist      → exclusion list + freshness context
"""

import logging

from agents.base import BaseAgent
from local_db import ensure_db, get_customer_snapshot
from sector_analysis import sector_from_nace, match_sectors

logger = logging.getLogger("swarm.agents.db_enrichment")


class LocalDBAgent(BaseAgent):
    name = "db_enrichment"
    description = "Load latest customer financial metrics and product usage flags from the local bank DB"
    required_inputs = ["tax_id"]
    output_keys = ["db_financial_metrics", "db_product_flags", "db_meta"]

    def execute(self, state: dict) -> dict:
        tax_id = state.get("tax_id", "")
        company_name = state.get("company_name", "")

        ensure_db()  # creates + seeds demo data on first run
        snapshot = get_customer_snapshot(tax_id, company_name)

        meta = {
            "found": snapshot["found"],
            "matched_by": snapshot["matched_by"],
            "financial_period": snapshot["financial_period"],
        }

        if snapshot["found"]:
            active = [k for k, v in snapshot["product_flags"].items() if v == 1]
            logger.info(
                f"💾 DB enrichment: matched_by={meta['matched_by']}, "
                f"period={meta['financial_period']}, active products={active}"
            )
        else:
            logger.warning(
                "💾 DB enrichment: customer not found in local DB — "
                "pipeline continues with Mizan-only evidence"
            )

        output = {
            "db_financial_metrics": snapshot["financial_metrics"],
            "db_product_flags": snapshot["product_flags"],
            "db_meta": meta,
        }

        # ── NACE-BASED SECTOR VERIFICATION ──
        # The LLM predictor remains the primary source; the DB NACE code is
        # used to VERIFY it (and to fill in when the prediction is absent).
        predicted_sector = state.get("sector") or ""
        nace_code = snapshot.get("nace_code")
        nace_sector = sector_from_nace(nace_code) if nace_code else None
        verification = {
            "nace_code": nace_code,
            "nace_sector": nace_sector,
            "predicted_sector": predicted_sector or None,
            "verified": None,  # None = cannot verify (no NACE data)
        }

        if nace_sector:
            match_info = match_sectors(predicted_sector)
            predicted_set = (
                [] if match_info["is_fallback"]
                else [match_info["primary"]] + match_info["secondary"]
            )
            if not predicted_set:
                # Prediction absent/unmatched → NACE fills the gap
                verification["verified"] = False
                logger.warning(
                    f"🏷️ NACE verification: sector prediction absent/unmatched "
                    f"('{predicted_sector}') — using DB NACE {nace_code} → "
                    f"'{nace_sector}' as the sector"
                )
                output["sector"] = nace_sector
            elif nace_sector in predicted_set:
                verification["verified"] = True
                logger.info(
                    f"🏷️ NACE verification: predicted sector '{predicted_sector}' "
                    f"CONFIRMED by DB NACE {nace_code} → '{nace_sector}'"
                )
            else:
                verification["verified"] = False
                logger.warning(
                    f"🏷️ NACE verification MISMATCH: predictor says "
                    f"'{predicted_sector}' but DB NACE {nace_code} maps to "
                    f"'{nace_sector}' — review sector assignment "
                    f"(prediction kept, NACE sector reported for context)"
                )
        else:
            logger.info(
                "🏷️ NACE verification skipped: no NACE code in local DB for this customer"
            )

        meta["sector_verification"] = verification
        return output


# Module-level callable for LangGraph
db_enrichment_agent = LocalDBAgent()
