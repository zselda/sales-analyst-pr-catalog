"""
Agent 3: Verifier — Local validation + LLM semantic check

Refactored to use BaseAgent for tracing and error isolation.
"""

import logging
from agents.base import BaseAgent
from agents.state import SwarmState
from llm_config import invoke_llm, VERIFIER_SYSTEM_PROMPT

logger = logging.getLogger("swarm.agents.verifier")

MAX_RETRIES = 3

# Enforcing strict Tekdüzen rules based on the new Quant calculations
VALID_ACCOUNTS = {
    "gross_margin":            {"required": ["600", "620"], "forbidden": ["630", "760", "780"]},
    "operating_margin":        {"required": ["600", "620", "630", "631", "632"], "forbidden": ["780", "770"]},
    "current_ratio":           {"required": [], "forbidden": ["255", "257", "400"]},
    "quick_ratio":             {"required": ["150", "151", "152", "153"], "forbidden": ["255", "400"]},
    "collection_period":       {"required": ["120", "121", "600"], "forbidden": ["320", "321"]},
    "payment_period":          {"required": ["320", "321", "620"], "forbidden": ["120", "121"]},
    "debt_to_equity":          {"required": ["400", "500", "570"], "forbidden": []},
    "bank_debt_ratio":         {"required": ["300", "309", "400"], "forbidden": ["320", "500"]},
    "financial_expense_ratio": {"required": ["780", "600"], "forbidden": ["620", "630"]},
    "pos_commission_ratio":    {"required": ["780.01", "600"], "forbidden": []},
}


class VerifierAgent(BaseAgent):
    name = "verifier"
    description = "Validate financial ratios against Turkish Chart of Accounts rules"
    required_inputs = ["financial_ratios"]
    output_keys = ["verification_status", "verification_errors"]

    def execute(self, state: dict) -> dict:
        retry_count = state.get("retry_count", 0)
        ratios = state.get("financial_ratios", {})
        logger.info(f"Reviewing (attempt #{retry_count})...")

        if not ratios or "error" in ratios:
            return {"verification_status": "rejected", "verification_errors": "No ratios found."}

        errors = []

        # ── LOCAL CHECK 1: Account codes ──
        for name, rules in VALID_ACCOUNTS.items():
            if name not in ratios:
                errors.append(f"Missing ratio: {name}")
                continue
            used = ratios[name].get("accounts_used", [])
            for req in rules["required"]:
                if req not in used:
                    errors.append(f"[{name}] Missing required account '{req}'")
            for acc in used:
                if acc.split(".")[0] in rules["forbidden"] or acc in rules["forbidden"]:
                    errors.append(f"[{name}] Forbidden account '{acc}'")

        # ── LOCAL CHECK 2: Competitor Bank Mappings ──
        cb = ratios.get("competitor_banks", {})
        if cb:
            if not isinstance(cb.get("102"), list) or not isinstance(cb.get("300"), list) or not isinstance(cb.get("400"), list):
                errors.append("[competitor_banks] Missing or malformed 102, 300, or 400 array.")

        # ── LOCAL CHECK 3: Math consistency ──
        def _raw_float(raw_val):
            """Extract numeric value from raw_values (supports dict or float format)."""
            if isinstance(raw_val, dict) and "value" in raw_val:
                return raw_val["value"]
            if isinstance(raw_val, (int, float)):
                return raw_val
            return 0

        if "gross_margin" in ratios:
            raw = ratios["gross_margin"].get("raw_values", {})
            r, c = _raw_float(raw.get("revenue_600", 0)), _raw_float(raw.get("cogs_620", 0))
            if r > 0:
                expected = round((r - c) / r * 100, 2)
                if abs(expected - ratios["gross_margin"]["value"]) > 0.01:
                    errors.append(f"[gross_margin] Math: expected {expected}%")

        # ── LOCAL CHECK 4: Bounds ──
        bounds = {
            "gross_margin": (-50, 100), 
            "operating_margin": (-100, 100),
            "current_ratio": (0, 50),
            "quick_ratio": (0, 50),
            "collection_period": (0, 1000), # Days can be high
            "payment_period": (0, 1000),    # Days can be high
            "debt_to_equity": (0, 100), 
            "bank_debt_ratio": (0, 200),
            "financial_expense_ratio": (0, 100),
            "pos_commission_ratio": (0, 50)
        }
        for name, (lo, hi) in bounds.items():
            if name in ratios:
                v = ratios[name].get("value", 0)
                if v < lo or v > hi:
                    errors.append(f"[{name}] {v} out of range [{lo},{hi}]")

        # ── LLM CHECK 5: Semantic verification ──
        if not errors:
            try:
                # Exclude interpretation and competitor banks from standard formatting to avoid dict unpacking errors
                lines = "\n".join(
                    f"- {n}: {d.get('value')}{d.get('unit')} | Accounts: {d.get('accounts_used')} | {d.get('formula')}"
                    for n, d in ratios.items() if n not in ["llm_interpretation", "competitor_banks"]
                )
                
                cb_lines = (
                    f"- Competitor Banks mapped: "
                    f"102 Present: {bool(cb.get('102'))}, "
                    f"300 Present: {bool(cb.get('300'))}, "
                    f"400 Present: {bool(cb.get('400'))}"
                )

                prompt = (
                    f"Verify these ratios and mappings:\n{lines}\n{cb_lines}\n\n"
                    "Check account codes per Turkish Chart of Accounts. "
                    "Respond APPROVED or REJECTED (with reason)."
                )
                verdict = invoke_llm(VERIFIER_SYSTEM_PROMPT, prompt, temperature=0.1, max_tokens=256)
                self.metrics.record_llm_call(tokens=len(verdict.split()))
                logger.info(f"LLM: {verdict[:80]}")
                if "REJECTED" in verdict.upper():
                    errors.append(f"LLM: {verdict}")
            except Exception as e:
                logger.warning(f"LLM skipped: {e}")

        # ── DECISION ──
        if errors:
            msg = "\n".join(f"  ❌ {e}" for e in errors)
            logger.warning(f"FAILED:\n{msg}")
            if retry_count >= MAX_RETRIES:
                return {"verification_status": "approved", "verification_errors": f"FORCED. {msg}"}
            return {"verification_status": "rejected", "verification_errors": msg}

        logger.info("✅ APPROVED")
        return {"verification_status": "approved", "verification_errors": ""}


# Module-level callable for LangGraph
verifier_agent = VerifierAgent()


def should_retry_or_continue(state: SwarmState) -> str:
    """Router function for the verifier conditional edge."""
    status = state.get("verification_status", "rejected")
    retries = state.get("retry_count", 0)
    if status == "approved":
        logger.info("[Router] → network_mapper")
        return "network_mapper"
    if retries >= MAX_RETRIES:
        logger.info("[Router] Max retries → continue")
        return "network_mapper"
    logger.info(f"[Router] REJECTED → retry #{retries}")
    return "quant_analyst"