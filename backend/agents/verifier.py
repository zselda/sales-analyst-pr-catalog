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
    "inventory_period":        {"required": ["150", "151", "152", "153", "620"], "forbidden": ["600"]},
    "cash_conversion_cycle":   {"required": ["120", "121", "150", "151", "152", "153", "320", "321", "600", "620"], "forbidden": ["300", "400", "500"]},
    "debt_to_equity":          {"required": ["400", "500", "570"], "forbidden": []},
    "bank_debt_ratio":         {"required": ["300", "309", "400"], "forbidden": ["320", "500"]},
    "financial_expense_ratio": {"required": ["780", "600"], "forbidden": ["620", "630"]},
    "pos_commission_ratio":    {"required": ["780.01", "600"], "forbidden": []},
    "insider_lending_ratio":   {"required": ["131"], "forbidden": ["331", "500"]},
    "check_risk_ratio":        {"required": ["102", "103"], "forbidden": ["300", "400"]},
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
 
        # ── LOCAL CHECK 1: Account Codes (Dynamic Existence Check) ──
        dynamic_mapping = ratios.get("dynamic_mapping", {})
        for name, rules in VALID_ACCOUNTS.items():
            # Rasyo hiç hesaplanmamışsa atla
            if name not in ratios:
                continue 
            used = ratios[name].get("accounts_used", [])
            used_str = [str(acc) for acc in used]
            # 1. ZORUNLU HESAP KONTROLÜ (Sadece Mizan'da varsa zorunludur)
            for req in rules["required"]:
                exists_in_mizan = any(str(k).startswith(req) for k in dynamic_mapping.keys())
                if exists_in_mizan:
                    if not any(acc.startswith(req) for acc in used_str):
                        errors.append(f"[{name}] Missing expected account '{req}'. (Hesap mizan'da mevcut ancak hesaplamaya dahil edilmemiş)")
                else:
                    logger.debug(f"[{name}] Account '{req}' not found in dynamic mapping. Bypassing requirement.")
            # 2. YASAKLI HESAP KONTROLÜ
            for acc in used_str:
                if any(acc.startswith(forb) for forb in rules["forbidden"]):
                    errors.append(f"[{name}] Forbidden account used '{acc}'")
 
 
        # ── LOCAL CHECK 2: Competitor Bank Mappings ──
        cb = ratios.get("competitor_banks", {})
        if cb:
            if not isinstance(cb.get("102"), list) or not isinstance(cb.get("300"), list) or not isinstance(cb.get("400"), list):
                errors.append("[competitor_banks] Missing or malformed 102, 300, or 400 array.")
 
        # ── LOCAL CHECK 2b: Hierarchy Consistency ──
        hierarchy = ratios.get("account_hierarchy", {})
        if hierarchy:
            for code, hdata in hierarchy.items():
                if isinstance(hdata, dict) and hdata.get("validation_status") == "mismatch":
                    logger.warning(
                        f"[hierarchy_warning] {code}-{hdata.get('hesap_kodu_aciklama', '?')}: "
                        f"leaf sum does not match reported parent value"
                    )
 
        # ── LOCAL CHECK 2c: Dynamic Mapping Validation ──
        if not dynamic_mapping:
            logger.warning("No dynamic mapping found — using static fallback?")
        else:
            logger.info(f"Dynamic mapping: {len(dynamic_mapping)} codes present")
 
        # ── Helper: extract numeric from raw_values (dict or float) ──
        def _raw_float(raw_val):
            if isinstance(raw_val, dict) and "value" in raw_val:
                return raw_val["value"]
            if isinstance(raw_val, (int, float)):
                return raw_val
            return 0
 
        # ── LOCAL CHECK 2d: Dönem (Temporal) Validation ──
        donem_ctx = ratios.get("donem_context", {})
        if donem_ctx:
            d_period_days = donem_ctx.get("period_days", 0)
            d_period_months = donem_ctx.get("period_months", 0)
 
            if not (1 <= d_period_months <= 12):
                errors.append(f"[donem] period_months={d_period_months} out of range [1-12]")
            if not (30 <= d_period_days <= 360):
                errors.append(f"[donem] period_days={d_period_days} out of range [30-360]")
 
            for time_ratio in ["collection_period", "payment_period", "inventory_period"]:
                if time_ratio in ratios:
                    ratio_data = ratios[time_ratio]
                    formula_str = ratio_data.get("formula", "")
                    embedded_days = ratio_data.get("period_days_used")
 
                    if embedded_days is not None and embedded_days != d_period_days:
                        errors.append(
                            f"[{time_ratio}] period_days mismatch: ratio used {embedded_days} "
                            f"but donem says {d_period_days}"
                        )
                    if "365" in formula_str:
                        errors.append(
                            f"[{time_ratio}] Formula contains hardcoded '365' "
                            f"but donem period_days={d_period_days}"
                        )
 
            logger.info(f"Dönem validation: {donem_ctx.get('label', '?')} ({d_period_days} days)")
        else:
            logger.warning("No donem_context found in ratios — skipping temporal validation")
 
        # ── LOCAL CHECK 4: Bounds (Esnetilmiş Sınırlar) ──
        bounds = {
            "gross_margin": (-5000, 100), 
            "operating_margin": (-5000, 500),
            "current_ratio": (0, 100),
            "quick_ratio": (0, 100),
            "collection_period": (0, 1800),
            "payment_period": (0, 1800),
            "inventory_period": (0, 1800),
            "cash_conversion_cycle": (-1800, 1800),
            "debt_to_equity": (-500, 500), 
            "bank_debt_ratio": (0, 150),
            "financial_expense_ratio": (0, 5000),
            "pos_commission_ratio": (0, 100),
            "insider_lending_ratio": (0, 100),
            "check_risk_ratio": (0, 10000)
        }
        for name, (lo, hi) in bounds.items():
            if name in ratios:
                v = ratios[name].get("value", 0)
                if v < lo or v > hi:
                    logger.warning(f"[{name}] {v} out of usual range [{lo},{hi}] - Please review.")
        # ── LLM CHECK 5: Semantic verification ──
        if not errors:
            try:
                skip_keys = ["llm_interpretation", "competitor_banks", "account_hierarchy",
                             "dynamic_mapping", "validation_warnings"]
                lines = "\n".join(
                    f"- {n}: {d.get('value')}{d.get('unit')} | Accounts: {d.get('accounts_used')} | {d.get('formula')}"
                    for n, d in ratios.items() if n not in skip_keys and isinstance(d, dict) and "value" in d
                )
 
                cb_lines = (
                    f"- Competitor Banks mapped: "
                    f"102 Present: {bool(cb.get('102'))}, "
                    f"300 Present: {bool(cb.get('300'))}, "
                    f"400 Present: {bool(cb.get('400'))}"
                )
 
                hierarchy_status = "No hierarchy data"
                validation_warnings = ratios.get("validation_warnings", [])
                if hierarchy:
                    mismatches = [c for c, h in hierarchy.items()
                                  if isinstance(h, dict) and h.get("validation_status") == "mismatch"]
                    if mismatches:
                        hierarchy_status = f"MISMATCHES in: {', '.join(mismatches)}"
                    else:
                        hierarchy_status = "All hierarchy sums validated"
 
                donem_line = ""
                if donem_ctx:
                    donem_line = (
                        f"- Dönem: {donem_ctx.get('raw', '?')} ({donem_ctx.get('period_months', '?')} months, "
                        f"{donem_ctx.get('period_days', '?')} days)\n"
                    )
 
                prompt = (
                    f"Verify these ratios and mappings:\n{lines}\n{cb_lines}\n"
                    f"- Hierarchy Validation: {hierarchy_status}\n"
                    f"- Dynamic Mapping: {len(dynamic_mapping)} account codes mapped from document\n"
                    f"- Validation Warnings: {len(validation_warnings)}\n"
                    f"{donem_line}\n"
                    "Check account codes per Turkish Chart of Accounts. "
                    "Verify hierarchy consistency, dynamic mapping completeness, "
                    "and temporal period alignment. "
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
    status = state.get("verification_status", "rejected")
    retries = state.get("retry_count", 0)
    if status == "approved":
        logger.info("[Router] → product_analyst")
        return "product_analyst"
    if retries >= MAX_RETRIES:
        logger.info("[Router] Max retries → continue")
        return "product_analyst"
    logger.info(f"[Router] REJECTED → retry #{retries}")
    return "quant_analyst"
