"""
Agent Evaluation Framework
============================
Multi-dimensional evaluation for agent outputs with rubric scoring,
output validation, and pipeline quality gates.

Based on evaluation skill patterns:
- Multi-dimensional rubrics (accuracy, completeness, efficiency)
- Output schema validation
- Pipeline-level quality gate
"""

import logging
from typing import Any, Optional

logger = logging.getLogger("swarm.evaluation")


# ============================================================================
# Output Validators — schema checks for each agent's output
# ============================================================================

def validate_standardized_mizan(output: list) -> dict:
    """Validate data_ingestion agent output."""
    errors = []
    if not isinstance(output, list):
        return {"valid": False, "errors": ["standardized_mizan must be a list"]}
    if len(output) == 0:
        return {"valid": False, "errors": ["standardized_mizan is empty"]}

    required_keys = {"account_code", "account_name", "debit", "credit", "category"}
    for i, row in enumerate(output[:5]):  # Check first 5
        missing = required_keys - set(row.keys())
        if missing:
            errors.append(f"Row {i} missing keys: {missing}")

    return {"valid": len(errors) == 0, "errors": errors}


def validate_financial_ratios(output: dict) -> dict:
    """Validate quant_analyst agent output."""
    errors = []
    if not isinstance(output, dict):
        return {"valid": False, "errors": ["financial_ratios must be a dict"]}

    required_ratios = [
        "gross_margin", "current_ratio", "debt_to_equity",
        "financial_expense_ratio", "pos_commission_ratio",
    ]
    for ratio_name in required_ratios:
        if ratio_name not in output:
            errors.append(f"Missing ratio: {ratio_name}")
            continue
        ratio = output[ratio_name]
        if not isinstance(ratio, dict):
            errors.append(f"{ratio_name} must be a dict")
            continue
        for key in ["value", "unit", "formula", "accounts_used"]:
            if key not in ratio:
                errors.append(f"{ratio_name} missing '{key}'")

    return {"valid": len(errors) == 0, "errors": errors}




def validate_network_data(output: dict) -> dict:
    """Validate network_mapper agent output."""
    errors = []
    if not isinstance(output, dict):
        return {"valid": False, "errors": ["network_data must be a dict"]}

    for key in ["nodes", "edges", "stats"]:
        if key not in output:
            errors.append(f"Missing key: {key}")

    nodes = output.get("nodes", [])
    if len(nodes) == 0:
        errors.append("No nodes in network graph")

    return {"valid": len(errors) == 0, "errors": errors}


def validate_strategy_report(output: str) -> dict:
    """Validate strategist agent output."""
    errors = []
    if not isinstance(output, str):
        return {"valid": False, "errors": ["strategy_report must be a string"]}
    if len(output) < 100:
        errors.append(f"Report too short ({len(output)} chars, minimum 100)")

    return {"valid": len(errors) == 0, "errors": errors}


# ============================================================================
# Rubric Scoring — multi-dimensional quality assessment
# ============================================================================

class EvaluationRubric:
    """
    Multi-dimensional rubric for scoring agent pipeline output.

    Dimensions:
    - data_completeness: All expected outputs present and valid
    - ratio_accuracy: Financial ratios within expected bounds
    - behavioral_depth: Transaction analysis covers key metrics
    - network_coverage: Network graph includes all counterparties
    - strategy_quality: Report cites data and provides actionable items
    """

    def __init__(self):
        self.dimensions: dict[str, float] = {}
        self.weights = {
            "data_completeness": 0.30,
            "ratio_accuracy": 0.25,
            "network_coverage": 0.20,
            "strategy_quality": 0.25,
        }

    def score_pipeline(self, state: dict) -> dict:
        """
        Score the complete pipeline output across all dimensions.

        Args:
            state: Final swarm state after pipeline completion

        Returns:
            dict with per-dimension scores, overall score, and pass/fail
        """
        scores = {}

        # 1. Data Completeness
        data_checks = [
            state.get("standardized_mizan") is not None,
            state.get("financial_ratios") is not None,
            state.get("network_data") is not None,
            state.get("strategy_report") is not None,
        ]
        scores["data_completeness"] = sum(data_checks) / len(data_checks)

        # 2. Ratio Accuracy (check bounds)
        ratios = state.get("financial_ratios", {})
        bounds = {
            "gross_margin": (-50, 100),
            "current_ratio": (0, 50),
            "debt_to_equity": (0, 100),
            "financial_expense_ratio": (0, 50),
            "pos_commission_ratio": (0, 20),
        }
        ratio_checks = []
        for name, (lo, hi) in bounds.items():
            if name in ratios and isinstance(ratios[name], dict):
                val = ratios[name].get("value", 0)
                ratio_checks.append(lo <= val <= hi)
            else:
                ratio_checks.append(False)
        scores["ratio_accuracy"] = sum(ratio_checks) / len(ratio_checks) if ratio_checks else 0


        # 4. Network Coverage
        network = state.get("network_data", {})
        stats = network.get("stats", {})
        network_checks = [
            stats.get("customer_count", 0) > 0,
            stats.get("supplier_count", 0) > 0,
            stats.get("bank_count", 0) > 0,
            len(network.get("edges", [])) > 0,
        ]
        scores["network_coverage"] = sum(network_checks) / len(network_checks)

        # 5. Strategy Quality
        report = state.get("strategy_report", "")
        if not isinstance(report, str):
            report = str(report) if report else ""
        strategy_checks = [
            len(report) > 200,
            "₺" in report or "TL" in report,  # cites monetary values
            "#" in report,  # uses markdown headers
            any(word in report.lower() for word in ["pos", "factoring", "cash"]),  # product mentions
        ]
        scores["strategy_quality"] = sum(strategy_checks) / len(strategy_checks)

        # Overall weighted score
        overall = sum(scores[dim] * self.weights[dim] for dim in scores)

        result = {
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "overall_score": round(overall, 3),
            "passed": overall >= 0.7,
            "threshold": 0.7,
        }

        logger.info(f"[Evaluation] Overall score: {result['overall_score']} | "
                     f"{'PASSED ✅' if result['passed'] else 'FAILED ❌'}")

        return result


# Singleton rubric instance
rubric = EvaluationRubric()


# ============================================================================
# Validators Registry
# ============================================================================

VALIDATORS = {
    "standardized_mizan": validate_standardized_mizan,
    "financial_ratios": validate_financial_ratios,
    "network_data": validate_network_data,
    "strategy_report": validate_strategy_report,
}


def validate_agent_output(key: str, value: Any) -> dict:
    """
    Validate an agent's output using the appropriate validator.

    Args:
        key: The state key being validated
        value: The output value to validate

    Returns:
        dict with 'valid' (bool) and 'errors' (list)
    """
    validator = VALIDATORS.get(key)
    if validator is None:
        return {"valid": True, "errors": []}
    return validator(value)
