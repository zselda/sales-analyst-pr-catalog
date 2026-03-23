"""
Evaluation Framework Tests
============================
Tests for the multi-dimensional evaluation rubric and output validators.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.evaluation import (
    validate_standardized_mizan,
    validate_financial_ratios,
    validate_network_data,
    validate_strategy_report,
    rubric,
)


class TestOutputValidators:
    def test_mizan_valid(self):
        data = [{"account_code": "100", "account_name": "Cash", "debit": 100, "credit": 0, "category": "Current Assets"}]
        result = validate_standardized_mizan(data)
        assert result["valid"] is True

    def test_mizan_empty(self):
        result = validate_standardized_mizan([])
        assert result["valid"] is False

    def test_mizan_missing_keys(self):
        result = validate_standardized_mizan([{"account_code": "100"}])
        assert result["valid"] is False

    def test_ratios_valid(self):
        data = {
            "gross_margin": {"value": 30, "unit": "%", "formula": "test", "accounts_used": ["600"]},
            "current_ratio": {"value": 2, "unit": "x", "formula": "test", "accounts_used": []},
            "debt_to_equity": {"value": 1.5, "unit": "x", "formula": "test", "accounts_used": []},
            "financial_expense_ratio": {"value": 3, "unit": "%", "formula": "test", "accounts_used": []},
            "pos_commission_ratio": {"value": 1, "unit": "%", "formula": "test", "accounts_used": []},
        }
        result = validate_financial_ratios(data)
        assert result["valid"] is True

    def test_ratios_missing_ratio(self):
        data = {"gross_margin": {"value": 30, "unit": "%", "formula": "test", "accounts_used": ["600"]}}
        result = validate_financial_ratios(data)
        assert result["valid"] is False


    def test_network_valid(self):
        data = {
            "nodes": [{"id": "1", "label": "Test"}],
            "edges": [],
            "stats": {},
        }
        result = validate_network_data(data)
        assert result["valid"] is True

    def test_network_empty_nodes(self):
        data = {"nodes": [], "edges": [], "stats": {}}
        result = validate_network_data(data)
        assert result["valid"] is False

    def test_strategy_valid(self):
        result = validate_strategy_report("x" * 200)
        assert result["valid"] is True

    def test_strategy_too_short(self):
        result = validate_strategy_report("short")
        assert result["valid"] is False


class TestRubricScoring:
    def test_full_pipeline_scores(self):
        """Test scoring with complete pipeline state."""
        state = {
            "standardized_mizan": [{"account_code": "100"}],
            "financial_ratios": {
                "gross_margin": {"value": 30},
                "current_ratio": {"value": 2},
                "debt_to_equity": {"value": 1.5},
                "financial_expense_ratio": {"value": 3},
                "pos_commission_ratio": {"value": 1},
            },
            "network_data": {
                "nodes": [{"type": "customer"}, {"type": "supplier"}, {"type": "bank"}],
                "edges": [{"source": "a", "target": "b"}],
                "stats": {"customer_count": 1, "supplier_count": 1, "bank_count": 1},
            },
            "strategy_report": "# Report\n₺100,000 in POS factoring cash management\n" + "x" * 200,
        }

        result = rubric.score_pipeline(state)
        assert "scores" in result
        assert "overall_score" in result
        assert "passed" in result
        assert 0 <= result["overall_score"] <= 1.0
        assert result["passed"] is True

    def test_empty_pipeline_fails(self):
        """Test that empty state fails evaluation."""
        result = rubric.score_pipeline({})
        assert result["passed"] is False
        assert result["overall_score"] < 0.7
